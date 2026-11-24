"""RumiFold: Qwen3-8B with LoRA that folds a protein in its own hidden states and answers questions about it.

Folding.  ESM-C residue features are projected into Qwen's embedding space and run through the 36 blocks with
bidirectional attention. The final states are fed back once (recycling) and a linear head reads one backbone frame
per residue from the second pass.

Language.  The text pass attends to the keys and values that the second folding pass computed, so nothing is re-run
and no structure is written out. An address such as "A/57" in the prompt or reply is followed by that residue's state.
"""
from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import torch
import torch.utils.checkpoint
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .addresses import AddressParser
from .hub import DEFAULT_MODELS
from .refiner import ConfidenceHead, Refiner

TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
SINK = 151644                 # <|im_start|>: the first token of every chat turn, kept in front of the residues
STOP = (151645, 151643)       # <|im_end|>, <|endoftext|>

Fold = namedtuple("Fold", "backbone rotation translation states hidden positions rows plddt", defaults=[None])


class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.normal_(self.a.weight, std=.02); nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + (self.b(self.a(x.float())) * self.scale).to(x.dtype)


class PointerHead(nn.Module):
    """Scores every residue row against the text state at the "/" of an address the model is writing."""

    def __init__(self, width, dim=256):
        super().__init__()
        self.q, self.k, self.scale = nn.Linear(width, dim, bias=False), nn.Linear(width, dim, bias=False), dim ** -0.5

    def forward(self, h, rows):
        return (self.q(h.float()) @ self.k(rows.float()).T) * self.scale


def rotation(x):
    """Six numbers -> rotation matrix (Gram-Schmidt), with fallbacks for degenerate vectors."""
    a, b = x[..., :3].float(), x[..., 3:].float()
    fallback = torch.zeros_like(a); fallback[..., 0] = 1
    a = torch.where((a.norm(dim=-1) > 1e-6)[..., None], a, fallback)
    e1 = nn.functional.normalize(a, dim=-1)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    axis = nn.functional.one_hot(e1.abs().argmin(-1), 3).float()
    fallback = axis - (axis * e1).sum(-1, keepdim=True) * e1
    b = torch.where((b.norm(dim=-1) > 1e-6)[..., None], b, fallback)
    e2 = nn.functional.normalize(b, dim=-1)
    return torch.stack((e1, e2, torch.linalg.cross(e1, e2, dim=-1)), -1)


class Folder(nn.Module):
    def __init__(self, qwen, rank, alpha, gap):
        super().__init__()
        self.qwen, self.gap = qwen, gap
        self.first_position = 1                                   # position 0 belongs to the sink token of the text pass
        qwen.requires_grad_(False)
        for module in list(qwen.modules()):
            for name, child in list(module.named_children()):
                if name in TARGETS and isinstance(child, nn.Linear):
                    setattr(module, name, LoRA(child, rank, alpha))
        w = qwen.config.hidden_size
        self.bridge = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, w), nn.LayerNorm(w))
        self.chain_start = nn.Parameter(torch.zeros(w))
        self.recycle_norm, self.recycle_gate = nn.LayerNorm(w), nn.Parameter(torch.zeros(()))
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 9))
        nn.init.normal_(self.head[-1].weight, std=.001); nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad(): self.head[-1].bias[3] = 1; self.head[-1].bias[7] = 1
        emb = qwen.get_input_embeddings().weight.detach()
        scale = (sum(p.float().square().sum(dtype=torch.float64) for p in emb.split(4096)) / emb.numel()).sqrt().float()
        self.register_buffer("embedding_scale", scale, persistent=False)
        self.register_buffer("template", torch.tensor([[.5256, 1.3612, 0.], [0., 0., 0.], [-1.5251, 0., 0.]]), persistent=False)

    def inputs(self, esm, chain_id):
        starts = torch.ones_like(chain_id, dtype=torch.bool); starts[1:] = chain_id[1:] != chain_id[:-1]
        return (self.bridge(esm.float()) + starts[:, None] * self.chain_start) * self.embedding_scale

    def recycle(self, z):
        return self.embedding_scale * self.recycle_gate.tanh() * self.recycle_norm(z.float())

    def positions(self, chain_id, residue_index):
        out, offset = torch.empty_like(residue_index), self.first_position
        for chain in torch.unique_consecutive(chain_id):
            mask = chain_id == chain
            out[mask] = residue_index[mask] + offset
            offset += int(residue_index[mask].max()) + 1 + self.gap
        return out

    def forward(self, esm, chain_id, residue_index):
        pos = self.positions(chain_id, residue_index)
        dtype = self.qwen.get_input_embeddings().weight.dtype
        mask = torch.zeros(1, 1, len(esm), len(esm), device=esm.device, dtype=dtype)
        run = lambda z: self.qwen(inputs_embeds=z[None].to(dtype), position_ids=pos[None], attention_mask=mask,
                                  use_cache=False, output_hidden_states=True)
        h = self.inputs(esm, chain_id)
        with torch.no_grad(): first = run(h).last_hidden_state[0].detach()      # recycling carries no gradient
        out = run(h + self.recycle(first))
        z = out.last_hidden_state[0]
        raw = self.head(z.float()); R = rotation(raw[:, 3:]); t = raw[:, :3] * 10
        t = t - t.mean(0, keepdim=True)
        backbone = torch.einsum("lij,aj->lai", R, self.template) + t[:, None]
        return Fold(backbone, R, t, z, out.hidden_states, pos, h + self.recycle(z))


class _Prefix:
    """Read-only key/value prefix: recomputing a checkpointed layer sees exactly what its first forward saw."""

    def __init__(self, kv): self.kv = kv

    def get_seq_length(self, layer_idx=0): return self.kv[0][0].shape[2]

    def update(self, k, v, layer_idx, cache_kwargs=None): return torch.cat([self.kv[layer_idx][0], k], 2), torch.cat([self.kv[layer_idx][1], v], 2)


def _text_layers(qwen, q, pos, mask, kv):
    """The text pass, layer by layer (transformers' own gradient checkpointing would drop the cache)."""
    cache, rope, h = _Prefix(kv), qwen.rotary_emb(q[None], pos[None]), q[None]
    for layer in qwen.layers:
        f = lambda x, layer=layer: layer.forward(x, attention_mask=mask[None, None], position_ids=pos[None], past_key_values=cache, use_cache=False, position_embeddings=rope)
        h = torch.utils.checkpoint.checkpoint(f, h, use_reentrant=False) if torch.is_grad_enabled() and qwen.training else f(h)
    return qwen.norm(h)[0]


def keys_values(qwen, hidden, pos):
    """Per-layer (key, value) of a pass, from that pass's layer inputs."""
    cos, sin = qwen.rotary_emb(hidden[0], pos[None])
    out = []
    for l, layer in enumerate(qwen.layers):
        a = layer.self_attn; x = layer.input_layernorm(hidden[l]); shape = (*x.shape[:-1], -1, a.head_dim)
        k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2); v = a.v_proj(x).view(shape).transpose(1, 2)
        out.append((apply_rotary_pos_emb(k, k, cos, sin)[1], v))
    return out


class RumiFold(nn.Module):
    def __init__(self, folder, lm_head_weight, tokenizer, pointer=True, refiner=False):
        super().__init__()
        self.fold_model, self.tok = folder, tokenizer
        self.register_buffer("lm_head_weight", lm_head_weight, persistent=False)
        if pointer: self.pointer_head = PointerHead(folder.qwen.config.hidden_size)
        self.addresses = AddressParser(tokenizer)
        if refiner: folder.first_position = 0                    # RumiFold-Struct was trained without a text pass
        if refiner: self.refiner, self.confidence_head = Refiner(folder.qwen.config.hidden_size), ConfidenceHead()

    # ---- loading -------------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path, qwen="Qwen/Qwen3-8B", device="cuda", subfolder=None):
        """path: a local checkpoint folder, a released checkpoint name such as "rumifold-156k", or a Hugging Face repo id
        together with `subfolder`."""
        if not Path(path).is_dir():
            if "/" not in str(path): path, subfolder = DEFAULT_MODELS, str(path)
            path = _download(path, subfolder)
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        rev = {} if Path(qwen).is_dir() else {"revision": cfg.get("base_model_revision")}
        base = AutoModel.from_pretrained(qwen, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **rev)
        folder = Folder(base, cfg["lora"]["rank"], cfg["lora"]["alpha"], cfg["chain_gap"])
        state = load_file(str(path / "model.safetensors"))
        state = {("fold." + k[len("base."):] if k.startswith("base.") else k): v for k, v in state.items()}     # RumiFold-Struct naming
        pointer, refiner = (any(k.startswith(p) for k in state) for p in ("pointer_head.", "refiner."))
        model = cls(folder, _lm_head(qwen, rev), AutoTokenizer.from_pretrained(qwen, **rev), pointer, refiner)
        want = {("fold." + k[len("fold_model."):] if k.startswith("fold_model.") else k): k
                for k, p in model.named_parameters() if p.requires_grad}
        missing = sorted(set(want) - set(state))
        if missing: raise ValueError(f"checkpoint lacks {len(missing)} parameters, e.g. {missing[:3]}")
        model.load_state_dict({v: state[k] for k, v in want.items()}, strict=False)
        model.unused = sorted({k.split(".")[0] for k in state if k not in want})     # e.g. auxiliary heads, not needed here
        return model.to(device).eval().requires_grad_(False)

    @classmethod
    def for_training(cls, rank=256, alpha=256, chain_gap=200, qwen="Qwen/Qwen3-8B", init_from=None, device="cuda", revision=None):
        """A trainable model: from scratch, or continuing a released checkpoint folder (its own LoRA rank is used)."""
        if init_from:
            model = cls.from_pretrained(init_from, qwen=qwen, device=device)
            for k, p in model.named_parameters():
                if ".base." not in k and not _frozen(k): p.requires_grad_(True)
        else:
            rev = {"revision": revision} if revision else {}
            base = AutoModel.from_pretrained(qwen, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **rev)
            model = cls(Folder(base, rank, alpha, chain_gap), _lm_head(qwen, rev), AutoTokenizer.from_pretrained(qwen, **rev)).to(device)
        model.fold_model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model.train()

    # ---- folding -------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def fold(self, features, chain_lengths=None):
        """features: [residues, 2560] ESM-C 6B final-layer embeddings; chain_lengths for complexes."""
        device = self.lm_head_weight.device
        lengths = chain_lengths or [len(features)]
        if sum(lengths) != len(features): raise ValueError("chain lengths do not add up to the number of residues")
        chain = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).to(device)
        index = torch.cat([torch.arange(n) for n in lengths]).to(device)
        fold = self.fold_model(features.to(device), chain, index)
        if not hasattr(self, "refiner"): return fold
        R, t, h = self.refiner(fold.states, fold.rotation, fold.translation, chain, index); t = t - t.mean(0, keepdim=True)
        backbone = torch.einsum("lij,aj->lai", R, self.fold_model.template) + t[:, None]
        return fold._replace(backbone=backbone, rotation=R, translation=t, plddt=self.confidence_head(h))

    # ---- language ------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def ask(self, question, fold=None, max_new_tokens=256, addresses=True):
        """Greedy answer. With `fold`, the protein is visible; residues are addressed as A/<1-based position>.

        question: a string, or a list of chat messages. addresses=False switches address lookup off (used by the
        Gene Ontology and Prot2Text evaluations).
        """
        messages = [{"role": "user", "content": question}] if isinstance(question, str) else question
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
        return self.tok.decode(self._generate(ids, fold, max_new_tokens, addresses), skip_special_tokens=True).strip()

    def _prefix(self, fold):
        qwen, dev = self.fold_model.qwen, self.lm_head_weight.device
        zero = torch.zeros(1, dtype=torch.long, device=dev)
        sink = qwen(inputs_embeds=qwen.get_input_embeddings()(torch.tensor([[SINK]], device=dev)), position_ids=zero[None],
                    use_cache=False, output_hidden_states=True)
        pairs = zip(keys_values(qwen, sink.hidden_states, zero), keys_values(qwen, fold.hidden, fold.positions))
        return [(torch.cat([sk, rk], 2), torch.cat([sv, rv], 2)) for (sk, sv), (rk, rv) in pairs]

    # ---- training ------------------------------------------------------------------------------------------------
    def text_logits(self, ids, fold=None, inline=None):
        """Next-token logits at every position of `ids` (a tensor), with gradients.

        With `fold`, the text attends to the second folding pass. inline: {text index: residue or residues} inserts
        those states after the text token, as addresses do at inference; a token is predicted from the state just before it.
        """
        qwen, dev = self.fold_model.qwen, self.lm_head_weight.device; embed = qwen.get_input_embeddings()
        if fold is None:
            n = len(ids); emb = embed(ids); mask = torch.triu(torch.full((n, n), torch.finfo(emb.dtype).min, device=dev, dtype=emb.dtype), diagonal=1)
            h = qwen(inputs_embeds=emb[None], position_ids=torch.arange(n, device=dev)[None], attention_mask=mask[None, None], use_cache=False).last_hidden_state[0]
            return (h @ self.lm_head_weight.T).float()
        zero = torch.zeros(1, dtype=torch.long, device=dev)
        sink = qwen(inputs_embeds=embed(torch.tensor([[SINK]], device=dev)), position_ids=zero[None], use_cache=False, output_hidden_states=True)
        pairs = zip(keys_values(qwen, sink.hidden_states, zero), keys_values(qwen, fold.hidden, fold.positions))
        kv = [(torch.cat([sk, rk], 2), torch.cat([sv, rv], 2)) for (sk, sv), (rk, rv) in pairs]
        emb = embed(ids); rows = fold.rows.to(emb.dtype); inline = inline or {}; parts, is_text = [], []
        for i in range(1, len(ids)):
            parts.append(emb[i:i + 1]); is_text.append(True)
            for r in ([inline[i]] if isinstance(inline.get(i), int) else inline.get(i, ())): parts.append(rows[r:r + 1]); is_text.append(False)
        q = torch.cat(parts); n, P = len(q), kv[0][0].shape[2]; pos = int(fold.positions.max()) + 1 + torch.arange(n, device=dev)
        mask = torch.zeros(n, P + n, device=dev, dtype=q.dtype)
        mask[:, P:] = torch.triu(torch.full((n, n), torch.finfo(q.dtype).min, device=dev, dtype=q.dtype), diagonal=1)
        h = _text_layers(qwen, q, pos, mask, kv)
        text = torch.nonzero(torch.tensor(is_text, device=dev)).flatten()
        predictor = torch.cat([text[1:] - 1, torch.tensor([n - 1], device=dev)])
        first = (sink.last_hidden_state[0] @ self.lm_head_weight.T)[0]
        return torch.cat([first[None], h[predictor] @ self.lm_head_weight.T]).float()

    def _generate(self, ids, fold, max_new, addresses=True):
        qwen, dev = self.fold_model.qwen, self.lm_head_weight.device
        embed = qwen.get_input_embeddings(); emb = embed(torch.tensor(ids, device=dev)); dtype = emb.dtype
        rows = None if fold is None else fold.rows.to(dtype)
        L = 0 if rows is None else len(rows); cache = DynamicCache(); start = 0
        if rows is not None:
            at = dict(self.addresses.find(ids, L)) if addresses else {}; parts = []
            for i in range(1, len(ids)):                          # the sink (token 0) is already in the prefix
                parts.append(emb[i:i + 1])
                if i in at: parts.append(rows[at[i]:at[i] + 1])
            q = torch.cat(parts); kv = self._prefix(fold); P = kv[0][0].shape[2]
            for l, (k, v) in enumerate(kv): cache.update(k, v, l)
            start = int(fold.positions.max()) + 1
        else: q, P = emb, 0
        n = len(q); pos = start + torch.arange(n, device=dev)
        mask = torch.zeros(n, P + n, device=dev, dtype=dtype)
        mask[:, P:] = torch.triu(torch.full((n, n), torch.finfo(dtype).min, device=dev, dtype=dtype), diagonal=1)
        state = {"out": qwen(inputs_embeds=q[None], position_ids=pos[None], attention_mask=mask[None, None],
                             past_key_values=cache, use_cache=True), "next": start + n}

        def feed(e):
            state["out"] = qwen(inputs_embeds=e.view(1, 1, -1), past_key_values=cache, use_cache=True,
                                position_ids=torch.tensor([[state["next"]]], device=dev))
            state["next"] += 1

        last = lambda: state["out"].last_hidden_state[0, -1]
        ad = self.addresses; pointing = addresses and rows is not None and hasattr(self, "pointer_head")
        # addresses the prompt already names may be echoed back; the language model spells those itself
        echo = {str(r + 1)[0] for _, r in ad.find(ids, L)} if pointing else set()
        new, forced, cited, fed, pointed = [], [], set(), set(), None
        for _ in range(max_new):
            finishing = False
            if forced: token = forced.pop(0); finishing = not forced
            else:
                token = int((last() @ self.lm_head_weight.T).argmax())
                if token in STOP: break
            new.append(token)
            if addresses and rows is not None:                    # a spelled address just closed: its residue goes in first
                hit = ad.closed(ids + new, L)
                if hit is not None and hit[0] not in fed: feed(rows[hit[1]])
            feed(embed(torch.tensor([token], device=dev)))
            if finishing and pointed is not None:                 # a pointed address is complete: its residue goes in now
                feed(rows[pointed]); fed.add(len(ids) + len(new) - 1); pointed = None
            if pointing and token == ad.slash and (new[-2] if len(new) > 1 else ids[-1]) in ad.letters:
                if echo:
                    nxt = int((last() @ self.lm_head_weight.T).argmax())
                    if nxt in ad.digits and str(ad.digits[nxt]) in echo: continue
                score = self.pointer_head(last()[None], fold.rows)[0]
                if cited: score[list(cited)] = float("-inf")
                pointed = int(score.argmax()); cited.add(pointed)
                forced = [ad.digit_token[int(d)] for d in str(pointed + 1)]
        return new


def _frozen(name):
    """Qwen's own weights stay frozen; only LoRA, the bridge, recycling, the structure head and task heads train."""
    return ".qwen." in name and not name.endswith((".a.weight", ".b.weight"))


def _download(repo, subfolder):
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, allow_patterns=[f"{subfolder}/*"] if subfolder else None)
    return Path(root) / (subfolder or "")


def _lm_head(qwen, rev):
    from huggingface_hub import hf_hub_download
    get = (lambda f: str(Path(qwen) / f)) if Path(qwen).is_dir() else (lambda f: hf_hub_download(qwen, f, **rev))
    shard = json.loads(Path(get("model.safetensors.index.json")).read_text())["weight_map"]["lm_head.weight"]
    with safe_open(get(shard), framework="pt") as f: return f.get_tensor("lm_head.weight").to(torch.bfloat16)
