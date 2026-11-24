"""Questions about the predicted structure, worded from a phrasing bank and labelled on the model's own (detached) prediction."""
import json
import re

import numpy as np
import torch
from torch.nn.functional import log_softmax

MARK, END = "<|vision_pad|>", "<|im_end|>"        # MARK shows where a residue's state is inserted; it is not a token of the question
RG_TERCILES, BURIAL_TERCILES, MAX_NUMBER = (7.32, 8.54), (12, 17), 999
CA_BINS, SEGMENT_BINS = ((8, "under8"), (16, "8to16"), (np.inf, "over16")), ((12, "under12"), (24, "12to24"), (np.inf, "over24"))
SIGMA = {"d": 1., "dmin": 1., "dnear": 1., "dfar": 1., "rg": .5, "n": 1.}
SLOT = re.compile(r"\{(\w+)\}")
REVIEW = re.compile(r"aligned|from there|the center is|the core|located about \{dmin\}", re.I)
HEADS = {"closer_segment": ["{near} is closer.", "The {which} one, {near}.", "{near}.", "{near} is nearer."],
         "interface_contact": {"yes": ["Yes.", "Yes, it is in contact.", "Yes, they touch."], "no": ["No.", "No, it is not in contact.", "No, they do not touch."]}}
LOCATORS = ["The protein tokens are the residues in chain then residue order, numbered from 0: {loc}.",
            "Residues map to protein tokens in chain then residue order, counting from 0 ({loc}).",
            "For reference, protein tokens follow chain then residue order from 0: {loc}."]


def addresses(row):
    """One "chain/residue" name per protein row, numbering unresolved residues from their nearest resolved neighbour."""
    out = []
    for c in row["chains"]:
        ids = c.get("auth_residue_ids") or [str(k + 1) for k in range(len(c["sequence"]))]
        out += [f"{c['instance_id']}/{n}{i}" for n, i in _fill(ids)]
    return out


def _fill(ids):
    parsed = [None if x is None else (int((m := re.fullmatch(r"(-?\d+)([A-Z]?)", x)).group(1)), m.group(2)) for x in ids]
    known = [k for k, p in enumerate(parsed) if p is not None]; out = []
    for s, p in enumerate(parsed):
        if p is None: t = min(known, key=lambda k: (abs(k - s), k > s)); p = (parsed[t][0] + (s - t), "")
        out.append(p)
    if len(set(out)) == len(out): return out
    out, first = [], known[0]
    for s, p in enumerate(parsed):
        if p is not None: out.append(p)
        elif s < first: out.append((parsed[first][0] - (first - s), ""))
        else:
            prev = max(k for k in known if k < s); code = s - prev - 1
            if code >= 26 or parsed[prev][1]: raise ValueError("cannot number unresolved residues")
            out.append((parsed[prev][0], chr(65 + code)))
    if len(set(out)) != len(out): raise ValueError("addresses collide")
    return out


def label(family, ca, p):
    """(word label, value) of a question family on CA coordinates [N, 3]."""
    c = lambda rows: ca[list(rows)].mean(0)
    if family == "ca_distance_bin": v = float(np.linalg.norm(ca[p["i"]] - ca[p["j"]])); return next(n for hi, n in CA_BINS if v < hi), v
    if family == "segment_distance_bin": v = float(np.linalg.norm(c(p["a"]) - c(p["b"]))); return next(n for hi, n in SEGMENT_BINS if v < hi), v
    if family == "closer_segment":
        db, dc = (float(np.linalg.norm(c(p[k]) - c(p["anchor"]))) for k in ("b", "c")); return ("first" if db < dc else "second"), dc - db
    if family == "local_compaction":
        x = ca[list(p["rows"])]; v = float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean())); lo, hi = RG_TERCILES
        return ("compact" if v < lo else "intermediate" if v < hi else "extended"), v
    if family == "interface_contact":
        v = float(np.linalg.norm(ca[list(p["other_chain_rows"])] - ca[p["i"]], axis=1).min()); return ("yes" if v < 8. else "no"), v
    if family == "burial_bin":
        keep = np.ones(len(ca), bool); keep[list(p.get("exclude_rows", []))] = False
        v = int((np.linalg.norm(ca[keep] - ca[p["i"]], axis=1) < 10.).sum() - 1); lo, hi = BURIAL_TERCILES
        return ("exposed" if v < lo else "intermediate" if v < hi else "buried"), float(v)
    if family == "absent_address": return "not_in_structure", None
    raise ValueError(family)


class Phrasing:
    def __init__(self, path):
        t = json.load(open(path))["templates"]
        use = lambda v: [x["text"] for x in v if x.get("judge", "keep") in ("keep", "seed") and not REVIEW.search(x["text"])]
        self.q, self.a = ({f: use(v) for f, v in t[k].items()} for k in ("question", "answer"))

    def render(self, item, names, ca, rng):
        """-> question (with MARKs), answer, scored spans of the answer, and the residue rows behind each MARK."""
        f, p = item["family"], item["params"]; ca = np.asarray(ca, np.float64)
        t = lambda k: names[k]; seg = lambda rows: f"{t(min(rows))}-{t(max(rows))}"; s, num, head, scored = {}, {}, "", None
        if "i" in p: s["a"] = t(p["i"])
        if "j" in p: s["b"] = t(p["j"])
        if f == "segment_distance_bin": s["sa"], s["sb"] = seg(p["a"]), seg(p["b"])
        if f == "local_compaction": s["seg"] = seg(p["rows"])
        if f == "absent_address": s["ghost"] = p["ghost"]
        lab, val = label(f, ca, p)
        if f in ("ca_distance_bin", "segment_distance_bin"): num["d"] = val
        elif f == "local_compaction": num["rg"] = val; s["comp_word"] = lab; scored = lab
        elif f == "burial_bin": num["n"] = val; s["burial_word"] = lab; scored = lab
        elif f == "closer_segment":
            c = lambda rows: ca[list(rows)].mean(0); db, dc = (float(np.linalg.norm(c(p[k]) - c(p["anchor"]))) for k in ("b", "c"))
            s.update(anchor=seg(p["anchor"]), s1=seg(p["b"]), s2=seg(p["c"]), which=lab); near, far = (("b", "c"), ("c", "b"))[lab == "second"]
            s["near"], s["far"] = seg(p[near]), seg(p[far]); num["dnear"], num["dfar"] = (db, dc) if lab == "first" else (dc, db)
            head = rng.choice(HEADS[f]); scored = s["near"]
        elif f == "interface_contact":
            s["chain"] = names[p["other_chain_rows"][0]].split("/")[0]; num["dmin"] = val
            head = rng.choice(HEADS[f][lab]); scored = head.split(",")[0].rstrip(".")
        locs = [f"{t(p[k])} is protein token {p[k]}" for k in ("i", "j") if k in p]
        locs += [f"{seg(p[k])} is protein tokens {min(p[k])}-{max(p[k])}" for k in ("anchor", "a", "b", "c", "rows") if k in p and k != "i"]
        if f == "absent_address": locs.append(f'{p["ghost"]} has no protein token')
        loc = rng.choice(LOCATORS).format(loc="; ".join(locs))
        rows = {"a": [p["i"]] if "i" in p else None, "b": [p["j"]] if "j" in p else None}
        if f == "segment_distance_bin": rows.update(sa=list(p["a"]), sb=list(p["b"]))
        if f == "local_compaction": rows["seg"] = list(p["rows"])
        if f == "closer_segment": rows.update(anchor=list(p["anchor"]), s1=list(p["b"]), s2=list(p["c"]))
        inline = []; question = _fill_slots(rng.choice(self.q[f]), s, num, {k: v for k, v in rows.items() if v}, inline)[0]
        answer, spans = _fill_slots((head + " " if head else "") + rng.choice(self.a[f]), s, num)
        if scored is not None: k = answer.find(scored); spans.append(("word", k, k + len(scored), None, None))
        if f == "absent_address": spans.append(("word", 0, len(answer), None, None))
        return loc + " " + question, answer, spans, inline


def _fill_slots(template, s, num, marks=None, inline=None):
    out, spans, pos = "", [], 0
    for m in SLOT.finditer(template):
        out += template[pos:m.start()]; k = m.group(1); pos = m.end()
        if k in SIGMA:
            text = str(int(round(num[k]))); spans.append(("number", len(out), len(out) + len(text), float(num[k]), SIGMA[k])); out += text
        else:
            out += s[k]
            if marks and k in marks: out += MARK; inline.append(marks[k])
    out += template[pos:]
    return (out[0].upper() + out[1:] if out and out[0].islower() and not out.startswith("{") else out), spans


def prepare(tok, question, answer, spans, inline_rows):
    """Token ids, {text index: rows inserted after it}, prompt length and a per-answer-token kind/soft-target list."""
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    prompt = list(prompt["input_ids"]) if hasattr(prompt, "keys") else list(prompt); mark = tok.convert_tokens_to_ids(MARK); ids, at = [], []
    for t in prompt:
        if t == mark: at.append(len(ids) - 1)
        else: ids.append(t)
    enc = tok(answer, return_offsets_mapping=True, add_special_tokens=False); ans, off = enc["input_ids"], enc["offset_mapping"]
    digit = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]; kind, soft = ["other"] * (len(ans) + 1), {}
    for k, a, b, value, sigma in spans:
        toks = [i for i, (x, y) in enumerate(off) if x >= a and y <= b and y > x]
        if k == "word":
            for i in toks: kind[i] = "word"
        else:
            d = [i for i in toks if ans[i] in digit]
            for n, i in enumerate(d + [d[-1] + 1]): kind[i] = "number"; soft[i] = (value, sigma, n)
    end = tok.convert_tokens_to_ids(END)
    return ids + ans + [end], dict(zip(at, inline_rows)), len(ids), kind, soft, digit


def qa_loss(logits, ids, n_prompt, kind, soft, digit, other_weight=.3):
    """Cross-entropy on the answer; digits of a number get a Gaussian over integers, given the digits written so far."""
    lp = log_softmax(logits[n_prompt - 1:len(ids) - 1].float(), -1); ans = ids[n_prompt:]; total, weight = lp.new_zeros(()), 0.
    for i, k in enumerate(kind[:len(ans)]):
        if k != "number": c = 1. if k == "word" else other_weight; total = total - c * lp[i, ans[i]]; weight += c; continue
        value, sigma, pos = soft[i]; modal = ""
        for t in ans[i - pos:]:
            if t not in digit: break
            modal += str(digit.index(t))
        v = np.arange(MAX_NUMBER + 1); g = np.exp(-.5 * ((v - min(max(value, 0.), MAX_NUMBER)) / sigma) ** 2); tgt = {}
        for n, w in zip(v, g):
            s = str(n)
            if w < 1e-6 or not s.startswith(modal[:pos]): continue
            tok_id = digit[int(s[pos])] if len(s) > pos else ans[i]; tgt[tok_id] = tgt.get(tok_id, 0.) + w
        z = sum(tgt.values()); idx = torch.tensor(list(tgt), device=lp.device); w = torch.tensor([x / z for x in tgt.values()], device=lp.device, dtype=lp.dtype)
        total = total - (w * lp[i, idx]).sum(); weight += 1.
    return total / weight
