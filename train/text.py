"""Answering in text with the protein in view: cross-entropy on the answer tokens and the end-of-turn token."""
import torch
from torch.nn.functional import cross_entropy

END = "<|im_end|>"


def answer_loss(model, fold, question, answer):
    """question: a string or chat messages. Addresses are not resolved (as in the Gene Ontology and Prot2Text tasks)."""
    tok = model.tok; messages = [{"role": "user", "content": question}] if isinstance(question, str) else question
    prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    prompt = list(prompt["input_ids"]) if hasattr(prompt, "keys") else list(prompt)
    target = tok.encode(answer, add_special_tokens=False) + [tok.convert_tokens_to_ids(END)]
    ids = torch.tensor(prompt + target, device=model.lm_head_weight.device)
    logits = model.text_logits(ids, fold)
    return cross_entropy(logits[len(prompt) - 1:len(ids) - 1], ids[len(prompt):])
