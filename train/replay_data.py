"""Reading the published replay archives (RumiFold_Functional/replay): one tar per shard, npz chunks inside."""
import hashlib
import io
import json
import tarfile
from collections import OrderedDict

import numpy as np


class Replay:
    def __init__(self, root, max_tokens=2048, dev_fraction=.002, cache=8):
        self.root, self.max_tokens, self.cache, self.size = root, max_tokens, OrderedDict(), cache
        self.train, self.dev, self.members = [], [], {}
        for line in open(f"{root}/index.jsonl"):
            r = json.loads(line); h = int(hashlib.sha256(("replay-dev|" + r["id"]).encode()).hexdigest(), 16) % 10000
            (self.dev if h < dev_fraction * 10000 else self.train).append((r["shard"], r["chunk"], r["id"]))

    def _chunk(self, shard, chunk):
        if chunk in self.cache: self.cache.move_to_end(chunk); return self.cache[chunk]
        path = f"{self.root}/{shard}"
        if shard not in self.members:
            with tarfile.open(path) as t: self.members[shard] = {m.name: m for m in t}
        with tarfile.open(path) as t: z = np.load(io.BytesIO(t.extractfile(self.members[shard][chunk]).read()))
        d = {k: z[k] for k in z.files if k != "metadata"}; d["meta"] = {c["id"]: c for c in json.loads(str(z["metadata"]))["conversations"]}
        self.cache[chunk] = d
        while len(self.cache) > self.size: self.cache.popitem(last=False)
        return d

    def turns(self, key):
        d = self._chunk(key[0], key[1]); out = []
        for t in d["meta"][key[2]]["turns"]:
            P, N, a, b = t["input_tokens"], t["output_tokens"], t["input_offset"], t["response_offset"]
            if N == 0 or P + N > self.max_tokens: continue
            out.append({k: d[k][b:b + N] for k in ("response_ids", "topk_ids", "topk_logprobs", "tail_mass")} | {"input_ids": d["input_ids"][a:a + P]})
        return out
