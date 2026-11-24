"""Reading RumiFold's Hugging Face repositories without downloading whole shards."""
import hashlib
import json
import struct

import torch

DEFAULT_MODELS = "RumiFold/RumiFold_Models"


def features_from_hub(sequence, repo="RumiFold/RumiFold_GO", revision=None):
    """ESM-C final-layer features of `sequence` from a RumiFold dataset, read by byte range (a few MB, not the 2 GB shard).

    Works for RumiFold/RumiFold_GO and RumiFold/RumiFold_Functional. Raises KeyError if the exact sequence is not there.
    """
    from huggingface_hub import HfFileSystem, hf_hub_download
    sha = hashlib.sha256(sequence.encode()).hexdigest(); hit = None
    with open(hf_hub_download(repo, "features/index.jsonl", repo_type="dataset", revision=revision)) as f:
        for line in f:
            if sha in line[:200]: hit = json.loads(line); break
    if hit is None or hit["sequence_sha256"] != sha: raise KeyError(f"this exact sequence is not in {repo}; use scripts/make_esmc_features.py")
    a, b = hit["residue_offset"], hit["residue_end"]
    with HfFileSystem().open(f"datasets/{repo}/features/{hit['shard']}", "rb", revision=revision, block_size=2 ** 20) as f:
        n = struct.unpack("<Q", f.read(8))[0]; info = json.loads(f.read(n))["embeddings"]
        if info["dtype"] != "BF16" or info["shape"][1] != 2560: raise ValueError("unexpected feature format")
        f.seek(8 + n + info["data_offsets"][0] + a * 5120); raw = f.read((b - a) * 5120)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(b - a, 2560)
