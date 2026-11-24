import torch
from safetensors import safe_open

THREE = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))


def load_features(path, offset=0, end=None):
    """ESM-C features [residues, 2560]; offset/end select one protein from a packed shard."""
    with safe_open(path, framework="pt") as f:
        s = f.get_slice("embeddings")
        return s[offset:end if end is not None else s.get_shape()[0]]


def write_pdb(path, backbone, sequence=None, chain_lengths=None):
    """backbone: [residues, 3 (N, CA, C), 3]."""
    xyz = backbone.float().cpu(); lengths = chain_lengths or [len(xyz)]; seq = sequence or "G" * len(xyz)
    if len(seq) != len(xyz): raise ValueError("sequence length does not match the backbone")
    serial, r = 1, 0
    with open(path, "w") as f:
        for c, n in enumerate(lengths):
            name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[c]
            for i in range(n):
                for atom, (x, y, z) in zip(("N", "CA", "C"), xyz[r].tolist()):
                    f.write(f"ATOM  {serial:5d}  {atom:<3s} {THREE.get(seq[r], 'UNK')} {name}{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom[0]}\n")
                    serial += 1
                r += 1
            f.write("TER\n")
        f.write("END\n")
