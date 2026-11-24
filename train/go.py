"""Gene Ontology: names, ancestors and the most specific terms of a set."""
from functools import lru_cache

SPACE = {"molecular_function": "mf", "biological_process": "bp", "cellular_component": "cc"}


class Ontology:
    def __init__(self, path):
        self.name, self.space, self.parents, cur, obsolete = {}, {}, {}, None, False
        def close():
            if cur and obsolete: self.name.pop(cur, None); self.space.pop(cur, None); self.parents.pop(cur, None)
        for l in open(path):
            l = l.rstrip("\n")
            if l.startswith("["): close(); cur, obsolete = None, False
            elif l.startswith("id: GO:"): cur = l[4:]
            elif cur and l.startswith("name: "): self.name[cur] = l[6:]
            elif cur and l.startswith("namespace: "): self.space[cur] = SPACE.get(l[11:])
            elif cur and l.startswith("is_a: "): self.parents.setdefault(cur, set()).add(l[6:].split(" ! ")[0])
            elif cur and l.startswith("relationship: part_of "): self.parents.setdefault(cur, set()).add(l.split()[2])
            elif cur and l.startswith("is_obsolete: true"): obsolete = True
        close()

    @lru_cache(maxsize=None)
    def ancestors(self, t):
        out = set()
        for p in self.parents.get(t, ()): out.add(p); out |= self.ancestors(p)
        return frozenset(out)

    def leaves(self, terms):
        s = set(terms); covered = set().union(*(self.ancestors(t) for t in s)) if s else set()
        return [t for t in s if t not in covered]
