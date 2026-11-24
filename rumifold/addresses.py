"""Residue addresses: "A/57" in a prompt or reply names residue 57 (1-based)."""


class AddressParser:
    def __init__(self, tok):
        one = lambda s: tok.encode(s, add_special_tokens=False)[0]
        self.letters, self.slash = {one("A"), one(" A")}, one("/")
        self.digits = {one(str(d)): d for d in range(10)}
        self.digit_token = {d: t for t, d in self.digits.items()}

    def find(self, ids, length=None):
        """[(index of the last digit, 0-based residue)] for every address in `ids`."""
        out, i = [], 0
        while i + 2 < len(ids):
            if ids[i] in self.letters and ids[i + 1] == self.slash and ids[i + 2] in self.digits:
                j, n = i + 2, 0
                while j < len(ids) and ids[j] in self.digits: n = n * 10 + self.digits[ids[j]]; j += 1
                if n >= 1 and (length is None or n <= length): out.append((j - 1, n - 1))
                i = j
            else: i += 1
        return out

    def closed(self, ids, length):
        """The address that the last token of `ids` just closed, or None."""
        if len(ids) < 4 or ids[-1] in self.digits or ids[-2] not in self.digits: return None
        found = self.find(ids[:-1], length)
        return found[-1] if found and found[-1][0] == len(ids) - 2 else None
