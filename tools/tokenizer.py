#!/usr/bin/env python3
"""Qwen3 byte-level BPE tokenizer in pure Python (no third-party deps).

Reimplements the HuggingFace fast tokenizer pipeline from tokenizer.json:
  pretokenizer split (GPT-2 style regex, hand-rolled scanner) -> byte-level
  encoding -> BPE merges. Special tokens (<|im_start|> ...) are matched
  literally before BPE. Decoding reverses the byte-level map.

Usage:
  tokenizer.py "prompt text"                 # print token ids
  tokenizer.py --decode 151644,9707,11       # print text
"""

import json
import sys
import unicodedata

TOKENIZER_JSON = "external/qwen3/tokenizer.json"


def bytes_to_unicode():
    # GPT-2 byte<->unicode map: printable bytes map to themselves, the rest
    # to code points starting at 256.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    def __init__(self, path=TOKENIZER_JSON):
        tk = json.load(open(path))
        self.vocab = tk["model"]["vocab"]  # token string -> id
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.bpe_ranks = {}
        for rank, merge in enumerate(tk["model"]["merges"]):
            a, b = merge.split(" ") if isinstance(merge, str) else merge
            self.bpe_ranks[(a, b)] = rank
        self.byte_encode = bytes_to_unicode()
        self.byte_decode = {v: k for k, v in self.byte_encode.items()}
        self.special = {}  # literal string -> id
        for t in tk.get("added_tokens", []):
            self.special[t["content"]] = t["id"]
            self.id_to_token[t["id"]] = t["content"]
        self.eos_id = self.special.get("<|im_end|>", 151645)

    # ---- pretokenizer -----------------------------------------------
    # (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
    #  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    # Python re lacks \p{L}; scan by hand with unicodedata classes.

    @staticmethod
    def _is_l(c):
        return c.isalpha()

    @staticmethod
    def _is_n(c):
        return unicodedata.category(c).startswith("N")

    def _split(self, text):
        is_l, is_n = self._is_l, self._is_n

        def is_s(c):
            return c.isspace()

        def is_sym(c):  # not space, not L, not N
            return not is_s(c) and not is_l(c) and not is_n(c)

        def is_prefix(c):  # [^\r\n\p{L}\p{N}]
            return c not in "\r\n" and not is_l(c) and not is_n(c)

        chunks = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            for con in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d"):
                if text[i : i + len(con)].lower() == con:
                    chunks.append(text[i : i + len(con)])
                    i += len(con)
                    break
            else:
                if is_l(c) or (is_prefix(c) and i + 1 < n and is_l(text[i + 1])):
                    j = i if is_l(c) else i + 1
                    while j < n and is_l(text[j]):
                        j += 1
                    chunks.append(text[i:j])
                    i = j
                elif is_n(c):
                    chunks.append(c)
                    i += 1
                elif is_sym(c) or (c == " " and i + 1 < n and is_sym(text[i + 1])):
                    j = i if is_sym(c) else i + 1
                    while j < n and is_sym(text[j]):
                        j += 1
                    while j < n and text[j] in "\r\n":
                        j += 1
                    chunks.append(text[i:j])
                    i = j
                else:
                    # whitespace run: \s*[\r\n]+ | \s+(?!\S) | \s+
                    j = i
                    while j < n and is_s(text[j]):
                        j += 1
                    run = text[i:j]
                    last_nl = max(run.rfind("\n"), run.rfind("\r"))
                    if last_nl >= 0:
                        chunks.append(run[: last_nl + 1])
                        i += last_nl + 1
                    elif j == n or len(run) == 1:
                        chunks.append(run)
                        i = j
                    else:
                        # \s+(?!\S) leaves the last space to prefix the word
                        chunks.append(run[:-1])
                        i += len(run) - 1
        return chunks

    # ---- BPE ---------------------------------------------------------

    def _bpe(self, word):
        if len(word) == 1:
            return word
        while True:
            best, best_rank = None, None
            for k in range(len(word) - 1):
                rank = self.bpe_ranks.get((word[k], word[k + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = k, rank
            if best is None:
                return word
            word = word[:best] + [word[best] + word[best + 1]] + word[best + 2 :]

    def _encode_chunk(self, chunk):
        word = [self.byte_encode[b] for b in chunk.encode("utf-8")]
        return [self.vocab[t] for t in self._bpe(word)]

    def encode(self, text):
        ids = []
        # match special tokens literally, leftmost-longest
        i = 0
        while i < len(text):
            hit = None
            for s, tid in self.special.items():
                if text.startswith(s, i) and (hit is None or len(s) > len(hit[0])):
                    hit = (s, tid)
            if hit:
                ids.append(hit[1])
                i += len(hit[0])
                continue
            # plain run up to the next special token
            j = i
            while j < len(text) and not any(
                text.startswith(s, j) for s in self.special
            ):
                j += 1
            for chunk in self._split(text[i:j]):
                ids.extend(self._encode_chunk(chunk))
            i = j
        return ids

    def decode(self, ids):
        text = "".join(self.id_to_token[i] for i in ids)
        data = bytes(self.byte_decode[c] for c in text)
        return data.decode("utf-8", errors="replace")


def main():
    args = sys.argv[1:]
    tk = Tokenizer()
    if args and args[0] == "--decode":
        ids = [int(x) for x in args[1].split(",") if x]
        print(tk.decode(ids))
        return
    if not args:
        print(__doc__)
        return
    ids = tk.encode(args[0])
    print(" ".join(str(i) for i in ids))
    print(f"# {len(ids)} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
