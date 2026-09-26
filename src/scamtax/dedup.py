"""Near-duplicate clustering with MinHash LSH.

Smishing campaigns send thousands of near-identical messages that differ only
in a name, link or amount. If copies of the same template land in both train
and test, offline metrics measure memorization instead of generalization.
We assign every message a *template id* and split by template, not by row.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from datasketch import MinHash, MinHashLSH


def _shingles(text: str, k: int) -> set[str]:
    text = text if len(text) >= k else text.ljust(k)
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str, k: int, num_perm: int, seed: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    for s in _shingles(text, k):
        m.update(s.encode("utf-8"))
    return m


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def template_ids(
    texts: Iterable[str],
    threshold: float = 0.8,
    shingle_size: int = 5,
    num_perm: int = 128,
    seed: int = 13,
) -> np.ndarray:
    """Cluster texts whose estimated Jaccard similarity >= threshold.

    Returns an int array of cluster ids (connected components of the LSH
    candidate graph), one per input text.
    """
    texts = list(texts)
    n = len(texts)
    uf = _UnionFind(n)

    # Exact duplicates are unioned directly; only unique strings go to LSH.
    first_seen: dict[str, int] = {}
    unique_idx: list[int] = []
    for i, t in enumerate(texts):
        if t in first_seen:
            uf.union(first_seen[t], i)
        else:
            first_seen[t] = i
            unique_idx.append(i)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i in unique_idx:
        m = _minhash(texts[i], shingle_size, num_perm, seed)
        for j in lsh.query(m):
            uf.union(i, int(j))
        lsh.insert(str(i), m)

    roots = np.array([uf.find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids
