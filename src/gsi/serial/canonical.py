"""Canonical labeling of a (graph, query) pair: every relabeling of the same instance gets the
same labels, so the rendered prompt is byte-identical.

Individualization-refinement, the scheme behind nauty, in its simplest exact form:

1. Colour every node by (out-degree, in-degree, which query arguments name it) and refine the
   colouring to a fixed point (1-dimensional Weisfeiler-Lehman, edge weights and direction
   included). Colours are *ordered* by comparing structural signatures, never labels.
2. If some colour class still holds several nodes, branch: individualize each of its nodes in
   turn, refine, and recurse. Every branch ends in a total order of the nodes.
3. Each total order gives a relabeled edge list. The canonical labeling is the order whose edge
   list (and query labels) is lexicographically smallest.

Nothing in 1-3 reads a node's label, so the result depends only on the (graph, query) pair up to
isomorphism. Only the *search* is pruned: two nodes of the same class that are twins -- swapping
them is an automorphism (same neighbours with the same weights and directions) -- lead to
isomorphic sub-searches with the same minimum, so one of them is explored. That keeps stars,
complete graphs and isolated nodes linear instead of factorial, and changes no result.

The ordering in step 1 puts low-degree nodes first, as the earlier `degree_wl` heuristic did;
that heuristic broke the remaining ties by the input label, which is why it was not invariant.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class SearchLimit(RuntimeError):
    """The search tree exceeded its leaf budget (a highly symmetric graph that twin pruning does
    not cover). Raised rather than silently falling back to a non-canonical order."""


def canonical_order(nodes, edges, directed: bool, query=(), max_leaves: int = 100_000) -> list:
    """The nodes in canonical order. `edges` are (u, v, *attrs) with attrs compared as a tuple
    (the weight, if any); `query` lists the question's node labels in query-argument order."""
    nodes = list(nodes)
    out: dict = defaultdict(list)
    inn: dict = defaultdict(list)
    for e in edges:
        u, v, attrs = e[0], e[1], tuple(e[2:])
        out[u].append((v, attrs))
        if directed:
            inn[v].append((u, attrs))
        elif u != v:
            out[v].append((u, attrs))
    qtag = {u: tuple(i for i, q in enumerate(query) if q == u) for u in nodes}

    def compress(key: dict) -> dict:
        order = sorted(set(key.values()))
        idx = {k: i for i, k in enumerate(order)}
        return {u: idx[key[u]] for u in nodes}

    def refine(ranks: dict) -> dict:
        n_classes = len(set(ranks.values()))
        while True:
            new = compress({u: (ranks[u],
                                tuple(sorted((ranks[y], a) for y, a in out[u])),
                                tuple(sorted((ranks[y], a) for y, a in inn[u])))
                            for u in nodes})
            k = len(set(new.values()))
            if k == n_classes:
                return new
            ranks, n_classes = new, k

    def twins(u, w) -> bool:
        def view(x, adj):
            return Counter(("self" if y == x else "other" if y in (u, w) else y, a) for y, a in adj[x])
        return view(u, out) == view(w, out) and view(u, inn) == view(w, inn)

    def representatives(cell: list) -> list:
        parent = {u: u for u in cell}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, u in enumerate(cell):
            for w in cell[i + 1:]:
                if find(u) != find(w) and twins(u, w):
                    parent[find(w)] = find(u)
        return [u for u in cell if find(u) == u]

    def certificate(order: list):
        pos = {u: i for i, u in enumerate(order)}
        rel = []
        for e in edges:
            a, b = pos[e[0]], pos[e[1]]
            if not directed and a > b:
                a, b = b, a
            rel.append((a, b, *e[2:]))
        return tuple(sorted(rel)), tuple(pos[q] for q in query)

    best: list = [None, None]
    leaves = [0]

    def search(ranks: dict) -> None:
        ranks = refine(ranks)
        cells: dict = defaultdict(list)
        for u in nodes:
            cells[ranks[u]].append(u)
        target = next((cells[r] for r in sorted(cells) if len(cells[r]) > 1), None)
        if target is None:
            leaves[0] += 1
            if leaves[0] > max_leaves:
                raise SearchLimit(f"more than {max_leaves} leaves")
            order = sorted(nodes, key=ranks.__getitem__)
            cert = certificate(order)
            if best[0] is None or cert < best[0]:
                best[0], best[1] = cert, order
            return
        for v in representatives(sorted(target)):
            search(compress({u: (ranks[u], 0 if u == v else 1) for u in nodes}))

    search(compress({u: (len(out[u]), len(inn[u]), qtag[u]) for u in nodes}))
    return best[1]
