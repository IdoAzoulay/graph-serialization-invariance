"""Exact canonical labeling: every relabeling of a (graph, query) pair gets the same labels."""
import random

import networkx as nx
import pytest

from gsi.serial.canonical import canonical_order


def certificate(nodes, edges, directed, query):
    order = canonical_order(nodes, edges, directed, query)
    assert sorted(order) == sorted(nodes)
    pos = {u: i for i, u in enumerate(order)}
    rel = sorted((min(pos[a], pos[b]), max(pos[a], pos[b])) + tuple(r) if not directed
                 else (pos[a], pos[b]) + tuple(r) for a, b, *r in edges)
    return tuple(rel), tuple(pos[q] for q in query)


def relabelings(G, directed=False, weighted=False, query_size=1, trials=15, seed=0):
    """(reference certificate, certificates of random relabelings with shuffled edges and nodes,
    and flipped orientation for undirected edges)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    edges = [(u, v, G[u][v]["w"]) if weighted else (u, v) for u, v in G.edges()]
    query = rng.sample(nodes, query_size)
    ref = certificate(nodes, edges, directed, query)
    out = []
    for _ in range(trials):
        perm = nodes[:]
        rng.shuffle(perm)
        pi = dict(zip(nodes, perm))
        e2 = [(pi[a], pi[b], *r) for a, b, *r in edges]
        rng.shuffle(e2)
        if not directed:
            e2 = [(b, a, *r) if rng.random() < 0.5 else (a, b, *r) for a, b, *r in e2]
        n2 = [pi[u] for u in nodes]
        rng.shuffle(n2)
        out.append(certificate(n2, e2, directed, [pi[q] for q in query]))
    return ref, out


SYMMETRIC = {
    "star": nx.star_graph(19),                   # 19 interchangeable leaves
    "complete": nx.complete_graph(12),
    "path": nx.path_graph(12),                   # the case the degree_wl heuristic got wrong
    "cycle": nx.cycle_graph(15),
    "petersen": nx.petersen_graph(),             # vertex-transitive, no twins: WL splits nothing
    "regular": nx.random_regular_graph(3, 20, seed=1),
    "isolated": nx.empty_graph(8),
    "hypercube": nx.convert_node_labels_to_integers(nx.hypercube_graph(3)),
}


@pytest.mark.parametrize("name", sorted(SYMMETRIC))
@pytest.mark.parametrize("query_size", [0, 1, 2])
def test_symmetric_graphs_are_relabel_invariant(name, query_size):
    ref, outs = relabelings(SYMMETRIC[name], query_size=query_size)
    assert all(c == ref for c in outs)


@pytest.mark.parametrize("seed", range(4))
def test_random_directed_and_weighted_graphs_are_relabel_invariant(seed):
    ref, outs = relabelings(nx.gnp_random_graph(18, 0.2, seed=seed, directed=True), directed=True,
                            query_size=2, seed=seed)
    assert all(c == ref for c in outs)
    W = nx.gnp_random_graph(18, 0.25, seed=seed)
    for u, v in W.edges():
        W[u][v]["w"] = (u * 7 + v) % 3 + 1
    ref, outs = relabelings(W, weighted=True, query_size=2, seed=seed)
    assert all(c == ref for c in outs)


def test_the_query_is_part_of_the_canonical_form():
    """On a star, asking about the centre and asking about a leaf are different instances."""
    nodes, edges = list(range(5)), [(0, i) for i in range(1, 5)]
    assert certificate(nodes, edges, False, [0]) != certificate(nodes, edges, False, [3])
    assert certificate(nodes, edges, False, [3]) == certificate(nodes, edges, False, [1])


def test_non_isomorphic_graphs_with_equal_colourings_are_told_apart():
    """Two triangles and a 6-cycle are both 2-regular: 1-WL alone cannot separate them."""
    nodes = list(range(6))
    two_triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    hexagon = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    assert certificate(nodes, two_triangles, False, []) != certificate(nodes, hexagon, False, [])


def test_weights_and_direction_matter():
    nodes = [0, 1, 2]
    assert certificate(nodes, [(0, 1, 1), (1, 2, 2)], False, [0]) != certificate(nodes, [(0, 1, 2), (1, 2, 1)], False, [0])
    assert certificate(nodes, [(0, 1), (1, 2)], True, [1]) != certificate(nodes, [(1, 0), (1, 2)], True, [1])
