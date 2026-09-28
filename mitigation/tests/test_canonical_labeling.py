"""E3 canonical labeling as rendered prompts (scripts/canonical_labeling.py, loaded by path)."""
import importlib.util
from pathlib import Path

from gsi.data.base import GraphInstance
from gsi.prompts.modes import build_prompt
from gsi.serial.variant import canonical, make_variants

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


canonical_labeling = load("canonical_labeling")

STAR = GraphInstance(id="t-star-node_degree", dataset="graphqa", task="node_degree", directed=False, weighted=False,
                     nodes=list(range(8)), edges=[[0, i] for i in range(1, 8)] + [[3, 4]],
                     query_args={"node": 5}, answer_type="int", ground_truth=1)


def test_canonical_scheme_renders_every_relabeling_identically():
    canon = canonical(STAR, structure="edge_list", syntax="graphqa_nl")
    vs = make_variants(STAR, canon, {"relabel": {"seeds": 6}, "order": {"kinds": ["shuffle_all"], "seeds": 3}})
    texts = {build_prompt("code", STAR, canonical_labeling.canonical_variant(v, "canonical", STAR.query_args),
                          "native").user for v in vs}
    assert len(texts) == 1


def test_canonical_scheme_writes_undirected_edges_low_to_high_and_maps_answers_back():
    v = canonical_labeling.canonical_variant(canonical(STAR, structure="edge_list", syntax="plain"),
                                             "canonical", STAR.query_args)
    assert all(a <= b for a, b, *_ in v.edge_seq)
    back = v.inverse_label_map
    assert sorted(sorted((back[a], back[b])) for a, b in v.edge_seq) == sorted(sorted(e) for e in STAR.edges)


def test_merge_rows_replaces_what_was_computed_in_place_and_appends_the_rest():
    import pandas as pd
    row = lambda m, s, a, acc: {"model": m, "scheme": s, "arm": a, "acc": acc}
    old = pd.DataFrame([row("A", "canonical", "direct", 0.1), row("A", "degree_wl", "direct", 0.2),
                        row("B", "canonical", "direct", 0.3), row("B", "canonical", "code/native", 0.4)])
    out = pd.DataFrame([row("C", "canonical", "direct", 0.9), row("B", "canonical", "direct", 0.5),
                        row("B", "canonical", "code/native", 0.6)])
    got = canonical_labeling.merge_rows(old, out)
    assert got.to_dict("records") == [row("A", "canonical", "direct", 0.1), row("A", "degree_wl", "direct", 0.2),
                                      row("B", "canonical", "direct", 0.5), row("B", "canonical", "code/native", 0.6),
                                      row("C", "canonical", "direct", 0.9)]
