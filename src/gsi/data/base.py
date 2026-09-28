from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx

ANSWER_TYPES = ("bool", "int", "float", "node", "node_set", "path", "edge_set")


@dataclass
class GraphInstance:
    id: str
    dataset: str
    task: str
    directed: bool
    weighted: bool
    nodes: list[int]
    edges: list[list]
    query_args: dict[str, Any]
    answer_type: str
    ground_truth: Any
    native_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GraphInstance":
        return cls(**d)

    def to_nx(self) -> nx.Graph:
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(self.nodes)
        if self.weighted:
            G.add_weighted_edges_from((u, v, w) for u, v, w in self.edges)
        else:
            G.add_edges_from((u, v) for u, v, *_ in self.edges)
        return G


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Rows of a JSONL file. A final line cut off mid-write (a killed run) is skipped with a warning, so the
    run can resume; a malformed line anywhere else is an error."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        pending = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if pending is not None:
                yield json.loads(pending)
            pending = line
        if pending is not None:
            try:
                row = json.loads(pending)
            except json.JSONDecodeError:
                print(f"warning: {path}: skipping a truncated last line", file=sys.stderr)
                return
            yield row


def trim_partial_line(path: str | Path) -> bool:
    """Cut a final line that was not terminated (a killed writer) so that appends start on a new line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    path.write_bytes(data[:data.rfind(b"\n") + 1])
    print(f"warning: {path}: removed a truncated last line", file=sys.stderr)
    return True


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
