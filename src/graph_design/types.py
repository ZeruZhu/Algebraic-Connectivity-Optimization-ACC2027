from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class DesignProblem:
    n: int
    m: int
    seed: int | None = None

    @property
    def max_edges(self) -> int:
        return comb(self.n, 2)

    @property
    def density(self) -> float:
        min_edges = self.n - 1
        denom = self.max_edges - min_edges
        if denom <= 0:
            return 1.0
        return (self.m - min_edges) / denom


@dataclass
class BackboneCandidate:
    family: str
    graph: nx.Graph
    backbone_lambda2: float
    metadata: dict[str, Any] = field(default_factory=dict)
