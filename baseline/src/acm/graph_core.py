from __future__ import annotations

import heapq
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


def max_edges(n: int) -> int:
    return n * (n - 1) // 2


def validate_nm(n: int, m: int) -> None:
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if m < n - 1:
        raise ValueError(f"m must be >= n-1 for connected graphs, got n={n}, m={m}")
    if m > max_edges(n):
        raise ValueError(f"m must be <= n(n-1)/2, got n={n}, m={m}")


def target_density(n: int, m: int) -> float:
    validate_nm(n, m)
    denom = max_edges(n) - (n - 1)
    if denom == 0:
        return 0.0
    return (m - (n - 1)) / denom


def adjacency_from_edges(n: int, edges: Iterable[Edge]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        add_edge(adj, u, v)
    return adj


def edge_count(adj: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(adj, k=1)))


def edges_from_adjacency(adj: np.ndarray) -> List[Edge]:
    iu, iv = np.where(np.triu(adj, k=1))
    return [(int(u), int(v)) for u, v in zip(iu, iv)]


def non_edges(adj: np.ndarray) -> List[Edge]:
    n = adj.shape[0]
    iu, iv = np.where(np.triu(~adj, k=1))
    return [(int(u), int(v)) for u, v in zip(iu, iv)]


def degrees(adj: np.ndarray) -> np.ndarray:
    return adj.sum(axis=1).astype(np.int64)


def add_edge(adj: np.ndarray, u: int, v: int) -> None:
    n = adj.shape[0]
    if u == v:
        raise ValueError("self-loop not allowed")
    if not (0 <= u < n and 0 <= v < n):
        raise ValueError(f"edge out of range: {(u, v)} for n={n}")
    if adj[u, v]:
        raise ValueError(f"edge {(u, v)} already exists")
    adj[u, v] = True
    adj[v, u] = True


def remove_edge(adj: np.ndarray, u: int, v: int) -> None:
    if not adj[u, v]:
        raise ValueError(f"edge {(u, v)} does not exist")
    adj[u, v] = False
    adj[v, u] = False


def path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        adj[i, i + 1] = True
        adj[i + 1, i] = True
    return adj


def cycle_graph(n: int) -> np.ndarray:
    adj = path_graph(n)
    adj[0, n - 1] = True
    adj[n - 1, 0] = True
    return adj


def star_graph(n: int, center: int = 0) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for v in range(n):
        if v == center:
            continue
        adj[center, v] = True
        adj[v, center] = True
    return adj


def deterministic_tree_graph(n: int, seed: int | None = None) -> np.ndarray:
    """
    Build a deterministic pseudo-random tree using a Prüfer-sequence decode.
    For fixed (n, seed), output is identical across runs.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n == 2:
        return path_graph(2)

    if seed is None:
        seed = 1_000_003 + 9_973 * n
    rng = np.random.default_rng(int(seed))

    prufer = rng.integers(0, n, size=n - 2, dtype=np.int64)
    degree = np.ones(n, dtype=np.int64)
    for x in prufer:
        degree[int(x)] += 1

    leaves = [int(i) for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    adj = np.zeros((n, n), dtype=bool)

    for x_raw in prufer:
        x = int(x_raw)
        leaf = int(heapq.heappop(leaves))
        add_edge(adj, leaf, x)
        degree[leaf] -= 1
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)

    u = int(heapq.heappop(leaves))
    v = int(heapq.heappop(leaves))
    add_edge(adj, u, v)
    return adj


def complement_graph(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    comp = ~adj.copy()
    np.fill_diagonal(comp, False)
    return comp


def graph_edge_index(adj: np.ndarray) -> np.ndarray:
    """Returns undirected edges as shape [E,2] int64."""
    edges = edges_from_adjacency(adj)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def budget_decomposition(n: int, m: int) -> Tuple[int, int]:
    validate_nm(n, m)
    k = int((2 * m) // n)
    r = int(m - (n * k // 2))
    return k, r


def is_connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        cur = stack.pop()
        neigh = np.where(adj[cur])[0]
        for nxt in neigh:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(int(nxt))
    return bool(seen.all())


def safe_add_edges(adj: np.ndarray, edges: Sequence[Edge]) -> np.ndarray:
    out = adj.copy()
    for u, v in edges:
        if u == v or out[u, v]:
            continue
        out[u, v] = True
        out[v, u] = True
    return out


def divisors(n: int) -> List[int]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    out: List[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            j = n // i
            if j != i:
                out.append(j)
        i += 1
    out.sort()
    return out


def complete_r_partite_graph(part_sizes: Sequence[int]) -> np.ndarray:
    if not part_sizes:
        raise ValueError("part_sizes must be non-empty")
    if any(s <= 0 for s in part_sizes):
        raise ValueError(f"part sizes must be positive, got {part_sizes}")

    n = int(sum(int(s) for s in part_sizes))
    # Build a part-id label per vertex, then connect vertices from different parts.
    part_id = np.empty(n, dtype=np.int32)
    start = 0
    for pid, s in enumerate(part_sizes):
        end = start + int(s)
        part_id[start:end] = int(pid)
        start = end

    adj = part_id[:, None] != part_id[None, :]
    np.fill_diagonal(adj, False)
    return adj


def validate_partite_spec(n: int, r: int, part_sizes: Sequence[int]) -> List[int]:
    """
    Validate complete r-partite specification and return normalized integer sizes.
    Requires:
      - n >= 2
      - 1 <= r <= n
      - len(part_sizes) == r
      - all part sizes > 0
      - sum(part_sizes) == n
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if r < 1 or r > n:
        raise ValueError(f"r must be in [1,n], got n={n}, r={r}")
    if not part_sizes:
        raise ValueError("part_sizes must be non-empty")

    sizes = [int(s) for s in part_sizes]
    if len(sizes) != r:
        raise ValueError(
            f"number of parts must be r, got r={r}, len(part_sizes)={len(sizes)}"
        )
    if any(s <= 0 for s in sizes):
        raise ValueError(f"part sizes must be positive, got {part_sizes}")

    ssum = int(sum(sizes))
    if ssum != n:
        raise ValueError(
            f"sum(part_sizes) must equal n, got n={n}, sum={ssum}, part_sizes={part_sizes}"
        )
    return sizes


def complete_r_partite_graph_from_n(
    n: int, r: int, part_sizes: Sequence[int]
) -> np.ndarray:
    """
    Construct complete r-partite graph with explicit (n, r, part_sizes).
    """
    sizes = validate_partite_spec(n=n, r=r, part_sizes=part_sizes)
    return complete_r_partite_graph(sizes)


def balanced_part_sizes(n: int, r: int, require_divisible: bool = True) -> List[int]:
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if r < 1 or r > n:
        raise ValueError(f"r must be in [1,n], got n={n}, r={r}")
    q, rem = divmod(n, r)
    if require_divisible and rem != 0:
        raise ValueError(f"n must be divisible by r, got n={n}, r={r}")
    # Balanced sizes differ by at most 1 when n is not divisible.
    sizes = [q + 1] * rem + [q] * (r - rem)
    return sizes


def balanced_r_partite_graph(
    n: int, r: int, require_divisible: bool = True
) -> Tuple[np.ndarray, List[int]]:
    sizes = balanced_part_sizes(n, r, require_divisible=require_divisible)
    return complete_r_partite_graph(sizes), sizes


def multipartite_edge_count_from_sizes(part_sizes: Sequence[int]) -> int:
    n = int(sum(part_sizes))
    s2 = int(sum(int(s) * int(s) for s in part_sizes))
    return (n * n - s2) // 2


def multipartite_density_from_sizes(part_sizes: Sequence[int]) -> float:
    n = int(sum(part_sizes))
    m = multipartite_edge_count_from_sizes(part_sizes)
    m_max = max_edges(n)
    if m_max == 0:
        return 0.0
    return float(m) / float(m_max)


def multipartite_lambda2_closed_form(part_sizes: Sequence[int]) -> float:
    n = int(sum(part_sizes))
    if n <= 1:
        return 0.0
    if len(part_sizes) <= 1:
        return 0.0

    # Laplacian spectrum for complete multipartite K_{n1,...,nr}:
    # - 0 (mult 1)
    # - n - n_i (mult n_i - 1) for each part i
    # - n (mult r - 1)
    # Hence lambda2 is the smallest positive eigenvalue among present groups.
    cands: List[float] = [float(n)]  # present when r >= 2
    for s in part_sizes:
        si = int(s)
        if si > 1:
            cands.append(float(n - si))
    return float(min(cands))


def normalize_bipartite_circulant_shifts(half_n: int, shifts: Sequence[int]) -> List[int]:
    if half_n <= 0:
        raise ValueError(f"half_n must be positive, got {half_n}")
    if not shifts:
        raise ValueError("shifts must be non-empty")
    out = sorted({int(s) % int(half_n) for s in shifts})
    if not out:
        raise ValueError("normalized shift set is empty")
    return out


def bipartite_circulant_offsets(half_n: int, shifts: Sequence[int]) -> List[int]:
    """
    Expand step sizes S to undirected-offset set D = {+k, -k mod half_n}.
    This is the effective neighborhood used from each A-side vertex.
    """
    shifts_norm = normalize_bipartite_circulant_shifts(half_n, shifts)
    offsets = set()
    for k in shifts_norm:
        offsets.add(int(k) % int(half_n))
        offsets.add((-int(k)) % int(half_n))
    return sorted(offsets)


def validate_bipartite_circulant_spec(n: int, shifts: Sequence[int]) -> List[int]:
    """
    Validate (n, S) for bipartite circulant on two equal parts of size n/2.

    Vertices are indexed:
      - part A: 0..(n/2-1)
      - part B: n/2..(n-1)

    For each u in A and s in S, add edge:
      (u, n/2 + ((u + s) mod (n/2))).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n % 2 != 0:
        raise ValueError(f"n must be even for equal bipartition, got n={n}")
    half = n // 2
    return normalize_bipartite_circulant_shifts(half, shifts)


def bipartite_circulant_graph(n: int, shifts: Sequence[int]) -> np.ndarray:
    shifts_norm = validate_bipartite_circulant_spec(n, shifts)
    half = n // 2
    offsets = bipartite_circulant_offsets(half, shifts_norm)
    adj = np.zeros((n, n), dtype=bool)

    for u in range(half):
        for d in offsets:
            v = half + ((u + d) % half)
            adj[u, v] = True
            adj[v, u] = True
    return adj


def bipartite_circulant_edge_count(n: int, shifts: Sequence[int]) -> int:
    shifts_norm = validate_bipartite_circulant_spec(n, shifts)
    half = n // 2
    offsets = bipartite_circulant_offsets(half, shifts_norm)
    return int(half * len(offsets))


def bipartite_circulant_density(n: int, shifts: Sequence[int]) -> float:
    m = bipartite_circulant_edge_count(n, shifts)
    return float(m) / float(max_edges(n))


def bipartite_circulant_m_density_lambda2(
    n: int,
    shifts: Sequence[int],
) -> Tuple[int, float, float]:
    """
    Convenience helper returning (m, density, lambda2).
    """
    adj = bipartite_circulant_graph(n, shifts)
    m = int(edge_count(adj))
    rho = float(m) / float(max_edges(n))
    # Local import avoids making graph_core depend eagerly on scipy stack.
    from .spectral import algebraic_connectivity

    lam2 = float(algebraic_connectivity(adj))
    return m, rho, lam2


def normalize_bipartite_zn_shifts(n: int, shifts: Sequence[int]) -> List[int]:
    """
    Normalize shifts for Z_n parity-bipartite construction.
    Allowed shifts are odd residues mod n, so edges always cross parity classes.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n % 2 != 0:
        raise ValueError(f"n must be even for parity bipartition, got n={n}")
    if not shifts:
        raise ValueError("shifts must be non-empty")

    out = sorted({int(s) % int(n) for s in shifts})
    if not out:
        raise ValueError("normalized shift set is empty")
    bad = [s for s in out if s % 2 == 0]
    if bad:
        raise ValueError(
            f"all shifts must be odd modulo n for parity-bipartite graph, got even residues: {bad}"
        )
    return out


def bipartite_circulant_zn_offsets(n: int, shifts: Sequence[int]) -> List[int]:
    """
    Effective undirected offsets on Z_n: D = {+k, -k mod n}.
    """
    shifts_norm = normalize_bipartite_zn_shifts(n, shifts)
    offsets = set()
    for k in shifts_norm:
        offsets.add(int(k) % int(n))
        offsets.add((-int(k)) % int(n))
    # keep nonzero odd offsets only
    offsets = {d for d in offsets if d % 2 == 1 and d % n != 0}
    if not offsets:
        raise ValueError("effective Z_n offset set is empty after +/- expansion")
    return sorted(offsets)


def validate_bipartite_circulant_zn_spec(n: int, shifts: Sequence[int]) -> List[int]:
    """
    Validate (n, S) for Z_n parity-bipartite construction.

    Vertices are 0..n-1, bipartitioned by parity (even/odd).
    For each i and each odd offset d in effective set D={+S,-S}, add edge:
      (i, i+d mod n).
    """
    return normalize_bipartite_zn_shifts(n, shifts)


def bipartite_circulant_zn_graph(n: int, shifts: Sequence[int]) -> np.ndarray:
    shifts_norm = validate_bipartite_circulant_zn_spec(n, shifts)
    offsets = bipartite_circulant_zn_offsets(n, shifts_norm)
    adj = np.zeros((n, n), dtype=bool)
    for u in range(n):
        for d in offsets:
            v = (u + d) % n
            if (u % 2) == (v % 2):
                raise ValueError(
                    f"invalid odd-offset construction produced same-parity edge ({u},{v})"
                )
            adj[u, v] = True
            adj[v, u] = True
    return adj


def bipartite_circulant_zn_edge_count(n: int, shifts: Sequence[int]) -> int:
    shifts_norm = validate_bipartite_circulant_zn_spec(n, shifts)
    offsets = bipartite_circulant_zn_offsets(n, shifts_norm)
    # regular graph with degree len(offsets)
    return int((n * len(offsets)) // 2)


def bipartite_circulant_zn_density(n: int, shifts: Sequence[int]) -> float:
    m = bipartite_circulant_zn_edge_count(n, shifts)
    return float(m) / float(max_edges(n))


def bipartite_circulant_zn_m_density_lambda2(
    n: int,
    shifts: Sequence[int],
) -> Tuple[int, float, float]:
    adj = bipartite_circulant_zn_graph(n, shifts)
    m = int(edge_count(adj))
    rho = float(m) / float(max_edges(n))
    from .spectral import algebraic_connectivity

    lam2 = float(algebraic_connectivity(adj))
    return m, rho, lam2
