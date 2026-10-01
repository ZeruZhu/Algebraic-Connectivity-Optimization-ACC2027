"""Paper baseline algorithms, retained from the evaluated implementation.

Only the numerical dependency closure of the ACC2027 baselines is packaged.
Optimization functions and their numerical helpers are unchanged.
"""
from __future__ import annotations
from collections import deque
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from scipy import sparse
from .graph_core import (complement_graph, Edge, edge_count, is_connected,
                         max_edges, non_edges, path_graph, star_graph)
from .spectral import (algebraic_connectivity, lambda2_subspace_edge_scores,
                       laplacian, two_smallest_nontrivial)

def _bfs_distances(adj: np.ndarray, src: int) -> np.ndarray:
    n = int(adj.shape[0])
    dist = -np.ones(n, dtype=np.int64)
    q: deque[int] = deque()
    dist[int(src)] = 0
    q.append(int(src))
    while q:
        u = int(q.popleft())
        du = int(dist[u])
        nbrs = np.flatnonzero(adj[u])
        for v in nbrs.tolist():
            v = int(v)
            if dist[v] >= 0:
                continue
            dist[v] = du + 1
            q.append(v)
    return dist


def _choose_with_ec(
    candidates: np.ndarray,
    ec: np.ndarray | None,
    *,
    random_tie: bool,
    rng: np.random.Generator | None,
) -> int:
    cand = np.asarray(candidates, dtype=np.int64).reshape(-1)
    if cand.size == 0:
        raise RuntimeError("empty candidate set")
    if ec is None:
        best = cand
    else:
        vals = np.asarray(ec[cand], dtype=np.float64)
        mn = float(vals.min())
        best = cand[np.flatnonzero(vals <= (mn + 1e-12))]
    if best.size == 1:
        return int(best[0])
    if bool(random_tie) and rng is not None:
        return int(best[int(rng.integers(0, int(best.size)))])
    return int(best.min())


def mdmd_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    use_ec: bool = True,
    random_tie: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """
    Iterative MDMD edge-addition heuristic (paper-inspired single-edge MACP repeated).

    At each step:
    1) choose a min-degree vertex i0 (with EC tie-break if use_ec=True),
    2) BFS from i0, collect farthest vertices,
    3) choose j0 among farthest (with EC tie-break if use_ec=True),
    4) add edge (i0, j0).
    """
    work = adj.copy()
    n = int(work.shape[0])
    m_hi = int(max_edges(n))
    target = int(min(int(m_target), m_hi))
    cur_edges = int(edge_count(work))
    if cur_edges >= target:
        return work

    rng = np.random.default_rng(int(seed)) if bool(random_tie) else None
    while cur_edges < target:
        deg = work.sum(axis=1).astype(np.int64)
        ec: np.ndarray | None = None
        if bool(use_ec):
            # Extendibility centrality: sum of neighbor degrees.
            ec = (work.astype(np.int64) @ deg).astype(np.int64)

        mdeg = int(deg.min())
        M = np.flatnonzero(deg == mdeg).astype(np.int64)
        i0 = _choose_with_ec(M, ec, random_tie=bool(random_tie), rng=rng)

        dist = _bfs_distances(work, i0)
        non_i0 = np.flatnonzero(~work[i0]).astype(np.int64)
        non_i0 = non_i0[non_i0 != int(i0)]
        if non_i0.size == 0:
            break

        d_non = dist[non_i0]
        d_max = int(d_non.max())
        far_non = non_i0[np.flatnonzero(d_non == d_max)].astype(np.int64)
        if far_non.size == 0:
            break
        j0 = _choose_with_ec(far_non, ec, random_tie=bool(random_tie), rng=rng)

        if bool(work[i0, j0]):
            # Defensive fallback: this should not happen because j0 is picked
            # from non-neighbors of i0. If it does, choose any remaining non-edge.
            iu, iv = np.where(np.triu(~work, k=1))
            if int(iu.size) == 0:
                break
            if bool(random_tie) and rng is not None:
                k = int(rng.integers(0, int(iu.size)))
            else:
                k = 0
            i0, j0 = int(iu[k]), int(iv[k])
        work[i0, j0] = True
        work[j0, i0] = True
        cur_edges += 1
    return work


def _union_complete_components_graph(
    n: int,
    component_sizes: Sequence[int],
) -> np.ndarray:
    sizes = [int(s) for s in component_sizes if int(s) > 0]
    used = int(sum(sizes))
    if used > int(n):
        raise ValueError(f"component sizes exceed n: used={used}, n={n}, sizes={sizes}")
    # Remaining vertices are isolated K1 components.
    if used < int(n):
        sizes = sizes + [1] * (int(n) - used)

    adj = np.zeros((int(n), int(n)), dtype=bool)
    start = 0
    for s in sizes:
        if s >= 2:
            blk = np.ones((s, s), dtype=bool)
            np.fill_diagonal(blk, False)
            adj[start : start + s, start : start + s] = blk
        start += s
    return adj


def _lgm_paper_component_sizes(
    n: int,
    m_desired: int,
) -> Tuple[List[int], int]:
    """
    Implement Algorithm 1 from:
      "Algebraic Connectivity: Local and Global Maximizer Graphs" (2023)
    for constructing a union-of-complete-components graph.

    Returns:
      (component_sizes, m_actual), with m_actual <= m_desired.
    """
    n_i_rem = int(n)
    m_i_rem = int(m_desired)
    q = 2
    sizes: List[int] = []

    def c2(x: int) -> int:
        return int(x * (x - 1) // 2)

    while (m_i_rem > 0) and (n_i_rem > 0) and (q <= n_i_rem):
        l_max = int(n_i_rem // q)
        if m_i_rem > c2(n_i_rem):
            # "m_rem too high for any complete graph" branch from Algorithm 1.
            q_use = int(n_i_rem)
            sizes.append(q_use)
            n_i_rem -= q_use
            m_i_rem -= c2(q_use)
            q = 2
            continue

        cap = int(l_max * c2(q))
        if m_i_rem <= cap:
            e_q = int(c2(q))
            l = int(min(l_max, (m_i_rem // e_q) if e_q > 0 else 0))

            # Corner case in the paper pseudocode: if no q-size component fits,
            # reduce q by one and retry once.
            if l <= 0:
                if q > 2:
                    q -= 1
                    l_max = int(n_i_rem // q)
                    e_q = int(c2(q))
                    l = int(min(l_max, (m_i_rem // e_q) if e_q > 0 else 0))
                if l <= 0:
                    q += 1
                    continue

            sizes.extend([int(q)] * int(l))
            n_i_rem -= int(q * l)
            m_i_rem -= int(l * e_q)
            q = 2
        else:
            q += 1

    m_actual = int(m_desired - m_i_rem)
    return sizes, m_actual


def _fill_edges_balancing_degree(
    adj: np.ndarray,
    m_target: int,
) -> np.ndarray:
    """
    Add remaining edges while preferring low-degree endpoints.
    Used only when the paper's exact union-of-cliques construction
    under-fills the desired budget.
    """
    work = adj.copy()
    target = int(m_target)
    while edge_count(work) < target:
        cand = non_edges(work)
        if not cand:
            break
        uv = np.asarray(cand, dtype=np.int64)
        deg = work.sum(axis=1).astype(np.int64)
        u = uv[:, 0]
        v = uv[:, 1]
        s1 = np.maximum(deg[u], deg[v])
        best = np.flatnonzero(s1 == int(s1.min()))
        if int(best.size) > 1:
            s2 = deg[u[best]] + deg[v[best]]
            best = best[np.flatnonzero(s2 == int(s2.min()))]
        idx = int(best[0])
        a = int(uv[idx, 0])
        b = int(uv[idx, 1])
        work[a, b] = True
        work[b, a] = True
    return work


def local_global_maximizer_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    exact_budget: bool = True,
) -> np.ndarray:
    """
    Baseline from "Algebraic Connectivity: Local and Global Maximizer Graphs".

    Construct complement graph via Algorithm 1 (union of complete components
    minimizing Laplacian largest eigenvalue in their local/global sense), then
    return its complement as the ACM candidate graph.

    Notes:
    - The paper algorithm may return m_actual < m_desired for some (n,m).
    - If exact_budget=True, we fill the remaining complement-edge budget using
      a low-degree balancing rule to keep final edge count exact.
    """
    n = int(adj.shape[0])
    m = int(m_target)
    m_comp_target = int(max_edges(n) - m)
    if m_comp_target < 0:
        raise ValueError(f"Invalid target: n={n}, m_target={m_target}")

    comp_sizes, m_comp_actual = _lgm_paper_component_sizes(n=n, m_desired=m_comp_target)
    comp_adj = _union_complete_components_graph(n=n, component_sizes=comp_sizes)
    if edge_count(comp_adj) != int(m_comp_actual):
        raise RuntimeError(
            f"LGM construction mismatch: expected m_actual={m_comp_actual}, "
            f"built={edge_count(comp_adj)}"
        )

    if bool(exact_budget) and int(m_comp_actual) < int(m_comp_target):
        comp_adj = _fill_edges_balancing_degree(comp_adj, int(m_comp_target))

    out = complement_graph(comp_adj)
    return out


def _top_k_binary_vector(scores: np.ndarray, k: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    p = int(s.size)
    out = np.zeros(p, dtype=np.float64)
    kk = int(k)
    if kk <= 0 or p == 0:
        return out
    if kk >= p:
        out.fill(1.0)
        return out
    # Deterministic top-k under exact ties:
    # - all scores strictly above cutoff are always selected
    # - exact-cutoff ties are selected by increasing index
    part = np.argpartition(-s, kk - 1)[:kk]
    cutoff = float(s[part].min())
    strict_idx = np.flatnonzero(s > cutoff)
    need = int(kk - int(strict_idx.size))
    if need <= 0:
        idx = strict_idx[:kk]
    else:
        tie_idx = np.flatnonzero(s == cutoff)
        idx = np.concatenate([strict_idx, tie_idx[:need]])
    out[idx] = 1.0
    return out


def _fiedler_from_laplacian(L: np.ndarray) -> Tuple[float, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(L, dtype=np.float64))
    n = int(L.shape[0])
    if vals.size < 2:
        return 0.0, np.zeros(n, dtype=np.float64)
    lam2 = float(max(vals[1], 0.0))
    y = np.asarray(vecs[:, 1], dtype=np.float64)
    return lam2, y


def _laplacian_with_fractional_candidates(
    base_lap: np.ndarray,
    candidate_uv: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    L = np.asarray(base_lap, dtype=np.float64).copy()
    if candidate_uv.size == 0:
        return L
    uv = np.asarray(candidate_uv, dtype=np.int64)
    w = np.asarray(x, dtype=np.float64).reshape(-1)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"candidate_uv must have shape (p,2), got {uv.shape}")
    if int(w.size) != int(uv.shape[0]):
        raise ValueError(f"x length mismatch: len(x)={w.size}, p={uv.shape[0]}")
    u = uv[:, 0]
    v = uv[:, 1]
    np.add.at(L, (u, u), w)
    np.add.at(L, (v, v), w)
    np.add.at(L, (u, v), -w)
    np.add.at(L, (v, u), -w)
    return L


def mac_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    fw_iters: int = 20,
    duality_gap_tol: float = 1e-6,
    init_mode: str = "fiedler_topk",
    seed: int = 0,
) -> np.ndarray:
    """
    MAC (Maximizing Algebraic Connectivity) via Boolean relaxation + rounding.

    Paper-inspired implementation (Doherty et al., 2022):
    - Candidate set: all non-edges of the fixed base graph.
    - Solve relaxed selection x in [0,1]^p, 1^T x = K with Frank-Wolfe.
    - Round by selecting top-K entries of x.
    """
    base = adj.copy()
    q = int(m_target - edge_count(base))
    if q <= 0:
        return base

    cand = non_edges(base)
    p = int(len(cand))
    if p == 0:
        return base
    uv = np.asarray(cand, dtype=np.int64)
    if q >= p:
        out = base.copy()
        out[uv[:, 0], uv[:, 1]] = True
        out[uv[:, 1], uv[:, 0]] = True
        return out

    base_lap = laplacian(base)
    mode = str(init_mode).strip().lower()
    if mode not in ("fiedler_topk", "uniform", "random"):
        raise ValueError(
            f"Unsupported mac init_mode: {init_mode}. "
            "Use fiedler_topk|uniform|random."
        )

    if mode == "uniform":
        x = np.full(p, float(q) / float(p), dtype=np.float64)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(p)[:q]
        x = np.zeros(p, dtype=np.float64)
        x[idx] = 1.0
    else:
        _, y0 = _fiedler_from_laplacian(base_lap)
        score0 = (y0[uv[:, 0]] - y0[uv[:, 1]]) ** 2
        x = _top_k_binary_vector(score0, q)

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    ssum = float(x.sum())
    if ssum > 0.0 and abs(ssum - float(q)) > 1e-12:
        x = x * (float(q) / ssum)
        x = np.clip(x, 0.0, 1.0)

    max_iters = max(0, int(fw_iters))
    tol = float(duality_gap_tol)
    for t in range(max_iters):
        Lx = _laplacian_with_fractional_candidates(base_lap, uv, x)
        _, y = _fiedler_from_laplacian(Lx)
        grad = (y[uv[:, 0]] - y[uv[:, 1]]) ** 2

        s = _top_k_binary_vector(grad, q)
        gap = float(np.dot(grad, (s - x)))
        if tol > 0.0 and gap <= tol:
            break

        alpha = float(2.0 / (2.0 + float(t)))
        x = x + alpha * (s - x)
        x = np.clip(x, 0.0, 1.0)

    select = _top_k_binary_vector(x, q)
    sel_idx = np.flatnonzero(select > 0.5)
    out = base.copy()
    if int(sel_idx.size) > 0:
        sel_uv = uv[sel_idx]
        out[sel_uv[:, 0], sel_uv[:, 1]] = True
        out[sel_uv[:, 1], sel_uv[:, 0]] = True
    return out


def fiedler_greedy_complete(adj: np.ndarray, m_target: int) -> np.ndarray:
    work = adj.copy()
    target = int(m_target)
    cur_edges = int(edge_count(work))
    if cur_edges >= target:
        return work

    n = int(work.shape[0])
    iu, iv = np.triu_indices(n, k=1)
    available = ~work[iu, iv]
    while cur_edges < target:
        cand_idx = np.flatnonzero(available)
        if int(cand_idx.size) == 0:
            break
        uv = np.stack([iu[cand_idx], iv[cand_idx]], axis=1).astype(np.int64, copy=False)
        _, scores = lambda2_subspace_edge_scores(work, uv)
        pick_local = int(np.argmax(scores))
        pick = int(cand_idx[pick_local])
        u = int(iu[pick])
        v = int(iv[pick])
        work[u, v] = True
        work[v, u] = True
        available[pick] = False
        cur_edges += 1
    return work


def _fiedler_seed_candidate_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    k_add: int,
) -> List[int]:
    """Pick k candidate-edge indices using the same Fiedler greedy score."""
    work = adj.copy()
    p = int(candidate_uv.shape[0])
    available = np.ones(p, dtype=bool)
    selected: List[int] = []

    for _ in range(max(0, int(k_add))):
        idx_avail = np.flatnonzero(available)
        if idx_avail.size == 0:
            break
        uv = candidate_uv[idx_avail]
        _, scores = lambda2_subspace_edge_scores(work, uv)
        idx = int(idx_avail[int(np.argmax(scores))])
        selected.append(idx)
        available[idx] = False
        u, v = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        work[u, v] = True
        work[v, u] = True
    return selected


def _round_top_k_indices(x: np.ndarray, k_add: int) -> np.ndarray:
    k = int(k_add)
    idx = np.argpartition(-x, k - 1)[:k]
    # Stable order for reproducibility under ties.
    return idx[np.argsort(-x[idx], kind="mergesort")]


def _apply_candidate_edge_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    out = adj.copy()
    for sel in selected_idx.tolist():
        uu, vv = int(candidate_uv[sel, 0]), int(candidate_uv[sel, 1])
        out[uu, vv] = True
        out[vv, uu] = True
    return out


def _fiedler_rank_indices(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    *,
    descending: bool = True,
) -> np.ndarray:
    _, scores = lambda2_subspace_edge_scores(adj, candidate_uv)
    order = np.argsort(scores, kind="mergesort")
    return order[::-1] if descending else order


def _choose_edge_combos(
    pool: Sequence[int],
    k: int,
    max_combos: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    if k <= 0:
        return []
    if len(pool) < k:
        return []
    all_combos = list(combinations(pool, k))
    if max_combos <= 0 or len(all_combos) <= max_combos:
        return all_combos
    idx = rng.choice(len(all_combos), size=max_combos, replace=False)
    idx.sort()
    return [all_combos[int(i)] for i in idx.tolist()]


def kopt_exchange_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    k: int = 1,
    m: int = 20,
    max_rounds: int = 20,
    combo_cap_add: int = 300,
    combo_cap_del: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """
    Greedy k-opt exchange heuristic for augmentation from a fixed base graph.

    - Base graph edges are immutable.
    - A fixed number q of augment edges is maintained.
    - Each round: add k augment edges from top-ranked candidates and remove k
      augment edges from low-ranked selected edges, accepting improving swaps.
    """
    base = adj.copy()
    q = int(m_target - edge_count(base))
    if q <= 0:
        return base

    cand = non_edges(base)
    p = len(cand)
    if p == 0:
        return base
    if q >= p:
        out = base.copy()
        for uu, vv in cand:
            out[uu, vv] = True
            out[vv, uu] = True
        return out

    k_eff = max(1, int(k))
    pool_m = max(1, int(m))
    rng = np.random.default_rng(int(seed))
    candidate_uv = np.asarray(cand, dtype=np.int64)

    # Stage-1 initial feasible augmentation: exact sequential Fiedler-greedy
    # (same construction as the baseline path_fvg completion).
    selected = _fiedler_seed_candidate_indices(base, candidate_uv, q)
    if len(selected) < q:
        # Defensive fallback for degenerate numeric/tie cases.
        remaining = np.setdiff1d(np.arange(p, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        fill = min(int(q - len(selected)), int(remaining.size))
        if fill > 0:
            extra = remaining[:fill]
            selected.extend(int(x) for x in extra.tolist())

    selected_set = set(selected)
    current_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(selected, dtype=np.int64))
    current_lam2 = float(algebraic_connectivity(current_adj))

    for _ in range(max(0, int(max_rounds))):
        add_available = [idx for idx in range(p) if idx not in selected_set]
        if len(add_available) < k_eff:
            break

        # Add side ranking from current graph using Fiedler score.
        add_scores_order = _fiedler_rank_indices(
            current_adj,
            candidate_uv[np.asarray(add_available, dtype=np.int64)],
            descending=True,
        )
        add_pool = [add_available[int(i)] for i in add_scores_order[: min(pool_m, len(add_available))].tolist()]
        add_combos = _choose_edge_combos(add_pool, k_eff, int(combo_cap_add), rng)
        if not add_combos:
            break

        best_lam2 = current_lam2
        best_selected: List[int] | None = None

        for add_combo in add_combos:
            plus_set = set(selected_set)
            plus_set.update(int(x) for x in add_combo)
            plus_list = sorted(plus_set)
            plus_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(plus_list, dtype=np.int64))

            # Remove side ranking from augmented graph; choose low-score selected edges.
            plus_uv = candidate_uv[np.asarray(plus_list, dtype=np.int64)]
            rem_order_local = _fiedler_rank_indices(plus_adj, plus_uv, descending=False)
            rem_pool_local = [plus_list[int(i)] for i in rem_order_local[: min(pool_m, len(plus_list))].tolist()]
            rem_combos = _choose_edge_combos(rem_pool_local, k_eff, int(combo_cap_del), rng)
            if not rem_combos:
                continue

            for rem_combo in rem_combos:
                new_set = set(plus_set)
                for ridx in rem_combo:
                    new_set.discard(int(ridx))
                if len(new_set) != q:
                    continue
                new_list = sorted(new_set)
                new_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(new_list, dtype=np.int64))
                lam2_new = float(algebraic_connectivity(new_adj))
                if lam2_new > (best_lam2 + 1e-12):
                    best_lam2 = lam2_new
                    best_selected = new_list

        if best_selected is None:
            break
        selected = best_selected
        selected_set = set(selected)
        current_adj = _apply_candidate_edge_indices(base, candidate_uv, np.asarray(selected, dtype=np.int64))
        current_lam2 = best_lam2

    return current_adj


def _connected_components_nodes(adj: np.ndarray) -> List[List[int]]:
    n = int(adj.shape[0])
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for s in range(n):
        if bool(seen[s]):
            continue
        q: deque[int] = deque([int(s)])
        seen[s] = True
        comp: List[int] = []
        while q:
            u = int(q.popleft())
            comp.append(u)
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs.tolist():
                vv = int(v)
                if bool(seen[vv]):
                    continue
                seen[vv] = True
                q.append(vv)
        comps.append(comp)
    return comps


def oa_exact_global(
    n: int,
    m_target: int,
    *,
    solver: str = "HIGHS",
    max_iters: int = 120,
    tol: float = 1e-6,
    verbose: bool = False,
    warm_start_fvg: bool = True,
) -> np.ndarray:
    """
    Global OA exact solver over all m-edge connected graphs on n nodes.

    Decision variables are on all complete-graph edges, not augmentations
    from a fixed base graph. Connectivity is enforced via cut separation.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError("oa_exact_global solver requires cvxpy. Install with: pip install cvxpy") from exc

    n_i = int(n)
    m_tar = int(m_target)
    m_hi = int(max_edges(n_i))
    if n_i < 2:
        raise ValueError(f"n must be >= 2, got {n_i}")
    if m_tar < int(n_i - 1):
        raise ValueError(
            f"oa_exact_global assumes connected regime m>=n-1, got n={n_i}, m_target={m_tar}"
        )
    if m_tar > m_hi:
        raise ValueError(f"m_target exceeds complete graph edges: n={n_i}, m_target={m_tar}, m_max={m_hi}")
    if m_tar == m_hi:
        out = np.ones((n_i, n_i), dtype=bool)
        np.fill_diagonal(out, False)
        return out

    cand = list(combinations(range(n_i), 2))
    candidate_uv = np.asarray(cand, dtype=np.int64)
    p = int(candidate_uv.shape[0])

    J = np.eye(n_i, dtype=np.float64) - (np.ones((n_i, n_i), dtype=np.float64) / float(n_i))
    cuts_alpha: List[np.ndarray] = []
    # No base graph term here; beta is always 0 for global edge-selection.
    cuts_beta: List[float] = []

    def add_cut(v: np.ndarray) -> bool:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape[0] != n_i:
            return False
        denom = float(vv @ (J @ vv))
        if denom <= 1e-12:
            return False
        alpha = ((vv[candidate_uv[:, 0]] - vv[candidate_uv[:, 1]]) ** 2) / denom
        cuts_alpha.append(np.asarray(alpha, dtype=np.float64))
        cuts_beta.append(0.0)
        return True

    # Seed OA with a non-constant vector.
    v0 = np.linspace(-1.0, 1.0, num=n_i, dtype=np.float64)
    v0 -= float(v0.mean())
    if not add_cut(v0):
        raise RuntimeError("oa_exact_global failed to initialize eigenvector cuts.")

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("HIGHS", "SCIPY"):
            if name in installed:
                solver_upper = name
                break
        else:
            raise RuntimeError(f"No MILP-capable cvxpy solver found. Installed: {sorted(installed)}")
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested oa_exact_global solver '{solver_upper}' is not installed. Installed: {sorted(installed)}"
        )

    topol_cuts: List[np.ndarray] = []
    topol_cut_keys: set[Tuple[int, ...]] = set()

    best_adj: np.ndarray | None = None
    best_lam2 = -np.inf
    if bool(warm_start_fvg):
        warm1 = fiedler_greedy_complete(path_graph(n_i), m_tar)
        lam1 = float(algebraic_connectivity(warm1))
        warm2 = fiedler_greedy_complete(star_graph(n_i), m_tar)
        lam2 = float(algebraic_connectivity(warm2))
        if lam2 > lam1:
            best_adj = warm2
            best_lam2 = lam2
        else:
            best_adj = warm1
            best_lam2 = lam1

    last_adj: np.ndarray | None = None
    for _ in range(max(1, int(max_iters))):
        x = cp.Variable(p, boolean=True)
        gamma = cp.Variable()
        constraints = [
            cp.sum(x) == float(m_tar),
            gamma >= 0.0,
            gamma <= float(n_i),
        ]
        for alpha, beta in zip(cuts_alpha, cuts_beta):
            constraints.append(beta + alpha @ x >= gamma)
        for idx in topol_cuts:
            constraints.append(cp.sum(x[idx]) >= 1.0)

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve(solver=getattr(cp, solver_upper), verbose=bool(verbose))
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"oa_exact_global MILP failed with status={prob.status}")
        if x.value is None:
            raise RuntimeError("oa_exact_global MILP returned no x solution.")

        x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
        idx = np.flatnonzero(x_val > 0.5).astype(np.int64)
        if int(idx.size) != int(m_tar):
            idx = _round_top_k_indices(x_val, m_tar).astype(np.int64)

        cand_adj = np.zeros((n_i, n_i), dtype=bool)
        for sel in idx.tolist():
            uu = int(candidate_uv[int(sel), 0])
            vv = int(candidate_uv[int(sel), 1])
            cand_adj[uu, vv] = True
            cand_adj[vv, uu] = True
        last_adj = cand_adj

        # Separate connectivity cuts first: sum_{delta(S)} x >= 1 for each
        # disconnected component S in incumbent.
        if not bool(is_connected(cand_adj)):
            comps = _connected_components_nodes(cand_adj)
            added_topol = 0
            for comp in comps:
                s = tuple(sorted(int(v) for v in comp))
                if len(s) == 0 or len(s) == n_i:
                    continue
                comp_set = set(s)
                comp_bar = tuple(i for i in range(n_i) if i not in comp_set)
                key = min(s, comp_bar)
                if key in topol_cut_keys:
                    continue
                in_s_u = np.isin(candidate_uv[:, 0], np.asarray(s, dtype=np.int64))
                in_s_v = np.isin(candidate_uv[:, 1], np.asarray(s, dtype=np.int64))
                cross = np.flatnonzero(np.logical_xor(in_s_u, in_s_v)).astype(np.int64)
                if int(cross.size) <= 0:
                    continue
                topol_cuts.append(cross)
                topol_cut_keys.add(key)
                added_topol += 1
            if added_topol <= 0:
                break
            continue

        lam2_f, phi2, _ = two_smallest_nontrivial(cand_adj)
        lam2 = float(lam2_f)
        if best_adj is None or lam2 > (best_lam2 + 1e-12):
            best_lam2 = lam2
            best_adj = cand_adj

        ub = float(gamma.value) if gamma.value is not None else np.inf
        if ub <= (best_lam2 + float(tol)):
            break
        if not add_cut(phi2):
            break

    if best_adj is not None:
        return best_adj
    if last_adj is not None and bool(is_connected(last_adj)):
        return last_adj
    # Fallback should be connected and feasible in connected regime.
    return fiedler_greedy_complete(path_graph(n_i), m_tar)


def _laplacian_affine_operator(
    n: int,
    candidate_uv: np.ndarray,
) -> sparse.csc_matrix:
    """
    Build sparse A such that vec(L(x)) = vec(L0) + A @ x for candidate edges.
    """
    p = int(candidate_uv.shape[0])
    rows = np.empty(4 * p, dtype=np.int64)
    cols = np.empty(4 * p, dtype=np.int64)
    data = np.empty(4 * p, dtype=np.float64)
    for e in range(p):
        u = int(candidate_uv[e, 0])
        v = int(candidate_uv[e, 1])
        b = 4 * e
        rows[b : b + 4] = (
            u * n + u,
            v * n + v,
            u * n + v,
            v * n + u,
        )
        cols[b : b + 4] = (e, e, e, e)
        data[b : b + 4] = (1.0, 1.0, -1.0, -1.0)
    return sparse.csc_matrix((data, (rows, cols)), shape=(n * n, p))


def _solve_relaxed_sdp_x(
    adj: np.ndarray,
    candidate_uv: np.ndarray,
    k_add: int,
    *,
    solver: str = "SCS",
    max_iters: int = 10_000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve relaxed SDP and return the relaxed edge-selection vector x.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sdp_greedy completion requires cvxpy. Install with: pip install cvxpy"
        ) from exc

    p = int(candidate_uv.shape[0])
    if p <= 0:
        return np.zeros(0, dtype=np.float64)
    if k_add <= 0:
        return np.zeros(p, dtype=np.float64)
    if k_add >= p:
        return np.ones(p, dtype=np.float64)

    n = int(adj.shape[0])
    L0 = laplacian(adj).astype(np.float64, copy=False)
    J = np.eye(n, dtype=np.float64) - (np.ones((n, n), dtype=np.float64) / float(n))
    A = _laplacian_affine_operator(n, candidate_uv)

    x = cp.Variable(p)
    h = cp.Variable()
    vec_l0 = L0.reshape(n * n, order="C")
    vec_lx = vec_l0 + A @ x
    Lx = cp.reshape(vec_lx, (n, n), order="C")

    constraints = [
        x >= 0.0,
        x <= 1.0,
        cp.sum(x) == float(k_add),
        (Lx - h * J) >> 0,
    ]
    prob = cp.Problem(cp.Maximize(h), constraints)

    installed = {s.upper() for s in cp.installed_solvers()}
    solver_upper = str(solver).strip().upper()
    if solver_upper in ("", "AUTO"):
        for name in ("MOSEK", "CVXOPT", "SCS"):
            if name in installed:
                solver_upper = name
                break
        else:
            solver_upper = "SCS"
    if solver_upper not in installed:
        raise RuntimeError(
            f"Requested SDP solver '{solver_upper}' is not installed. "
            f"Installed: {sorted(installed)}"
        )

    solve_kwargs = {"verbose": bool(verbose)}
    if solver_upper == "SCS":
        solve_kwargs["max_iters"] = int(max_iters)
        solve_kwargs["eps"] = float(eps)

    prob.solve(solver=getattr(cp, solver_upper), **solve_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solve failed with status={prob.status}")
    if x.value is None:
        raise RuntimeError("SDP solve returned no x solution.")

    x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
    x_val = np.clip(x_val, 0.0, 1.0)
    return x_val


def sdp_step_rounding_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    solver: str = "SCS",
    max_iters: int = 10_000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Paper-faithful SDP step-by-step rounding:
    solve relaxed SDP, add one edge with largest x, and repeat.
    """
    work = adj.copy()
    while edge_count(work) < m_target:
        cand = non_edges(work)
        p = len(cand)
        if p == 0:
            break
        k_rem = int(m_target - edge_count(work))
        if k_rem <= 0:
            break
        candidate_uv = np.asarray(cand, dtype=np.int64)
        if k_rem >= p:
            for uu, vv in cand:
                work[uu, vv] = True
                work[vv, uu] = True
            break

        x_val = _solve_relaxed_sdp_x(
            work,
            candidate_uv,
            k_rem,
            solver=solver,
            max_iters=max_iters,
            eps=eps,
            verbose=verbose,
        )
        idx = int(np.argmax(x_val))
        uu, vv = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        work[uu, vv] = True
        work[vv, uu] = True
    return work


def wts_complete(
    adj: np.ndarray,
    m_target: int,
    *,
    seed: int = 0,
    iterations: int = 20,
    tabu_tenure: int = 16,
    max_old_edges_per_iter: int = 8,
    neighbor_sample_per_old: int = 3,
    random_jump_per_old: int = 1,
    init_mode: str = "fvg",
) -> np.ndarray:
    """
    Weighted Tabu Search-style completion for unweighted adjacency.

    This solver optimizes the set of k residual added edges jointly by one-edge
    replacement moves with tabu memory and aspiration.
    """
    k_add = int(m_target - edge_count(adj))
    if k_add <= 0:
        return adj.copy()

    cand = non_edges(adj)
    p = len(cand)
    if p == 0:
        return adj.copy()
    candidate_uv = np.asarray(cand, dtype=np.int64)
    if k_add >= p:
        out = adj.copy()
        out[candidate_uv[:, 0], candidate_uv[:, 1]] = True
        out[candidate_uv[:, 1], candidate_uv[:, 0]] = True
        return out

    n = int(adj.shape[0])
    rng = np.random.default_rng(int(seed))

    mode = str(init_mode).strip().lower()
    if mode not in ("fvg", "random"):
        raise ValueError(f"Unsupported wts init_mode: {init_mode}. Use fvg|random.")

    if mode == "fvg":
        selected = _fiedler_seed_candidate_indices(adj, candidate_uv, k_add)
        if len(selected) < k_add:
            remaining = np.setdiff1d(np.arange(p, dtype=np.int64), np.asarray(selected, dtype=np.int64))
            extra = rng.choice(remaining, size=k_add - len(selected), replace=False)
            selected.extend(int(x) for x in extra.tolist())
    else:
        selected = [int(x) for x in rng.choice(p, size=k_add, replace=False).tolist()]

    work = adj.copy()
    if selected:
        sel_uv0 = candidate_uv[np.asarray(selected, dtype=np.int64)]
        work[sel_uv0[:, 0], sel_uv0[:, 1]] = True
        work[sel_uv0[:, 1], sel_uv0[:, 0]] = True

    # Candidate-edge incidence map for local neighborhood proposals.
    incident: List[List[int]] = [[] for _ in range(n)]
    for idx in range(p):
        u, v = int(candidate_uv[idx, 0]), int(candidate_uv[idx, 1])
        incident[u].append(idx)
        incident[v].append(idx)

    selected_set = set(selected)
    idx_to_pos = {idx: pos for pos, idx in enumerate(selected)}
    current_lam2 = float(algebraic_connectivity(work))
    best_lam2 = current_lam2
    best_selected = selected.copy()

    tabu_q: deque[Tuple[int, int]] = deque()
    tabu_s: set[Tuple[int, int]] = set()
    tenure = max(1, int(tabu_tenure))
    tol = 1e-12

    for _ in range(max(0, int(iterations))):
        if not selected:
            break
        old_count = min(len(selected), max(1, int(max_old_edges_per_iter)))
        if old_count == len(selected):
            old_candidates = selected.copy()
        else:
            old_candidates = [
                int(x)
                for x in rng.choice(np.asarray(selected, dtype=np.int64), size=old_count, replace=False).tolist()
            ]

        best_move: Tuple[int, int] | None = None
        best_move_lam2 = -np.inf

        for old_idx in old_candidates:
            u_old, v_old = int(candidate_uv[old_idx, 0]), int(candidate_uv[old_idx, 1])
            local_pool: set[int] = set()
            for node in (u_old, v_old):
                nbrs = incident[node]
                if not nbrs:
                    continue
                take = min(len(nbrs), max(1, int(neighbor_sample_per_old)))
                if take == len(nbrs):
                    sampled = nbrs
                else:
                    sampled = rng.choice(
                        np.asarray(nbrs, dtype=np.int64),
                        size=take,
                        replace=False,
                    ).tolist()
                for idx in sampled:
                    local_pool.add(int(idx))

            for _ in range(max(0, int(random_jump_per_old))):
                local_pool.add(int(rng.integers(0, p)))

            for new_idx in local_pool:
                if new_idx == old_idx or new_idx in selected_set:
                    continue
                move = (old_idx, new_idx)

                u_new, v_new = int(candidate_uv[new_idx, 0]), int(candidate_uv[new_idx, 1])
                work[u_old, v_old] = False
                work[v_old, u_old] = False
                work[u_new, v_new] = True
                work[v_new, u_new] = True
                lam2 = float(algebraic_connectivity(work))
                work[u_new, v_new] = False
                work[v_new, u_new] = False
                work[u_old, v_old] = True
                work[v_old, u_old] = True

                is_tabu = move in tabu_s
                if is_tabu and lam2 <= (best_lam2 + tol):
                    continue
                if lam2 > (best_move_lam2 + tol):
                    best_move = move
                    best_move_lam2 = lam2

        if best_move is None:
            break

        old_idx, new_idx = best_move
        u_old, v_old = int(candidate_uv[old_idx, 0]), int(candidate_uv[old_idx, 1])
        u_new, v_new = int(candidate_uv[new_idx, 0]), int(candidate_uv[new_idx, 1])
        work[u_old, v_old] = False
        work[v_old, u_old] = False
        work[u_new, v_new] = True
        work[v_new, u_new] = True

        pos = idx_to_pos.pop(old_idx)
        selected[pos] = new_idx
        idx_to_pos[new_idx] = pos
        selected_set.remove(old_idx)
        selected_set.add(new_idx)
        current_lam2 = best_move_lam2
        if current_lam2 > (best_lam2 + tol):
            best_lam2 = current_lam2
            best_selected = selected.copy()

        reverse_move = (new_idx, old_idx)
        if len(tabu_q) >= tenure:
            expired = tabu_q.popleft()
            tabu_s.discard(expired)
        tabu_q.append(reverse_move)
        tabu_s.add(reverse_move)

    out = adj.copy()
    if best_selected:
        best_uv = candidate_uv[np.asarray(best_selected, dtype=np.int64)]
        out[best_uv[:, 0], best_uv[:, 1]] = True
        out[best_uv[:, 1], best_uv[:, 0]] = True
    return out


def random_complete(adj: np.ndarray, m_target: int, rng: np.random.Generator) -> np.ndarray:
    work = adj.copy()
    while edge_count(work) < m_target:
        non = non_edges(work)
        if not non:
            break
        idx = int(rng.integers(0, len(non)))
        u, v = non[idx]
        work[u, v] = True
        work[v, u] = True
    return work
