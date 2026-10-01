"""Portable classical-baseline sweeps, without the retired RL training stack.

Paper method identifiers, numerical calls, seed offsets, trajectory reuse, and
CSV fields follow the evaluated baseline runner. Relative output handling is
provided by the manifest CLI. No PyTorch or checkpoint loading is required.
"""
from __future__ import annotations
import csv
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple
import numpy as np
from .baselines import (fiedler_greedy_complete, mdmd_complete, mac_complete,
    wts_complete, kopt_exchange_complete, sdp_step_rounding_complete,
    oa_exact_global, local_global_maximizer_complete)
from .graph_core import edge_count, max_edges, path_graph
from .spectral import algebraic_connectivity, lambda2_subspace_edge_scores

DENSITY_SWEEP_METHODS = {
    "path_fvg": ("path", "fvg"), "path_mac": ("path", "mac"),
    "path_mdmd": ("path", "mdmd"), "path_wts": ("path", "wts"),
    "path_kopt": ("path", "kopt"), "path_sdps": ("path", "sdp_step"),
    "global_oa": ("path", "oa_exact_global"),
    "path_lgm": ("path", "lgm"), "path_lgm_raw": ("path", "lgm_raw"),
    "path_lgm_fill": ("path", "lgm_fill"),
}
METHOD_LABELS = {"path_fvg": "FVG", "path_mac": "MAC", "path_mdmd": "MDMD",
    "path_wts": "WTS", "path_kopt": "k-opt", "path_sdps": "SDP-step",
    "global_oa": "OA-PMC", "path_lgm": "EIG", "path_lgm_raw": "EIG",
    "path_lgm_fill": "EIG+fill"}

def method_label(method: str, completion: str) -> str:
    return METHOD_LABELS.get(method, method)

def _init_graph(n: int, m: int, init_mode_override: str = "path",
                init_seed_offset: int = 0) -> np.ndarray:
    if init_mode_override != "path":
        raise ValueError("The paper baseline runner uses path initialization.")
    return path_graph(n)

def density_to_m(n: int, rho: float) -> int:
    rho = float(np.clip(rho, 0.0, 1.0))
    m_lo = n - 1
    m_hi = max_edges(n)
    return int(np.clip(round(m_lo + rho * (m_hi - m_lo)), m_lo, m_hi))

def _complete_with_method(init_adj: np.ndarray, m_target: int, completion_method: str, wts_seed: int=0, wts_iterations: int=20, wts_tabu_tenure: int=16, wts_max_old_edges_per_iter: int=8, wts_neighbor_sample_per_old: int=3, wts_random_jump_per_old: int=1, wts_init_mode: str='fvg', mdmd_seed: int=0, mdmd_random_tie: bool=False, mac_fw_iters: int=20, mac_duality_gap_tol: float=1e-06, mac_init_mode: str='fiedler_topk', mac_seed: int=0, kopt_k: int=1, kopt_m: int=20, kopt_max_rounds: int=20, kopt_combo_cap_add: int=500, kopt_combo_cap_del: int=500, kopt_seed: int=0, oa_solver: str='HIGHS', oa_max_iters: int=60, oa_tol: float=1e-06, oa_verbose: bool=False, oa_warm_start_fvg: bool=True, sdp_solver: str='SCS', sdp_max_iters: int=10000, sdp_eps: float=1e-05, sdp_verbose: bool=False) -> np.ndarray:
    if completion_method == 'fvg':
        return fiedler_greedy_complete(init_adj, m_target)
    if completion_method == 'mdmd':
        return mdmd_complete(init_adj, m_target, use_ec=True, random_tie=bool(mdmd_random_tie), seed=int(mdmd_seed))
    if completion_method == 'lgm':
        return local_global_maximizer_complete(init_adj, m_target, exact_budget=False)
    if completion_method == 'lgm_raw':
        return local_global_maximizer_complete(init_adj, m_target, exact_budget=False)
    if completion_method == 'lgm_fill':
        return local_global_maximizer_complete(init_adj, m_target, exact_budget=True)
    if completion_method == 'mac':
        return mac_complete(init_adj, m_target, fw_iters=int(mac_fw_iters), duality_gap_tol=float(mac_duality_gap_tol), init_mode=str(mac_init_mode).strip().lower(), seed=int(mac_seed))
    if completion_method == 'kopt':
        return kopt_exchange_complete(init_adj, m_target, k=int(kopt_k), m=int(kopt_m), max_rounds=int(kopt_max_rounds), combo_cap_add=int(kopt_combo_cap_add), combo_cap_del=int(kopt_combo_cap_del), seed=int(kopt_seed))
    if completion_method == 'wts':
        return wts_complete(init_adj, m_target, seed=int(wts_seed), iterations=int(wts_iterations), tabu_tenure=int(wts_tabu_tenure), max_old_edges_per_iter=int(wts_max_old_edges_per_iter), neighbor_sample_per_old=int(wts_neighbor_sample_per_old), random_jump_per_old=int(wts_random_jump_per_old), init_mode=str(wts_init_mode).strip().lower())
    if completion_method == 'oa_exact_global':
        return oa_exact_global(int(init_adj.shape[0]), m_target, solver=str(oa_solver).strip(), max_iters=int(oa_max_iters), tol=float(oa_tol), verbose=bool(oa_verbose), warm_start_fvg=bool(oa_warm_start_fvg))
    if completion_method == 'sdp_step':
        return sdp_step_rounding_complete(init_adj, m_target, solver=str(sdp_solver).strip(), max_iters=int(sdp_max_iters), eps=float(sdp_eps), verbose=bool(sdp_verbose))
    raise ValueError(f'unknown completion_method: {completion_method}')

def evaluate_density_sweep(n_values: Sequence[int], methods: Sequence[str] | None=None, rho_steps: int=101, rho_values: Sequence[float] | None=None, wts_seed: int=0, wts_iterations: int=20, wts_tabu_tenure: int=16, wts_max_old_edges_per_iter: int=8, wts_neighbor_sample_per_old: int=3, wts_random_jump_per_old: int=1, wts_init_mode: str='fvg', mdmd_seed: int=0, mdmd_random_tie: bool=False, mac_fw_iters: int=20, mac_duality_gap_tol: float=1e-06, mac_init_mode: str='fiedler_topk', mac_seed: int=0, kopt_k: int=1, kopt_m: int=20, kopt_max_rounds: int=20, kopt_combo_cap_add: int=500, kopt_combo_cap_del: int=500, kopt_seed: int=0, oa_solver: str='HIGHS', oa_max_iters: int=60, oa_tol: float=1e-06, oa_verbose: bool=False, oa_warm_start_fvg: bool=True, sdp_solver: str='SCS', sdp_max_iters: int=10000, sdp_eps: float=1e-05, sdp_verbose: bool=False, repeats: int=1, progress: bool=False, progress_every: int=100) -> List[Dict[str, float | int | str]]:
    """
    Evaluate warm-start + completion methods over rho in [0,1].
    rho is interpreted on the feasible connected-edge scale:
      m = round((n-1) + rho * (n(n-1)/2 - (n-1))).
    """
    if methods is None:
        methods = ['path_fvg', 'path_mac', 'path_mdmd', 'path_wts', 'path_kopt']
    if rho_values is None:
        rho_arr = np.linspace(0.0, 1.0, max(2, int(rho_steps)), dtype=np.float64)
    else:
        rho_arr = np.asarray(list(rho_values), dtype=np.float64)
    rows: List[Dict[str, float | int | str]] = []
    rep_count = max(1, int(repeats))
    total_jobs = int(len(n_values) * len(rho_arr) * len(methods) * rep_count)
    done_jobs = 0
    t_all_start = time.perf_counter()

    def _progress_log(cur_n: int, cur_rho_conn: float, cur_density: float, cur_m: int, cur_method: str, cur_rep: int) -> None:
        if not bool(progress):
            return
        if total_jobs <= 0:
            return
        pct = 100.0 * float(done_jobs) / float(total_jobs)
        elapsed = max(1e-12, float(time.perf_counter() - t_all_start))
        rate = float(done_jobs) / elapsed
        rem = max(0, int(total_jobs - done_jobs))
        eta = float(rem) / max(1e-12, rate)
        print(f'[density_sweep] {done_jobs}/{total_jobs} ({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s n={cur_n} rho_conn={cur_rho_conn:.4f} density={cur_density:.4f} m={cur_m} method={cur_method} rep={cur_rep}', flush=True)

    def _append_row(*, n: int, rho_conn: float, m: int, m_actual: int, rep: int, method: str, init_mode: str, completion: str, init_e: int, lam2: float, runtime_init_s: float, runtime_complete_s: float, runtime_lambda2_s: float) -> None:
        nonlocal done_jobs
        m_lo = int(n - 1)
        m_hi = int(max_edges(n))
        if m_hi > 0:
            density = float(m) / float(m_hi)
            density_actual = float(m_actual) / float(m_hi)
        else:
            density = 0.0
            density_actual = 0.0
        if m_hi > m_lo:
            rho_conn_actual = float((m_actual - m_lo) / float(m_hi - m_lo))
        else:
            rho_conn_actual = 0.0
        rows.append({'n': int(n), 'density': float(density), 'rho': float(density), 'm': int(m), 'density_actual': float(density_actual), 'rho_actual': float(density_actual), 'rho_conn': float(rho_conn), 'rho_conn_actual': float(rho_conn_actual), 'm_max': int(m_hi), 'm_actual': int(m_actual), 'repeat_idx': int(rep), 'method': method, 'method_label': method_label(method, completion), 'warm_start_mode': init_mode, 'completion_method': completion, 'init_edges': int(init_e), 'residual_budget': int(m - init_e), 'lambda2': float(lam2), 'runtime_init_s': float(runtime_init_s), 'runtime_complete_s': float(runtime_complete_s), 'runtime_lambda2_s': float(runtime_lambda2_s), 'runtime_total_s': float(runtime_init_s + runtime_complete_s + runtime_lambda2_s)})
        done_jobs += 1
        pe = max(1, int(progress_every))
        if done_jobs == 1 or done_jobs == total_jobs or done_jobs % pe == 0:
            _progress_log(cur_n=int(n), cur_rho_conn=float(rho_conn), cur_density=float(density), cur_m=int(m), cur_method=str(method), cur_rep=int(rep))

    def _can_use_fvg_trajectory(init_mode: str, completion: str) -> bool:
        return completion == 'fvg' and init_mode == 'path'

    def _can_use_mdmd_trajectory(init_mode: str, completion: str) -> bool:
        return completion == 'mdmd' and init_mode == 'path' and (not mdmd_random_tie)
    for n in n_values:
        rho_meta: List[Tuple[float, int]] = []
        for rho in rho_arr:
            rho_f = float(np.clip(rho, 0.0, 1.0))
            m = int(density_to_m(n, rho_f))
            rho_meta.append((rho_f, m))
        for method in methods:
            if method not in DENSITY_SWEEP_METHODS:
                raise ValueError(f"unknown method '{method}'. supported: {sorted(DENSITY_SWEEP_METHODS.keys())}")
            init_mode, completion = DENSITY_SWEEP_METHODS[method]
            for rep in range(rep_count):
                seed_off = int(9973 * rep)
                tree_seed_off = 0
                if _can_use_fvg_trajectory(init_mode, completion):
                    t0 = time.perf_counter()
                    init_adj = _init_graph(n, n - 1, init_mode_override=init_mode, init_seed_offset=int(tree_seed_off))
                    t1 = time.perf_counter()
                    init_e = int(edge_count(init_adj))
                    work = init_adj.copy()
                    n_i = int(work.shape[0])
                    iu, iv = np.triu_indices(n_i, k=1)
                    available = ~work[iu, iv]
                    cur_edges = int(init_e)
                    comp_accum = 0.0
                    uniq_m = sorted({int(m) for _, m in rho_meta})
                    per_m: Dict[int, Tuple[int, float, float, float]] = {}
                    for m_t in uniq_m:
                        while cur_edges < int(m_t):
                            cand_idx = np.flatnonzero(available)
                            if int(cand_idx.size) == 0:
                                break
                            uv = np.stack([iu[cand_idx], iv[cand_idx]], axis=1).astype(np.int64, copy=False)
                            t_step0 = time.perf_counter()
                            _, scores = lambda2_subspace_edge_scores(work, uv)
                            pick_local = int(np.argmax(scores))
                            pick = int(cand_idx[pick_local])
                            u = int(iu[pick])
                            v = int(iv[pick])
                            work[u, v] = True
                            work[v, u] = True
                            available[pick] = False
                            cur_edges += 1
                            comp_accum += float(time.perf_counter() - t_step0)
                        t_l0 = time.perf_counter()
                        lam2 = float(algebraic_connectivity(work))
                        t_l1 = time.perf_counter()
                        per_m[int(m_t)] = (int(cur_edges), float(lam2), float(comp_accum), float(t_l1 - t_l0))
                    for rho_f, m in rho_meta:
                        m_actual, lam2, rt_comp, rt_lam = per_m[int(m)]
                        _append_row(n=int(n), rho_conn=float(rho_f), m=int(m), m_actual=int(m_actual), rep=int(rep), method=str(method), init_mode=str(init_mode), completion=str(completion), init_e=int(init_e), lam2=float(lam2), runtime_init_s=float(t1 - t0), runtime_complete_s=float(rt_comp), runtime_lambda2_s=float(rt_lam))
                    continue
                if _can_use_mdmd_trajectory(init_mode, completion):
                    t0 = time.perf_counter()
                    init_adj = _init_graph(n, n - 1, init_mode_override=init_mode, init_seed_offset=int(tree_seed_off))
                    t1 = time.perf_counter()
                    init_e = int(edge_count(init_adj))
                    work = init_adj.copy()
                    comp_accum = 0.0
                    use_ec = bool(completion == 'mdmd')
                    uniq_m = sorted({int(m) for _, m in rho_meta})
                    per_m: Dict[int, Tuple[int, float, float, float]] = {}
                    for m_t in uniq_m:
                        t_step0 = time.perf_counter()
                        work = mdmd_complete(work, int(m_t), use_ec=bool(use_ec), random_tie=False, seed=int(mdmd_seed))
                        comp_accum += float(time.perf_counter() - t_step0)
                        t_l0 = time.perf_counter()
                        lam2 = float(algebraic_connectivity(work))
                        t_l1 = time.perf_counter()
                        per_m[int(m_t)] = (int(edge_count(work)), float(lam2), float(comp_accum), float(t_l1 - t_l0))
                    for rho_f, m in rho_meta:
                        m_actual, lam2, rt_comp, rt_lam = per_m[int(m)]
                        _append_row(n=int(n), rho_conn=float(rho_f), m=int(m), m_actual=int(m_actual), rep=int(rep), method=str(method), init_mode=str(init_mode), completion=str(completion), init_e=int(init_e), lam2=float(lam2), runtime_init_s=float(t1 - t0), runtime_complete_s=float(rt_comp), runtime_lambda2_s=float(rt_lam))
                    continue
                for rho_f, m in rho_meta:
                    t0 = time.perf_counter()
                    init_adj = _init_graph(n, m, init_mode_override=init_mode, init_seed_offset=int(tree_seed_off))
                    t1 = time.perf_counter()
                    init_e = edge_count(init_adj)
                    final_adj = _complete_with_method(init_adj.copy(), m_target=m, completion_method=completion, wts_seed=int(wts_seed + 100003 * int(n) + int(m) + seed_off), wts_iterations=wts_iterations, wts_tabu_tenure=wts_tabu_tenure, wts_max_old_edges_per_iter=wts_max_old_edges_per_iter, wts_neighbor_sample_per_old=wts_neighbor_sample_per_old, wts_random_jump_per_old=wts_random_jump_per_old, wts_init_mode=wts_init_mode, mdmd_seed=int(mdmd_seed + 29993 * int(n) + int(m) + seed_off), mdmd_random_tie=mdmd_random_tie, mac_fw_iters=mac_fw_iters, mac_duality_gap_tol=mac_duality_gap_tol, mac_init_mode=mac_init_mode, mac_seed=int(mac_seed + 53633 * int(n) + int(m) + seed_off), kopt_k=kopt_k, kopt_m=kopt_m, kopt_max_rounds=kopt_max_rounds, kopt_combo_cap_add=kopt_combo_cap_add, kopt_combo_cap_del=kopt_combo_cap_del, kopt_seed=int(kopt_seed + 67097 * int(n) + int(m) + seed_off), oa_solver=oa_solver, oa_max_iters=oa_max_iters, oa_tol=oa_tol, oa_verbose=oa_verbose, oa_warm_start_fvg=oa_warm_start_fvg, sdp_solver=sdp_solver, sdp_max_iters=sdp_max_iters, sdp_eps=sdp_eps, sdp_verbose=sdp_verbose)
                    t2 = time.perf_counter()
                    t_l0 = time.perf_counter()
                    lam2 = float(algebraic_connectivity(final_adj))
                    t_l1 = time.perf_counter()
                    _append_row(n=int(n), rho_conn=float(rho_f), m=int(m), m_actual=int(edge_count(final_adj)), rep=int(rep), method=str(method), init_mode=str(init_mode), completion=str(completion), init_e=int(init_e), lam2=float(lam2), runtime_init_s=float(t1 - t0), runtime_complete_s=float(t2 - t1), runtime_lambda2_s=float(t_l1 - t_l0))
    return rows

def write_csv(rows: Sequence[Dict[str, float | int]], path: str) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
