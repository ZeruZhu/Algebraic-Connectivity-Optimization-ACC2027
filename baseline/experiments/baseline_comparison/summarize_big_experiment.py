from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _to_int(v: str) -> int:
    return int(float(v))


def _to_float(v: str) -> float:
    return float(v)


def _method_name(raw: str) -> str:
    m = str(raw).strip()
    mapping = {
        "path_fvg": "FVG",
        "path_mac": "MAC",
        "path_mdmd": "MDMD",
        "path_wts": "WTS",
        "path_kopt": "k-opt",
        "path_sdps": "SDP-step",
        "global_oa": "OA-PMC",
    }
    return mapping.get(m, m)


def _method_order(name: str) -> Tuple[int, str]:
    order = ["FVG", "MAC", "MDMD", "WTS", "k-opt", "SDP-step", "OA-PMC"]
    try:
        return (order.index(name), name)
    except ValueError:
        return (len(order), name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        type=str,
        default="logs/baseline_comparison/eval_big_type123_raw.csv",
    )
    parser.add_argument(
        "--out_cost_csv",
        type=str,
        default="logs/baseline_comparison/summary_cost.csv",
    )
    parser.add_argument(
        "--out_quality_csv",
        type=str,
        default="logs/baseline_comparison/summary_quality.csv",
    )
    parser.add_argument(
        "--tie_tol",
        type=float,
        default=1e-10,
        help="Tie threshold on absolute lambda2 difference vs FVG.",
    )
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_cost = Path(args.out_cost_csv)
    out_quality = Path(args.out_quality_csv)
    if not in_csv.is_absolute():
        in_csv = Path(__file__).resolve().parents[2] / in_csv
    if not out_cost.is_absolute():
        out_cost = Path(__file__).resolve().parents[2] / out_cost
    if not out_quality.is_absolute():
        out_quality = Path(__file__).resolve().parents[2] / out_quality

    rows_raw = _read_csv(in_csv)
    rows: List[Dict[str, object]] = []
    for r in rows_raw:
        n = _to_int(r["n"])
        m = _to_int(r["m"])
        m_max = _to_int(r["m_max"]) if str(r.get("m_max", "")).strip() else (n * (n - 1)) // 2
        if m <= (n - 1) or m >= m_max:
            continue
        rows.append(
            {
                "n": n,
                "m": m,
                "method_raw": str(r["method"]),
                "method": _method_name(str(r["method"])),
                "repeat_idx": _to_int(r["repeat_idx"]),
                "lambda2": _to_float(r["lambda2"]),
                "runtime_total_s": _to_float(r["runtime_total_s"]),
            }
        )

    # Cost summary:
    # 1) average runtime over repeats per (method,n,m)
    # 2) average those point-averages over m and compute std over m
    rt_by_point: Dict[Tuple[str, int, int], List[float]] = {}
    for r in rows:
        k = (str(r["method"]), int(r["n"]), int(r["m"]))
        rt_by_point.setdefault(k, []).append(float(r["runtime_total_s"]))

    rt_point_mean: Dict[Tuple[str, int], List[float]] = {}
    repeats_by_point: Dict[Tuple[str, int], List[int]] = {}
    for (method, n, _m), vv in rt_by_point.items():
        rt_point_mean.setdefault((method, n), []).append(float(np.mean(np.asarray(vv, dtype=np.float64))))
        repeats_by_point.setdefault((method, n), []).append(len(vv))

    cost_rows: List[Dict[str, object]] = []
    for (method, n), vals in sorted(rt_point_mean.items(), key=lambda x: (_method_order(x[0][0]), x[0][1])):
        arr = np.asarray(vals, dtype=np.float64)
        reps = repeats_by_point[(method, n)]
        cost_rows.append(
            {
                "method": method,
                "n": int(n),
                "runtime_mean_over_density_s": float(np.mean(arr)),
                "runtime_std_over_density_s": float(np.std(arr)),
                "density_points": int(arr.size),
                "avg_repeats_per_point": float(np.mean(np.asarray(reps, dtype=np.float64))),
                "min_repeats_per_point": int(min(reps)),
                "max_repeats_per_point": int(max(reps)),
            }
        )

    # Quality summary:
    # baseline FVG by (n,m,repeat)
    fvg_l2: Dict[Tuple[int, int, int], float] = {}
    for r in rows:
        if str(r["method"]) == "FVG":
            fvg_l2[(int(r["n"]), int(r["m"]), int(r["repeat_idx"]))] = float(r["lambda2"])

    delta_by_point: Dict[Tuple[str, int, int], List[float]] = {}
    delta_pct_by_point: Dict[Tuple[str, int, int], List[float]] = {}
    win_count: Dict[Tuple[str, int], int] = {}
    tie_count: Dict[Tuple[str, int], int] = {}
    total_count: Dict[Tuple[str, int], int] = {}
    reps_by_point_q: Dict[Tuple[str, int], List[int]] = {}

    # FVG self rows for completeness
    fvg_points: Dict[int, set[int]] = {}
    fvg_repeats: Dict[int, set[int]] = {}
    for (n, m, rep), _lam in fvg_l2.items():
        fvg_points.setdefault(n, set()).add(m)
        fvg_repeats.setdefault(n, set()).add(rep)

    for r in rows:
        method = str(r["method"])
        n = int(r["n"])
        m = int(r["m"])
        rep = int(r["repeat_idx"])
        lam = float(r["lambda2"])

        base = fvg_l2.get((n, m, rep), None)
        if base is None:
            continue
        d = float(lam - base)
        d_pct = float(d / max(1e-15, abs(base)) * 100.0)
        k_point = (method, n, m)
        delta_by_point.setdefault(k_point, []).append(d)
        delta_pct_by_point.setdefault(k_point, []).append(d_pct)
        reps_by_point_q.setdefault((method, n), []).append(rep)

        k_method_n = (method, n)
        total_count[k_method_n] = int(total_count.get(k_method_n, 0) + 1)
        if abs(d) <= float(args.tie_tol):
            tie_count[k_method_n] = int(tie_count.get(k_method_n, 0) + 1)
        elif d > float(args.tie_tol):
            win_count[k_method_n] = int(win_count.get(k_method_n, 0) + 1)

    # average over repeats first, then over density points
    q_delta_mean_by_method_n: Dict[Tuple[str, int], List[float]] = {}
    q_delta_pct_mean_by_method_n: Dict[Tuple[str, int], List[float]] = {}
    repeats_count_by_method_n: Dict[Tuple[str, int], List[int]] = {}
    for (method, n, _m), vv in delta_by_point.items():
        q_delta_mean_by_method_n.setdefault((method, n), []).append(float(np.mean(np.asarray(vv, dtype=np.float64))))
        q_delta_pct_mean_by_method_n.setdefault((method, n), []).append(
            float(np.mean(np.asarray(delta_pct_by_point[(method, n, _m)], dtype=np.float64)))
        )
        repeats_count_by_method_n.setdefault((method, n), []).append(len(vv))

    quality_rows: List[Dict[str, object]] = []
    all_method_n = sorted(set(q_delta_mean_by_method_n.keys()), key=lambda x: (_method_order(x[0]), x[1]))
    for method, n in all_method_n:
        d_arr = np.asarray(q_delta_mean_by_method_n[(method, n)], dtype=np.float64)
        p_arr = np.asarray(q_delta_pct_mean_by_method_n[(method, n)], dtype=np.float64)
        total = int(total_count.get((method, n), 0))
        win = int(win_count.get((method, n), 0))
        tie = int(tie_count.get((method, n), 0))
        reps_counts = repeats_count_by_method_n[(method, n)]
        quality_rows.append(
            {
                "method": method,
                "n": int(n),
                "delta_lambda2_mean_over_density": float(np.mean(d_arr)),
                "delta_lambda2_std_over_density": float(np.std(d_arr)),
                "delta_lambda2_pct_mean_over_density": float(np.mean(p_arr)),
                "delta_lambda2_pct_std_over_density": float(np.std(p_arr)),
                "win_pct_vs_fvg_over_all_points": float(100.0 * win / total) if total > 0 else 0.0,
                "tie_pct_vs_fvg_over_all_points": float(100.0 * tie / total) if total > 0 else 0.0,
                "win_count": int(win),
                "tie_count": int(tie),
                "total_points": int(total),
                "density_points": int(d_arr.size),
                "avg_repeats_per_point": float(np.mean(np.asarray(reps_counts, dtype=np.float64))),
            }
        )

    cost_fields = [
        "method",
        "n",
        "runtime_mean_over_density_s",
        "runtime_std_over_density_s",
        "density_points",
        "avg_repeats_per_point",
        "min_repeats_per_point",
        "max_repeats_per_point",
    ]
    quality_fields = [
        "method",
        "n",
        "delta_lambda2_mean_over_density",
        "delta_lambda2_std_over_density",
        "delta_lambda2_pct_mean_over_density",
        "delta_lambda2_pct_std_over_density",
        "win_pct_vs_fvg_over_all_points",
        "tie_pct_vs_fvg_over_all_points",
        "win_count",
        "tie_count",
        "total_points",
        "density_points",
        "avg_repeats_per_point",
    ]
    _write_csv(out_cost, cost_rows, cost_fields)
    _write_csv(out_quality, quality_rows, quality_fields)
    print(f"Wrote: {out_cost}")
    print(f"Wrote: {out_quality}")


if __name__ == "__main__":
    main()
