from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acm.eval import evaluate_density_sweep, write_csv  # noqa: E402
from acm.graph_core import max_edges  # noqa: E402


def _nontrivial_m_points(n: int, target_points: int) -> List[int]:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int(max_edges(n_i))
    all_nontrivial = list(range(m_lo + 1, m_hi))
    if len(all_nontrivial) <= int(target_points):
        return all_nontrivial

    idx = np.linspace(0, len(all_nontrivial) - 1, int(target_points), dtype=np.float64)
    idx_i = np.round(idx).astype(np.int64)
    idx_u = sorted(set(int(i) for i in idx_i.tolist()))
    if len(idx_u) < int(target_points):
        used = set(idx_u)
        for i in range(len(all_nontrivial)):
            if i not in used:
                idx_u.append(i)
                used.add(i)
            if len(idx_u) >= int(target_points):
                break
        idx_u = sorted(idx_u[: int(target_points)])
    return [int(all_nontrivial[i]) for i in idx_u]


def _rho_conn_from_m(n: int, m: int) -> float:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int(max_edges(n_i))
    den = int(m_hi - m_lo)
    if den <= 0:
        return 0.0
    return float((int(m) - m_lo) / float(den))


def _as_int_list(v: object) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str):
        return [int(tok.strip()) for tok in v.split(",") if tok.strip()]
    raise ValueError(f"Cannot parse n_values from: {type(v)}")


def _resolve(path_raw: str) -> Path:
    p = Path(path_raw)
    return p if p.is_absolute() else (ROOT / p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_csv", type=str, default="", help="Optional override for manifest out_csv")
    parser.add_argument("--out_meta", type=str, default="", help="Optional override for manifest out_meta")
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    n_values = _as_int_list(manifest.get("n_values", []))
    if not n_values:
        raise ValueError("manifest.n_values is empty")

    points_per_n = max(1, int(manifest.get("points_per_n", 100)))
    repeats = max(1, int(manifest.get("repeats", 5)))
    progress_every = max(1, int(manifest.get("progress_every", 200)))

    methods_default = [str(x) for x in manifest.get("methods_default", [])]
    methods_by_n_raw = manifest.get("methods_by_n", {})
    methods_by_n: Dict[str, List[str]] = {
        str(k): [str(x) for x in v] for k, v in dict(methods_by_n_raw).items()
    }

    common_params = dict(manifest.get("common_params", {}))

    out_csv_raw = str(args.out_csv).strip() or str(manifest.get("out_csv", "logs/baseline_comparison/eval.csv"))
    out_meta_raw = str(args.out_meta).strip() or str(manifest.get("out_meta", "logs/baseline_comparison/eval.meta.json"))
    out_csv = _resolve(out_csv_raw)
    out_meta = _resolve(out_meta_raw)

    plan: Dict[int, Dict[str, object]] = {}
    total_jobs = 0
    for n in n_values:
        m_points = _nontrivial_m_points(n, points_per_n)
        methods = methods_by_n.get(str(n), methods_default)
        if not methods:
            raise ValueError(f"No methods specified for n={n}. Set methods_default or methods_by_n['{n}'].")
        jobs_n = len(m_points) * len(methods) * repeats
        total_jobs += jobs_n
        plan[int(n)] = {
            "n": int(n),
            "m_points": len(m_points),
            "m_min": int(min(m_points)),
            "m_max": int(max(m_points)),
            "methods": methods,
            "method_count": len(methods),
            "repeats": repeats,
            "jobs_n": int(jobs_n),
        }

    preview = {
        "name": manifest.get("name", manifest_path.stem),
        "manifest": str(manifest_path),
        "n_values": n_values,
        "points_per_n": points_per_n,
        "repeats": repeats,
        "total_jobs": total_jobs,
        "out_csv": str(out_csv),
        "out_meta": str(out_meta),
        "plan": plan,
    }
    print("Manifest Experiment Plan")
    print(json.dumps(preview, indent=2))

    if args.dry_run:
        return

    all_rows: List[Dict[str, float | int | str]] = []

    for n in n_values:
        m_points = _nontrivial_m_points(n, points_per_n)
        rho_values = [_rho_conn_from_m(n, m) for m in m_points]
        methods = methods_by_n.get(str(n), methods_default)
        print(
            f"[run] n={n}, methods={methods}, nontrivial_points={len(m_points)}, repeats={repeats}",
            flush=True,
        )
        rows_n = evaluate_density_sweep(
            n_values=[int(n)],
            methods=methods,
            rho_values=rho_values,
            repeats=repeats,
            progress=True,
            progress_every=progress_every,
            **common_params,
        )
        m_set = set(int(m) for m in m_points)
        rows_n = [r for r in rows_n if int(r["m"]) in m_set]
        all_rows.extend(rows_n)
        print(f"[run] done n={n}, rows_kept={len(rows_n)}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, str(out_csv))

    meta = {
        "name": manifest.get("name", manifest_path.stem),
        "manifest": str(manifest_path),
        "n_values": n_values,
        "points_per_n_requested": points_per_n,
        "repeats": repeats,
        "total_rows": len(all_rows),
        "plan": plan,
        "common_params": common_params,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
