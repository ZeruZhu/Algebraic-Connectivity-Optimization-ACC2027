"""Reproduce ACC2027 Tables I/II from the released measurement records.

Only BB-RL and MAC have fresh, matched timings. The other methods retain
their historical measurements; the output explicitly preserves that distinction.
This script uses only the Python standard library and does not run experiments.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("FVG", "MAC", "MDMD", "WTS", "k-opt", "SDP-step", "OA-PMC", "BB-RL")
FRESH_METHODS = {"BB-RL", "MAC"}
HISTORICAL_METHODS = set(METHOD_ORDER) - FRESH_METHODS
TIE_TOLERANCE = 1e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_results(results_dir: Path = ROOT / "results") -> list[dict]:
    """Load public raw records, rejecting stale labels and ambiguous pairing."""
    rows = []
    seen = set()
    for relative, allowed, provenance in (
        ("paper/raw.csv", FRESH_METHODS, "fresh_matched"),
        ("baseline/raw.csv", HISTORICAL_METHODS, "historical"),
    ):
        for source in read_csv(results_dir / relative):
            method = source["method"]
            if method not in allowed:
                raise ValueError(f"Unexpected method {method!r} in {relative}")
            n, m, repeat = (int(source[key]) for key in ("n", "m", "repeat_idx"))
            key = (method, n, m, repeat)
            if key in seen:
                raise ValueError(f"Duplicate measurement key: {key}")
            seen.add(key)
            if not n - 1 < m < n * (n - 1) // 2:
                raise ValueError(f"Expected a nontrivial budget: {key}")
            row = {
                "method": method, "n": n, "m": m, "repeat_idx": repeat,
                "lambda2": float(source["lambda2"]),
                "runtime_total_s": float(source["runtime_total_s"]),
                "runtime_init_s": float(source["runtime_init_s"]),
                "runtime_complete_s": float(source["runtime_complete_s"]),
                "runtime_select_s": float(source.get("runtime_select_s", 0.0)),
                "timing_provenance": provenance,
            }
            for field in ("lambda2", "runtime_total_s", "runtime_init_s", "runtime_complete_s", "runtime_select_s"):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError(f"Invalid {field} at {key}")
            rows.append(row)
    reference = {(r["n"], r["m"], r["repeat_idx"]): r for r in rows if r["method"] == "FVG"}
    if not reference:
        raise ValueError("No FVG reference records")
    for row in rows:
        key = row["n"], row["m"], row["repeat_idx"]
        if key not in reference:
            raise ValueError(f"Missing paired FVG reference for {row['method']}: {key}")
        row["delta_lambda2"] = row["lambda2"] - reference[key]["lambda2"]
    return rows


def summarize(rows: list[dict], tie_tolerance: float = TIE_TOLERANCE) -> tuple[list[dict], list[dict]]:
    """Average repeats first; population std is across density-point means."""
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["n"])].append(row)
    quality, cost = [], []
    for (method, n), group in sorted(groups.items(), key=lambda item: (METHOD_ORDER.index(item[0][0]), item[0][1])):
        by_point = defaultdict(list)
        for row in group:
            by_point[row["m"]].append(row)
        points = list(by_point.values())
        repeats = [len(point) for point in points]

        def point_means(field: str) -> list[float]:
            return [fmean(row[field] for row in point) for point in points]

        values = point_means("lambda2")
        deltas = point_means("delta_lambda2")
        wins = sum(row["delta_lambda2"] > tie_tolerance for row in group)
        matches = sum(abs(row["delta_lambda2"]) <= tie_tolerance for row in group)
        quality.append({
            "method": method, "n": n,
            "lambda2_mean": fmean(values), "lambda2_std": pstdev(values),
            "delta_lambda2_mean": fmean(deltas), "delta_lambda2_std": pstdev(deltas),
            "win_pct_vs_fvg": 100.0 * wins / len(group),
            "match_pct_vs_fvg": 100.0 * matches / len(group),
            "win_count": wins, "match_count": matches,
            "case_count": len(group), "density_points": len(points),
            "min_repeats": min(repeats), "max_repeats": max(repeats),
        })
        runtimes = point_means("runtime_total_s")
        cost.append({
            "method": method, "n": n,
            "runtime_total_mean_s": fmean(runtimes),
            "runtime_total_std_s": pstdev(runtimes),
            "runtime_init_mean_s": fmean(point_means("runtime_init_s")) if method == "BB-RL" else "",
            "runtime_completion_including_selection_mean_s": (
                fmean(fmean(row["runtime_complete_s"] + row["runtime_select_s"] for row in point) for point in points)
                if method == "BB-RL" else ""
            ),
            "timing_provenance": group[0]["timing_provenance"],
            "case_count": len(group), "density_points": len(points),
            "min_repeats": min(repeats), "max_repeats": max(repeats),
        })
    return quality, cost


def print_tables(quality: list[dict], cost: list[dict]) -> None:
    sizes = sorted({row["n"] for row in quality})
    qlookup = {(row["method"], row["n"]): row for row in quality}
    clookup = {(row["method"], row["n"]): row for row in cost}
    header = "| Method | " + " | ".join(f"n={n}" for n in sizes) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in sizes) + " |"
    print("Table I: FVG absolute mean +/- population std; others paired delta mean +/- std / win% / match%.")
    print(header)
    print(separator)
    for method in METHOD_ORDER:
        cells = []
        for n in sizes:
            row = qlookup.get((method, n))
            if row is None:
                cells.append("--")
            elif method == "FVG":
                cells.append(f"{row['lambda2_mean']:.2f} +/- {row['lambda2_std']:.2f}")
            else:
                # Match the paper's integer rounding of historical percentages.
                # The CSV always retains the unrounded fractions.
                rate_format = ".0f" if method in HISTORICAL_METHODS else "g"
                win = format(row["win_pct_vs_fvg"], rate_format)
                match = format(row["match_pct_vs_fvg"], rate_format)
                cells.append(f"{row['delta_lambda2_mean']:.2f} +/- {row['delta_lambda2_std']:.2f} / {win} / {match}")
        print("| " + method + " | " + " | ".join(cells) + " |")
    print("\nTable II: mean online seconds. Historical rows are not matched reruns.")
    print(header)
    print(separator)
    for method in METHOD_ORDER:
        label = method + (" (historical)" if method in HISTORICAL_METHODS else "")
        cells = [f"{clookup[(method, n)]['runtime_total_mean_s']:.6g}" if (method, n) in clookup else "--" for n in sizes]
        print("| " + label + " | " + " | ".join(cells) + " |")
    for label, field in (("BB-RL init", "runtime_init_mean_s"), ("BB-RL completion + selection", "runtime_completion_including_selection_mean_s")):
        cells = [f"{clookup[('BB-RL', n)][field]:.6g}" if ("BB-RL", n) in clookup else "--" for n in sizes]
        print("| " + label + " | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, help="Default: RESULTS_DIR/paper")
    args = parser.parse_args()
    quality, cost = summarize(load_results(args.results_dir))
    output_dir = args.output_dir or args.results_dir / "paper"
    write_csv(output_dir / "quality.csv", quality)
    write_csv(output_dir / "cost.csv", cost)
    print_tables(quality, cost)


if __name__ == "__main__":
    main()
