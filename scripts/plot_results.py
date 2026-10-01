"""Regenerate quality curves from the same paired raw data as Tables I/II."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from summarize_results import METHOD_ORDER, ROOT, load_results


def plot_quality(rows: list[dict], n: int, methods: tuple[str, ...], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_point = defaultdict(list)
    for row in rows:
        if row["n"] == n and row["method"] in methods:
            by_point[(row["method"], row["m"])].append(row["lambda2"])
    if not by_point:
        raise ValueError(f"No measurements for n={n}")
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    styles = {
        "FVG": ("#222222", "o"), "MAC": ("#1f77b4", "s"),
        "MDMD": ("#9467bd", "^"), "WTS": ("#2ca02c", "v"),
        "k-opt": ("#ff7f0e", "D"), "SDP-step": ("#8c564b", "P"),
        "OA-PMC": ("#17becf", "X"), "BB-RL": ("#d62728", "^"),
    }
    for method in methods:
        budgets = sorted(m for candidate, m in by_point if candidate == method)
        if not budgets:
            continue
        rho = [(m - (n - 1)) / (n * (n - 1) / 2 - (n - 1)) for m in budgets]
        means = [fmean(by_point[(method, m)]) for m in budgets]
        color, marker = styles[method]
        ax.plot(rho, means, label=method, color=color, marker=marker,
                markevery=max(1, len(budgets) // 12), markersize=4, linewidth=1.5)
    ax.set(xlabel=r"Normalized edge budget $\rho$", ylabel=r"Algebraic connectivity $\lambda_2$",
           title=rf"$n={n}$; mean over five repeats", xlim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, help="Default: RESULTS_DIR/figures")
    parser.add_argument("--baseline-n", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--bbrl-n", type=int, nargs="+", default=[64])
    args = parser.parse_args()
    rows = load_results(args.results_dir)
    output_dir = args.output_dir or args.results_dir / "figures"
    for n in args.baseline_n:
        output = output_dir / f"baseline_n{n}.png"
        plot_quality(rows, n, tuple(method for method in METHOD_ORDER if method != "BB-RL"), output)
        print(output)
    for n in args.bbrl_n:
        output = output_dir / f"bbrl_vs_fvg_n{n}.png"
        plot_quality(rows, n, ("FVG", "BB-RL"), output)
        print(output)


if __name__ == "__main__":
    main()
