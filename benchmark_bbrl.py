"""Evaluate the ACC2027 BB-RL policy and MAC with the paper timing protocol."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import networkx as nx
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from acm.baselines import mac_complete
from acm.graph_core import path_graph
from graph_design.backbones import cayley
from graph_design.backbones.envelope import EnvelopeBackboneGenerator
from graph_design.config import DesignConfig
from graph_design.routing import DensityRoutingPolicy
from graph_design.types import DesignProblem
from rl_train.graph_math import algebraic_connectivity, edge_count, laplacian
from rl_train.inference import PolicyRunner

ROOT = Path(__file__).resolve().parent
MAC_PARAMETERS = dict(mac_fw_iters=50, mac_duality_gap_tol=1e-6,
                      mac_init_mode="fiedler_topk", mac_seed=0)
PAPER_VERSIONS = dict(numpy="1.26.4", scipy="1.14.0", torch="2.3.0", networkx="3.3")


def targets(n, points):
    """Same nontrivial integer-budget grid as the recorded paper run."""
    if n < 3 or points < 1:
        raise ValueError("n must be at least 3 and points must be positive")
    budgets = np.arange(n, n * (n - 1) // 2)
    indices = np.unique(np.rint(np.linspace(0, len(budgets) - 1,
                                           min(points, len(budgets)))).astype(int))
    return budgets[indices].tolist()


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def adjacency_hash(adj):
    return hashlib.sha256(np.asarray(adj, dtype=np.uint8).tobytes()).hexdigest()


def exact_lambda2(adj):
    return float(np.linalg.eigvalsh(laplacian(adj))[1])


def validate_graph(adj, n, m, initial=None):
    if adj.shape != (n, n) or not np.array_equal(adj, adj.T) or np.any(np.diag(adj)):
        raise AssertionError("Invalid simple undirected graph")
    if not np.all((adj == 0) | (adj == 1)) or edge_count(adj) != m:
        raise AssertionError("Graph edge budget mismatch")
    if initial is not None and np.any(np.asarray(adj) < np.asarray(initial)):
        raise AssertionError("Completion removed a backbone edge")


def build_bundle(module, config, n, m, seed):
    """Preserve the recorded construction order, seeds, overlap and fallback."""
    problem = DesignProblem(n=n, m=m, seed=seed)
    route = DensityRoutingPolicy(config).route(problem.density)
    generators = {"cayley": module.CayleyBackboneGenerator(),
                  "envelope": EnvelopeBackboneGenerator()}
    branches = []
    started = time.perf_counter()

    def build(family):
        start = time.perf_counter()
        candidate = generators[family].generate(
            problem, config, random.Random(seed * 10007 + (1 if family == "cayley" else 2)))
        if candidate is not None:
            adj = nx.to_numpy_array(candidate.graph, nodelist=range(n), dtype=np.uint8)
            branches.append((family, adj, candidate.backbone_lambda2,
                             candidate.metadata, time.perf_counter() - start))

    for family in route.families:
        build(family)
    if not branches:
        for family in generators:
            if family not in route.families:
                build(family)
    elapsed = time.perf_counter() - started
    if not branches:
        raise AssertionError(f"No backbone at {n=}, {m=}, {seed=}")
    return elapsed, branches


def run_bbrl(runner, config, n, m, seed):
    start = time.perf_counter()
    route = DensityRoutingPolicy(config).route(DesignProblem(n=n, m=m, seed=seed).density).label
    _, branches = build_bundle(cayley, config, n, m, seed)
    init_end = time.perf_counter()
    completed = []
    for family, initial, initial_score, metadata, build_s in branches:
        branch_start = time.perf_counter()
        final, score = runner.complete(initial, m, {"family": family, "route": route,
                                                   "lambda2": initial_score})
        final = final.copy()
        completed.append(dict(family=family, initial=initial, final=final, lambda2=score,
                              initial_lambda2=initial_score, init_edges=edge_count(initial),
                              build_s=build_s, complete_s=time.perf_counter() - branch_start,
                              backend=metadata.get("eval_backend", "envelope_formula")))
    completion_end = time.perf_counter()
    winner = max(completed, key=lambda b: (b["lambda2"], b["initial_lambda2"], b["init_edges"]))
    end = time.perf_counter()
    row = dict(n=n, m=m, repeat_idx=seed, method="BB-RL", lambda2=winner["lambda2"],
               runtime_init_s=init_end-start, runtime_complete_s=completion_end-init_end,
               runtime_select_s=end-completion_end, runtime_total_s=end-start,
               selected_family=winner["family"], branches=len(completed))
    return row, completed


def run_mac(params, n, m, seed):
    start = time.perf_counter()
    initial = path_graph(n)
    init_end = time.perf_counter()
    final = mac_complete(initial, m, fw_iters=params["mac_fw_iters"],
                         duality_gap_tol=params["mac_duality_gap_tol"],
                         init_mode=params["mac_init_mode"],
                         seed=params["mac_seed"] + 53633*n + m + 9973*seed)
    score = algebraic_connectivity(final)
    end = time.perf_counter()
    row = dict(n=n, m=m, repeat_idx=seed, method="MAC", lambda2=score,
               runtime_init_s=init_end-start, runtime_complete_s=end-init_end,
               runtime_select_s=0.0, runtime_total_s=end-start,
               selected_family="path", branches=1)
    return row, [dict(family="path", initial=initial, final=final, lambda2=score)]


def verify(row, branches):
    errors = []
    for branch in branches:
        initial, final = branch["initial"], branch["final"]
        validate_graph(initial, row["n"], edge_count(initial))
        validate_graph(final, row["n"], row["m"], initial=initial)
        verified_initial = exact_lambda2(initial)
        verified_final = exact_lambda2(final)
        if verified_initial <= 1e-9 or verified_final <= 1e-9:
            raise AssertionError("Disconnected graph")
        error = abs(branch["lambda2"] - verified_final)
        if error > 1e-8:
            raise AssertionError(f"Incorrect terminal score: {error}")
        if "initial_lambda2" in branch and abs(branch["initial_lambda2"] - verified_initial) > 1e-8:
            raise AssertionError("Incorrect backbone score")
        errors.append(error)
        branch["score_error"] = error
    if abs(row["runtime_total_s"] - sum(row[k] for k in (
            "runtime_init_s", "runtime_complete_s", "runtime_select_s"))) > 1e-10:
        raise AssertionError("Timer decomposition does not sum to total")
    winner = max(branches, key=lambda b: (b["lambda2"], b.get("initial_lambda2", 0), b.get("init_edges", 0)))
    row["graph_sha256"] = adjacency_hash(winner["final"])
    row["max_score_error"] = max(errors)


def source_hashes():
    paths = [Path(__file__), ROOT / "pyproject.toml"]
    for folder in ("src/graph_design", "rl_train", "baseline/src/acm"):
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    ns = [int(n) for n in args.n_values.split(",")]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(set(ns)) != len(ns) or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Sizes and seeds must be nonempty and unique")
    grids = {n: targets(n, args.points) for n in ns}
    if any(not grid for grid in grids.values()):
        raise ValueError("Every size must have a nontrivial edge budget (use n >= 4)")
    versions = dict(numpy=np.__version__, scipy=scipy.__version__,
                    torch=torch.__version__.split("+")[0], networkx=nx.__version__)
    if args.strict_environment and versions != PAPER_VERSIONS:
        raise RuntimeError(f"Environment {versions} differs from paper versions {PAPER_VERSIONS}")
    runner = PolicyRunner(config_path=args.config, checkpoint_path=args.checkpoint)
    torch.set_num_threads(1)
    config = DesignConfig(cayley_eval_mode="character", cayley_dense_solver="subset", cayley_samples=500,
                          cayley_index2_group="current", cayley_multi_index_overlap=True,
                          cayley_independent_index_seeds=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    metadata = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    host=platform.node(), python=sys.version, platform=platform.platform(),
                    versions=versions, paper_versions=PAPER_VERSIONS,
                    matches_paper_versions=versions == PAPER_VERSIONS,
                    slurm_job_id=os.getenv("SLURM_JOB_ID"),
                    slurm_cpus=os.getenv("SLURM_CPUS_PER_TASK"), threads=torch.get_num_threads(),
                    numerical_threadpools=threadpool_info(), config=asdict(config),
                    policy=runner.provenance, source_sha256=hashes,
                    n_values=ns, points=args.points, seeds=seeds,
                    expected_cases=sum(map(len, grids.values())) * len(seeds),
                    mac_parameters=MAC_PARAMETERS,
                    protocol="Resident policy; three representative warmups per size; warmed structural caches; alternating method order; serial construction and completion of all routed branches. Includes graph construction, reset, terminal scoring, copies and selection. Excludes imports, model loading, training, I/O and independent verification.",
                    score_backend="SciPy selected scalar lambda2 for both methods; MAC optimizer unchanged.")
    meta_path = args.out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows, branch_rows = [], []
    try:
        for n in ns:
            grid = grids[n]
            for index in sorted({0, len(grid)//2, len(grid)-1}):
                for operation in (lambda m: run_bbrl(runner, config, n, m, seeds[0]),
                                  lambda m: run_mac(MAC_PARAMETERS, n, m, seeds[0])):
                    row, branches = operation(grid[index])
                    verify(row, branches)
            for index, m in enumerate(grid):
                for seed in seeds:
                    order = ("BB-RL", "MAC") if (m + seed) % 2 else ("MAC", "BB-RL")
                    for method in order:
                        row, branches = (run_bbrl(runner, config, n, m, seed) if method == "BB-RL"
                                         else run_mac(MAC_PARAMETERS, n, m, seed))
                        verify(row, branches)
                        rows.append(row)
                        if method == "BB-RL":
                            for branch in branches:
                                branch_rows.append(dict(n=n, m=m, repeat_idx=seed,
                                    family=branch["family"], init_edges=branch["init_edges"],
                                    init_lambda2=branch["initial_lambda2"], lambda2=branch["lambda2"],
                                    build_s=branch["build_s"], complete_s=branch["complete_s"],
                                    backend=branch["backend"], score_error=branch["score_error"],
                                    initial_sha256=adjacency_hash(branch["initial"]),
                                    final_sha256=adjacency_hash(branch["final"])))
                if (index + 1) % 10 == 0 or index + 1 == len(grid):
                    write_csv(args.out_dir / "raw.csv", rows)
                    write_csv(args.out_dir / "branches.csv", branch_rows)
                    print(f"n={n} density_points={index+1}/{len(grid)} completed_cases={len(rows)//2}", flush=True)
        if hashes != source_hashes():
            raise AssertionError("Benchmark sources changed during execution")
        for kind in ("config", "checkpoint"):
            path = Path(runner.provenance[kind])
            if hashlib.sha256(path.read_bytes()).hexdigest() != runner.provenance[f"{kind}_sha256"]:
                raise AssertionError(f"Policy {kind} changed during execution")
        metadata.update(status="completed", validated_branch_count=len(branch_rows),
                        validated_result_count=len(rows),
                        max_score_error=max(r["max_score_error"] for r in rows))
    except Exception as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        write_csv(args.out_dir / "raw.csv", rows)
        write_csv(args.out_dir / "branches.csv", branch_rows)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("status", "validated_result_count", "max_score_error")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--n-values", default="8,16,32,64,128")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bbrl.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/bbrl.pt")
    parser.add_argument("--strict-environment", action="store_true",
                        help="Require the NumPy/SciPy/Torch/NetworkX versions of the paper run")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
