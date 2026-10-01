# Classical baseline experiments

The manifest runner evaluates FVG, MAC, MDMD, WTS, k-opt, SDP-step, and
OA-PMC over the paper's connected edge-budget grid. Despite its historical
name, `run_manifest_hpc.py` runs locally and does not require Slurm.

## Dependencies

The baseline package requires NumPy and SciPy. From the repository root:

```bash
python -m pip install -r baseline/requirements.txt
```

SDP-step and OA-PMC additionally require CVXPY and an appropriate solver.
The supplied paper manifests select MOSEK for SDP-step and GUROBI for
OA-PMC; both require a working license. Install these optional packages
only when running the corresponding manifests:

```bash
python -m pip install cvxpy mosek gurobipy
```

Other CVXPY solvers can be selected in a copied manifest, but changing the
solver can change results and runtime. FVG, MAC, MDMD, WTS, and k-opt do not
require CVXPY or a licensed solver.

## Run a manifest

Run these commands from the repository root. Relative manifest and output
paths are resolved against `baseline/`, independently of the working
directory. Absolute paths are also accepted.

Preview the complete experiment without running algorithms or loading solvers:

```bash
python baseline/experiments/baseline_comparison/run_manifest_hpc.py --manifest experiments/baseline_comparison/manifests/baseline_comparison.json --dry_run
```

Run FVG alone:

```bash
python baseline/experiments/baseline_comparison/run_manifest_hpc.py --manifest experiments/baseline_comparison/manifests/fvg_n8_128_rep5.json
```

Use `--out_csv logs/baseline_comparison/custom.csv` and
`--out_meta logs/baseline_comparison/custom.meta.json` to override output
locations. The metadata records the grid, methods, repetitions, and
algorithm parameters. Runtime depends on the machine, solver, and numerical
libraries; a fresh run does not reproduce historical timing measurements.

## Included manifests

| Manifest | Methods and graph sizes |
| --- | --- |
| `baseline_comparison.json` | All seven methods at n=8; excludes OA-PMC at n=16,32; excludes OA-PMC and SDP-step at n=64,128 |
| `fvg_n8_128_rep5.json` | FVG, n=8,16,32,64,128 |
| `mac_n8_128_rep5.json` | MAC, n=8,16,32,64,128 |
| `mdmd_n8_128_rep5.json` | MDMD, n=8,16,32,64,128 |
| `wts_n8_128_rep5.json` | WTS, n=8,16,32,64,128 |
| `kopt_n8_128_rep5.json` | k-opt, n=8,16,32,64,128 |
| `sdps_n8_32_rep5.json` | SDP-step, n=8,16,32 |
| `oa_pmc_n8_rep5.json` | OA-PMC, n=8 |

These manifests retain the evaluated numerical settings and five
repetitions. They request 100 nontrivial budgets per size, using all 20
available budgets at n=8. The complete manifest produces 11,700 rows.
FVG and deterministic MDMD reuse a growth trajectory across budgets and
record cumulative completion time, as in the original experiment runner.

## Summarize a run

```bash
python baseline/experiments/baseline_comparison/summarize_big_experiment.py --in_csv logs/baseline_comparison/eval_big_type123_raw.csv --out_cost_csv logs/baseline_comparison/summary_cost.csv --out_quality_csv logs/baseline_comparison/summary_quality.csv
```

This command uses the same path convention as the runner. It groups raw
rows into cost and quality summaries; inspect the raw CSV and metadata when
comparing runs with different methods, budgets, or solvers.

## Python interface

The installable `acm` package lives in `baseline/src/acm`. Its
`baselines` module contains the numerical implementations, including
`mac_complete`, `fiedler_greedy_complete`, and the EIG construction
`local_global_maximizer_complete`. `acm.eval.evaluate_density_sweep`
provides the manifest runner's evaluation interface. The BB-RL policy and
training code are maintained separately in the repository's main pipeline.
