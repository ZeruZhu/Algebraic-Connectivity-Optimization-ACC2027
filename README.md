# Algebraic Connectivity Optimization — ACC2027

Code, checkpoint, and measurement data for **BB-RL**, accompanying *Graph Theory
Guided Reinforcement Learning for Network Design with High Algebraic Connectivity
under Vertex and Edge Constraints*, by Zeru Zhu, Marco Gamarra, and Ji Liu.
The supplied checkpoint and reference measurements are those used for the
updated Tables I and II in the ACC2027 submission.

BB-RL constructs complete multipartite and Cayley backbones according to the
target density. It completes every routed family in an overlap window with the
same greedy MLP policy, then selects the graph with the largest final algebraic
connectivity. Intact Cayley graphs use group representations for spectral
scoring; graphs after vertex deletion use a dense selected-eigenvalue solver.
Sampling and completion are serial in the reported measurements.

## Install

Use Python 3.11 or newer from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
```

BB-RL and MAC need NumPy, SciPy, NetworkX, PyTorch, PyYAML, and threadpoolctl.
Plot regeneration additionally needs Matplotlib. Commercial solvers and
PyTorch Geometric are not required for BB-RL or MAC.

For the numerical package versions recorded on SeaWulf, use Python 3.11 or
3.12 and install these before the editable package:

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-paper.txt
python -m pip install -c requirements-paper.txt -e ".[test,analysis]"
```

The constraints flag keeps plotting dependencies from upgrading the pinned
numerical stack. The original run used Linux and Python 3.12.3. Local timings
depend on hardware and numerical libraries. Exact graph hashes can also differ when floating-point
ties are resolved differently across platforms.

## Rebuild the paper tables and figures

No training or graph experiments are needed to summarize the released data:

```bash
python scripts/summarize_results.py
python scripts/plot_results.py
```

The first command prints Tables I and II and writes `results/paper/quality.csv`
and `results/paper/cost.csv`. The second writes two baseline quality plots and
the BB-RL/FVG comparison to `results/figures/`. All use the same nontrivial budget
grid and repeat averages as the tables.

Quality differences are paired against FVG by `(n, m, repeat_idx)`. Means and
population standard deviations are across density-point means after averaging
five repeats. Win/match percentages use individual repeat cases, with match
tolerance `1e-10`. The standard deviation is variation across edge budgets,
not a confidence interval. Historical percentages are rounded to integers in
the displayed paper table; the summary CSVs retain unrounded values.

## Evaluate the supplied checkpoint

Run the tests and a small correctness check:

```bash
python -m pytest -q
python benchmark_bbrl.py --out-dir outputs/smoke --n-values 8 --points 3 --seeds 0
```

Run the complete paper grid:

```bash
python benchmark_bbrl.py --out-dir outputs/paper
```

The default grid has 20 nontrivial budgets at `n=8`, 100 each at
`n=16,32,64,128`, and five repeats: **2,100 cases per method**. The evaluator runs
BB-RL and MAC, writes `raw.csv`, `branches.csv`, and `metadata.json`, and verifies
budgets, connectivity, and spectral scores outside the timed intervals.
Output directories must be new to avoid mixing runs. Add `--strict-environment`
to require the paper's numerical package versions. Use `--config` and
`--checkpoint` to evaluate a separately trained policy.

For SeaWulf, activate the pinned environment, install the repository, and run
`sbatch run_paper.sbatch` from its root. The script uses one CPU/numerical thread,
an exclusive node, `short-40core`, and 8 GB RAM, matching the recorded allocation.
`PYTHON_BIN`, `N_VALUES`, `POINTS`, and `SEEDS` can be set through the job environment.
No user-specific filesystem paths are required.

## Runtime protocol and reference results

The fresh BB-RL and MAC measurements come from SeaWulf job **2202156** on `dn020`,
recorded September 24, 2026 (UTC). Both methods ran in the same job with one
numerical thread, three representative warmups per size, and alternating order.
The policy remains resident and structural lookup caches are warm; completed
graph outputs are not cached.

Timed work includes backbone construction, environment reset and features, all
completion branches, terminal scoring, graph copies, and final selection. It
excludes imports, model loading, training, file I/O, and independent verification.
The BB-RL completion column includes selection. MAC retains 50 Frank–Wolfe
iterations, `1e-6` gap tolerance, and Fiedler top-k initialization.

**Only BB-RL and MAC have fresh matched timings.** FVG, MDMD, WTS, k-opt,
SDP-step, and OA-PMC retain the historical measurements used in the paper.
The summaries mark these rows as historical; they are not controlled timing
comparisons against the fresh run.

The recorded data give BB-RL the largest mean improvement over FVG at
`n=16,32,64,128`, and lower matched runtime than MAC at `n=32,64,128`.
They do not establish superiority at every size or on every quality metric.

## Policy and training

`checkpoints/bbrl.pt` is the original checkpoint selected after **30,000 training
transitions**, all in the first curriculum phase (`12 <= n <= 24`). Its SHA-256:

```text
90ace2cca8cfc449486dd36161f82cb3f5797adf32873e0696c51d8a7721dfa8
```

The policy scores symmetric 14-dimensional pair features using two 64-unit
ReLU hidden layers. It uses three node features, five pair features, and three
global features, without intermediate spectral calculations during inference.
Candidate construction selects up to 32 nodes, supplements to 32 nonedges when
available, and retains at most 96 pairs. Exact feature definitions and ranking
coefficients are in `rl_train/features/`.

`configs/bbrl.yaml` contains the policy, PPO parameters, and original planned
curriculum. Later phases describe the schedule; the supplied checkpoint had not
reached them. Training initialization retains the original dense scoring setting.
The paper evaluator explicitly enables group scoring and the selected-eigenvalue
fallback.

Start a new PPO training run with:

```bash
python -m rl_train.train_bbrl --out-dir outputs/train --steps 30000
```

This cleaned training driver supports new experiments with the published
architecture and PPO update. Reproducing the reported tables uses the supplied
checkpoint; retraining is not asserted to reproduce it bit for bit. The
`lite_v3` identifier inside that unchanged checkpoint is a serialization
compatibility field for this BB-RL policy.

## Other baselines

The `baseline/` directory contains the paper's constructive, search, relaxation,
and exact methods, with portable manifests and their parameters. See the
[baseline instructions](baseline/experiments/baseline_comparison/README.md).
SDP-step and OA-PMC additionally need CVXPY and the licensed MOSEK/GUROBI solvers
specified by their manifests. These optional dependencies are unnecessary for
summarizing saved results or evaluating BB-RL/MAC.

## Layout and provenance

| Path | Contents |
| --- | --- |
| `src/graph_design/` | Envelope/Cayley construction, routing, spectral kernels |
| `rl_train/` | BB-RL features, MLP, inference, environment, PPO training |
| `configs/bbrl.yaml`, `checkpoints/bbrl.pt` | Policy settings and evaluated checkpoint |
| `benchmark_bbrl.py`, `run_paper.sbatch` | Local and SeaWulf evaluation |
| `baseline/` | Paper baseline implementations and manifests |
| `results/paper/` | Fresh BB-RL/MAC records, combined summaries, run metadata |
| `results/baseline/` | Historical baseline measurements used by the paper |
| `results/figures/` | Curves regenerated from released records |
| `scripts/` | Table and plot reproduction |
| `tests/` | Spectral, inference, checkpoint, benchmark, and result checks |

Result metadata records original artifact hashes and release transformations.
Method labels and paths are standardized; recorded numerical values are retained.
`rl_train/_source_provenance.json` identifies the evaluated sources from which
the clean release was extracted. The checkpoint is byte-for-byte unchanged.
