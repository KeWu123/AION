# AION

Conditional sampling for history-selected ratio queries on a shared forward
scan. The implementation includes serial sampling, committed epochs with
independent or shared cache randomness, full-prefix Rebase, indexed comparators,
and anytime ratio reporting.

## Setup

Use Python 3.11 and install the recorded numerical dependencies:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

The optional PostgreSQL helper in `code/anq/data.py` additionally uses
`requirements-postgres.txt`. The file-backed experiments do not need PostgreSQL.

## Quick start

The synthetic run exercises all six current policies, history-based query
selection, file/cache access, saved observations and deferred truth auditing:

```sh
python code/exp_history_file.py --synthetic-n 8192 --datasets tpch --reps 1 --skip-sweep --out outputs/history_smoke
```

Output directories must be new. A completed run has a `COMPLETE` manifest and
no `INCOMPLETE` marker. This small run is a functional check, separate from the
retained TPC-H/TPC-DS measurements in `results/`.

## Checks

```sh
python code/validate_stopping_metadata.py
python code/validate_position_scan.py --repetitions 5000 --out outputs/checks/positions
python code/validate_coordinated_scan.py --repetitions 5000 --out outputs/checks/epochs
python code/validate_file_store.py
python code/validate_file_workload.py --out outputs/checks/workload.json
python code/validate_ratio_contract.py
python code/validate_session.py
```

The checks cover exact finite stopping laws, position-pool transitions,
coordinated sampling, source/cache accounting, contribution evaluation and ratio
reporting. `VALIDATION.md` records checks executed for this distribution.

## File-backed studies

Place the retained inputs described in [data/README.md](data/README.md) under
`data/`, then run from the repository root:

```sh
python code/exp_file_streaming.py --data-root data --datasets tpch tpcds --reps 20 --seed 202610174000 --cap 131072 --K 8 --out outputs/matched_file
python code/analyze_file_streaming.py --input outputs/matched_file --out outputs/matched_analysis
python code/exp_history_file.py --reps 20 --seed 20261102000 --out outputs/history_file
python code/verify_history_file.py --study outputs/history_file --out outputs/history_checks.json
python code/analyze_history_file.py --study outputs/history_file --out outputs/history_analysis
```

The history study includes 480 sessions selecting new functions from each
method's released observations, plus 480 sessions with matched query definitions
and concurrency windows 1, 2, 4 and 8. Studies run methods serially and rotate
method order. Timings include initialization, selection, sampling, function
evaluation, inference, file/cache work and teardown. Source preparation, JIT and
deferred auditing are recorded separately. FileIO bytes describe API work with
a warm OS cache.

The source ledger counts new records; complete runtime also depends on the
query functions and stopping trajectories. The preparation model is
`C/M + T(h)` for application sessions sharing one prepared snapshot.

## Retained results and figures

`results/file_streaming/` contains the matched-study summaries.
`results/history_file/` contains the history/concurrency summaries and setup
ledger. CSV estimates are retained from the completed studies; rerunning on
another machine produces new timing measurements.

```sh
python figures/gen_file_streaming_comparison.py --out-dir outputs/figures
python figures/gen_history_lifecycle.py --analysis results/history_file --out outputs/figures/history_lifecycle_concurrency
```

The plotting scripts read saved estimates and intervals. Arial reproduces the
recorded figure typography. The repository contains portable source defaults
and a relative input-manifest path; the numerical sampling and inference
algorithms are unchanged from the retained implementation.

## Layout

| Path | Contents |
| --- | --- |
| `code/anq/` | Sampling pools, forward/file stores and ratio inference |
| `code/exp_*.py` | Experiment drivers and common loaders |
| `code/validate_*.py` | Exact and numerical checks |
| `code/*history*.py` | History-selected workload, analysis and replay audit |
| `code/stopping_metadata_coupling.py` | Exact finite stopping-family construction |
| `figures/` | Figure generators |
| `results/` | Retained small result summaries |
| `data/` | Input identity and loading instructions |

The implementation works on a fixed known-size population with a retained
random-order source. Queries freeze their functions at admission. Serial
returns condition on all earlier releases; an epoch commits participants and
stakes before its joint release. Each query uses its own observation clock.
