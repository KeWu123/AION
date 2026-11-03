# Distribution checks

The following checks passed on Python 3.11.7 on Windows, using the numerical
versions in `requirements.txt`. The validation venv reused installed scientific
packages and installed DuckDB 1.4.3 separately.

| Check | Result |
| --- | --- |
| Compile all Python modules | Pass |
| Exact finite stopping-family laws | Pass |
| Position-pool sampling, 5,000 statistical repetitions | Pass |
| Coordinated epochs, 5,000 statistical repetitions | Pass |
| File/cache backend accounting | Pass |
| Scalar contribution and compensated-moment references | Pass |
| Ratio-report and census regressions | Pass |
| Session controller and error-budget regressions | Pass |
| History-selected synthetic file study | 12 complete sessions, 96 queries, 206 reports |
| Matched-definition synthetic file study | 6 complete sessions, 48 queries |
| Source/time comparison figure | 56 retained estimates reproduced |
| Preparation/concurrency figure | 60 retained estimates reproduced |

The two smoke studies use 8,192 generated rows. Their outputs are kept separate
from the retained TPC-H/TPC-DS results. Both completed their manifests and
deferred audits. The plotted point ledgers match the retained figure ledgers.

The portable distribution changes default input/summary paths and initializes
the output directories used by two standalone checks. Numerical sampling and
inference modules retain the same syntax trees as the retained implementation.

The `checks` GitHub Actions workflow runs the numerical checks on Ubuntu with
Python 3.11 when a commit or pull request is published.
