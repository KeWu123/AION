# Input data

The file-backed experiments use `lo_data.npz` and `tpcds_sf1.duckdb` in this
directory. These large retained inputs are distributed separately from Git.
`population_manifest.json` records their byte hashes, row counts, normalizers,
normalized-array hashes and row order. The loader reads the original files in
read-only mode. A newly generated dataset with the same scale factor can have
a different row order; compare hashes before treating it as the retained input.

TPC-H uses 6,001,215 lineitem-orders rows. TPC-DS uses 2,637,338 complete rows
from `ss`; the query and row order are recorded in the manifest.

The synthetic quick start in the root README requires neither input file.
