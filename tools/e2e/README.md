# End-to-end timing harness

`picBunchBench` measures the compute path without the checkpoint writer. The
paper also reports writer-inclusive CLI wall times (Table `tab:h100`: 60.5 ms
and 147.9 ms; Table `tab:cost`: 278.28/908.14/1508.12 s near-shell walls). This
directory reproduces those numbers by running the shipped `picBunch` executable
and timing the whole process, including openPMD output and writer finalize.

Everything is configured per case in `cases.json`. The harness sources an
environment file **by path** (`--env-file`) and records the versions it finds;
no ADIOS2/openPMD/CUDA version is hardcoded, so the same command works after
the software stack is replaced.

## Quick start

```bash
# Tiny end-to-end check on GPU 0 (one repeat, no warm-up).
tools/e2e/run_e2e.sh --tiny \
  --binary "$HOME/picbunch/build/picBunch" \
  --input-dir "$HOME/meas" \
  --out "$HOME/e2e-check" \
  --env-file "$HOME/picbunch-bp5fix/env-adios2-2.12.1.sh" \
  --python "$HOME/opt/venv-picbunch/bin/python"

# Full campaign plan (prints every run and an estimated wall time).
tools/e2e/run_e2e.sh --plan --cases all --backend both \
  --input-dir "$HOME/meas" --out "$HOME/e2e-plan" \
  --env-file "$HOME/picbunch-bp5fix/env-adios2-2.12.1.sh"

# Full campaign, BP5 output, 1 warm-up + 3 timed repeats.
tools/e2e/run_e2e.sh --cases all --repeat 3 --warmup 1 --backend bp5 \
  --binary "$HOME/picbunch/build/picBunch" --input-dir "$HOME/meas" \
  --out "$HOME/e2e-run" \
  --env-file "$HOME/picbunch-bp5fix/env-adios2-2.12.1.sh" \
  --python "$HOME/opt/venv-picbunch/bin/python"
```

`--backend both` runs the BP5 (`.bp`) and HDF5 (`.h5`) writer paths for an A/B
comparison. The output extension selects the openPMD backend; the harness
verifies the backend that was actually used and records it in every row. The
paper's writer-inclusive tables were measured with HDF5 output, so the HDF5 row
is the one directly comparable to 60.5/147.9 ms and to the cost table; the BP5
row measures the replaced software stack.

Run `tools/e2e/gen_inputs.py --out-dir DIR` to regenerate the campaign inputs
(requires numpy and h5py). `--plan` lists missing inputs.

## Reconstructed configurations

`docs/measurements/h100_campaign.md` and `h100_campaign.csv` record the main
table per repeat: N in `{1e4, 1e5, 1e6, 2e6}`, box 20 or 100 cells, one rank,
one velocity bin, `--cell-size 1e-6 --origin 0,0,0 --tolerance 1e-3 --shape tsc`.
Those reconstruct the 60.5 ms (g20, N=2e6) and 147.9 ms (g100, N=2e6) rows.

The cost table records N=2e6, grid 66x68x66, cell 2.5e-7 m, tile 4, order 5,
near-shell radii 1.5/3/4, via the shipped fast backend (`--near-shell fast`).
The campaign labels `B1-g20_N2000000`, `B7-p6-cost-nearshell`, and `B6/P6` do
not appear in the repository; the configurations above are mapped from the
campaign CSV and paper text, not from those labels.

Not reconstructable from the repository, and therefore reconstructed here:

* the cost-case source drawer (the input file `input_N2000000_cost.h5` is not
  archived). `gen_inputs.py` stores plain `uniform(0, 20)` cell indices at
  2.5e-7 m; the openPMD reader maps a stored value `p` to `p * unitSI`, so the
  bunch occupies physical `[0, 20*cell)` and its centroid equals the grid
  centroid `(-5.75e-6, -6.0e-6, -5.75e-6) + 0.5 * (66,68,66) * 2.5e-7` m. No
  `(grid-20)/2` offset is applied: that offset assumed an origin of 0 and
  pinned the bunch to the far grid corner under the real origin. The origin
  `(-5.75e-6, -6.0e-6, -5.75e-6)` m follows from the centred choice; the
  original origin is not recorded. `tools/e2e/test_gen_inputs.py` pins the
  centroid, the `unitSI` round-trip, and the R=1.5 pair count.
* any far-tail, truncation, or `--global-grid` flags the cost campaign used.
* the exact input files for every case; only the generator parameters above are
  known.

## What is measured

For each run the driver records:

* `wall_ms` - monotonic wall time around the complete process, including series
  open, compute, write, writer finalize, and exit.
* the CLI stage timings (`reader`, `deposit`, `kernel`, `fft`, `transpose`,
  `readout`, `writer`, `total`) parsed from stdout.
* `near_shell_ms` - the `near shell wall [rank 0]` line the fast backend prints
  on stderr (empty when the near shell is off).
* `setup_ms = wall_ms - total_ms` - process and I/O overhead outside the CLI
  stage timers (signed; it can be negative when the stage sum includes work the
  wall clock overlaps).
* the openPMD backend reported by `inspect_series.py`, the ADIOS2 engine marker
  (`md.0` + `mmd.0` mean BP5), and whether it matches the requested backend.

The summary uses medians over the timed repeats; warm-up runs are written to
`runs.csv` but never enter the medians.

## CSV schema

`runs.csv` has one row per warm-up or timed run. `summary.csv` has one row per
case and backend, aggregating the timed repeats. Both carry host, versions,
binary SHA-256, requested backend, and the planned repeat count.

| column | meaning |
| --- | --- |
| `case` | case id from `cases.json` |
| `backend` | requested backend: `bp5` or `h5` |
| `io_backend` | backend openPMD reported when reading the output (`ADIOS2`, `HDF5`) |
| `engine` | `BP5` when `md.0` and `mmd.0` are present, else `BP4-or-other-ADIOS2` |
| `backend_match` | `yes`/`no` against the requested backend |
| `repeats` | planned timed repeats for this case |
| `repeat` | timed-repeat index (runs.csv only) |
| `phase` | `warmup` or `repeat` (runs.csv only) |
| `ranks` | MPI rank count |
| `host` | `gethostname()` |
| `binary_sha256` | SHA-256 of the measured executable |
| `source_commit` | git HEAD of `--source-dir`, when given |
| `compiled_backend` | `Compiled backend:` line from `picBunch --help` |
| `cuda_version` | `nvcc --version` release, from the sourced environment |
| `adios2_version` | first line of `bpls --version` |
| `openpmd_version` | `openpmd_api.__version__` |
| `env_file` | environment file sourced before every run |
| `wall_ms` | full-process wall time |
| `reader_ms` ... `writer_ms` | CLI stage timings |
| `total_ms` | CLI stage sum |
| `near_shell_ms` | near-shell wall printed by the fast backend |
| `setup_ms` | `wall_ms - total_ms` |
| `output_bytes` | bytes in the written series |
| `status` | `ok`, `failed`, or `timeout` |
| `notes` | case note plus error/verification detail |

Summary-only columns are `wall_median_ms`, `wall_min_ms`, `wall_max_ms`,
`wall_stddev_ms`, and `<stage>_median_ms` for every stage plus
`near_shell_median_ms` and `setup_median_ms`.

`plan.csv` lists each planned run with `estimate_s` (from the paper's measured
wall times) and whether the input file is present.

## Files

* `run_e2e.sh` - entry point.
* `e2e.py` - driver.
* `cases.json` - reconstructed campaign cases and plan estimates.
* `inspect_series.py` - openPMD backend/engine probe.
* `gen_inputs.py` - campaign input generator (numpy + h5py).
