# Figure provenance records

One JSON record per measured case documents how a set of paper figures was
produced. The record is the machine-checkable answer to "was this figure
replotted from a fresh measurement?". The schema is
[`provenance.schema.json`](provenance.schema.json); records live in
[`records/`](records/).

## Why this exists

The committed reduced data under `docs/paper/data/` is a legacy copy. Nothing
in the repository proved that a figure came from a re-run rather than from those
old arrays. A record binds a figure to its input data by SHA-256 and records the
environment and command of the measurement. A strict gate then refuses to ship
a figure unless a fresh H100 run claims it.

## Record fields

Every record has exactly these fields (unknown values are `null`, never
invented):

| field | type | meaning |
|---|---|---|
| `case_id` | string | key in `docs/paper/data/cases.json`; unique across records |
| `figure_ids` | string[] | figure basenames in `docs/paper/figs` (no `.pdf`) |
| `measurement_kind` | enum | `fresh-h100`, `fresh-reference`, `legacy-reduced`, or `not-reproduced` |
| `reference_only` | boolean | true when a `fresh-reference` figure plots a recomputed analytic/exact reference rather than a measured PIC field |
| `generator` | object/null | name -> digest map for the code that generated a fresh reference (driver binary/source, kernel header, reducer); artifacts outside the checkout carry a bare SHA-256 |
| `command` | string/null | exact command that produced the data |
| `config_sha256` | digest/null | input config file and its SHA-256 |
| `binary_sha256` | digest/null | executable/artifact and its SHA-256 |
| `data_sha256` | digest[] | every reduced data file the figures consume |
| `script_sha256` | digest/null | measurement or replot script and its SHA-256 |
| `host` | string/null | host name / allocation |
| `gpu_model` | string/null | GPU model |
| `cuda_version` | string/null | CUDA toolkit version |
| `openpmd_version` | string/null | openPMD API version |
| `adios2_version` | string/null | ADIOS2 version |
| `io_backend` | string/null | I/O backend (for example `adios2`, `hdf5`) |
| `seed` | integer/integer[]/null | RNG seed(s) |
| `start_utc` | string/null | `YYYY-MM-DDTHH:MM:SSZ` |
| `end_utc` | string/null | `YYYY-MM-DDTHH:MM:SSZ` |
| `git_commit` | string/null | 7-40 hex digits of the measured code |
| `axis_rule` | string/null | propagation axis convention of the case |
| `repeats` | integer/null | number of repeats |
| `statistic` | string/null | statistic used, for example `median` |
| `notes` | string | free-form; must record what is unknown and why |

`fresh-reference` documents a data figure whose plotted quantity is a freshly
recomputed analytic/exact reference, not a GPU measurement. The four
`val_fluct_*` panels are the order-12 reference computed on a CPU. Such a record
must set `reference_only: true`, name the CPU host, and list the generator
digests; the strict gate accepts it only for these reference-only figures. A
figure that plots a measured PIC field still needs a `fresh-h100` record.

`data_sha256` has at least one entry for every `measurement_kind` except
`not-reproduced`. A not-reproduced record may be empty: it documents a figure
whose renderer exists while the campaign data is still pending. The checker
enforces that rule, because JSON Schema alone cannot make `minItems` depend on
another field. A digest is either
`{"path": <repo-relative path>, "sha256": <64 hex>}` for data files, or
`{"path": <repo-relative path>, "sha256": <64 hex>}` / `null` for the
config, binary and script fields. `path` is optional for those three, so a
binary that was not committed can still be identified by hash alone; the
checker only verifies a digest that carries a path.

## What the checker verifies

`docs/paper/scripts/check_provenance.py`:

1. validates every `records/*.json` against the schema;
2. re-hashes every referenced file and fails on any mismatch or missing file;
3. fails if a figure in `docs/paper/figs` is neither claimed by a record nor
   declared in [`not_reproduced.json`](not_reproduced.json) with a non-empty
   reason;
4. rejects unknown figure ids, duplicate `case_id`s, and a figure that is both
   claimed and declared not-reproduced;
5. rejects a `case_id` that is not a registered case in
   [`docs/paper/data/cases.json`](../data/cases.json);
6. rejects a figure id claimed by more than one record. Exactly one record owns
   a figure id; a single record may still list several figure ids for one case.

`--strict` is the final gate. It additionally requires every *data* figure to
be claimed by a record with `measurement_kind == "fresh-h100"` or by a
`measurement_kind == "fresh-reference"` record with `reference_only: true`.
Figures listed in [`non_data_figures.json`](non_data_figures.json) (the
schematic) are exempt. A `fresh-reference` record can never satisfy strict mode
for a measured PIC field, because the reference-only flag is checked explicitly.
A not-reproduced figure is a data figure and therefore fails strict mode until a
fresh measurement replaces it.

Run it from the repository root:

```bash
python3 docs/paper/scripts/check_provenance.py            # current state
python3 docs/paper/scripts/check_provenance.py --strict   # final replot gate
```

Both exit non-zero on any problem. `--json` prints the summary object.

## Making a record

Do not hand-edit JSON. Use:

```bash
python3 docs/paper/scripts/make_provenance.py \
  --case-id <case> --measurement-kind fresh-h100 \
  --command "<exact command>" \
  --data <path> [--data <path> ...] \
  --script <path> --config <path> --binary <path> \
  --host <host> --gpu-model <gpu> --cuda-version <ver> \
  --openpmd-version <ver> --adios2-version <ver> --io-backend <backend> \
  --seed <seed> --start-utc <UTC> --end-utc <UTC> \
  --git-commit auto --axis-rule <rule> \
  --repeats <n> --statistic <stat> --notes "<what is still unknown>" \
  --out docs/paper/provenance/records/<case>.json
```

The script hashes every file it is given, fills the case defaults (figures,
axis rule, seed, host) from `cases.json`, validates the result against the
schema and only then writes the record. `--stdout` prints instead of writing.
With `--measurement-kind not-reproduced` the `--data` arguments are optional;
such a record marks a figure whose renderer exists but whose measurement has not
landed yet. The normal checker accepts it and the strict gate rejects it.

For a recomputed analytic reference use `--measurement-kind fresh-reference
--reference-only --host <cpu-host>` and pass each generating artifact with
`--generator name=<path-or-sha256>` (repeatable). The script stores a bare
digest for an artifact outside the checkout so the record still validates in a
fresh clone.

## Seeded legacy records

`records/` currently holds 4 `legacy-reduced` records derived from
`docs/paper/data/cases.json` (`line-mpi`, `line-w4b`, `line-gamma-error` and
`line-multivel`), alongside the `fresh-h100` and `fresh-reference` records for
the replotted cases. The legacy records bind each case to the committed reduced
data by SHA-256 and keep the registry's host, seed, axis rule and picBunch
version.
The legacy runs did not record the command, config hash, binary hash, CUDA /
openPMD / ADIOS2 versions or timestamps, so those fields are `null` and the
`notes` field says so. These records pass the normal checker and fail
`--strict`, which is the intended direction of travel: replacing a legacy
record with a `fresh-h100` record can only improve the gate.

`h100-mpi-scaling` is the one pending record: the H100 renderer and the
composer exist, but the per-rank campaign CSVs have not landed, so the record
uses `measurement_kind: not-reproduced`, lists no data files and points
`script_sha256` at `docs/paper/scripts/compose_mpi_scaling.py`. The normal run
passes; `--strict` fails until a fresh measurement replaces it.
[`not_reproduced.json`](not_reproduced.json) stays available for figures that
have no record at all.

## Self-test

`provenance/selftest/run_selftest.py` feeds three deliberately broken inputs to
the real checker in a temporary directory: a tampered data SHA-256, a record
missing a required field, and an empty records directory. Each must make the
checker exit non-zero. It prints one PASS/FAIL line per case and exits non-zero
if the checker ever accepts a broken input.

```bash
python3 docs/paper/provenance/selftest/run_selftest.py
# self-test tampered-data-sha256: PASS
# self-test schema-violation: PASS
# self-test unclaimed-figures: PASS
```
