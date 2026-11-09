#!/usr/bin/env python3
"""Reduce the per-rank H100 MPI-scaling logs into one committed table.

The campaign writes one raw CSV per rank under its log directory::

    mpi_g256_r1.csv
    mpi_g256_r2.csv
    mpi_g256_r4.csv

Each raw file has one row per repeat.  Required columns are ``repeat``,
``total_s``, ``fft_s`` and ``writer_s``.  Optional metadata columns are
``host``, ``gpu_model``, ``grid_cells`` and ``particles``; a missing metadata
value is taken from the command line, and an inconsistent value across rows or
ranks is an error.  Lines beginning with ``#`` are
ignored, so a campaign log may carry a comment header.

The output is exactly one reduced table, one row per rank, with deterministic
rank ordering and six significant decimal digits per number::

    case_id,host,gpu_model,grid_cells,particles,ranks,repeats,
    total_median_s,total_spread_s,fft_median_s,fft_spread_s,
    writer_median_s,writer_spread_s

``*_median_s`` is the median over the repeats and ``*_spread_s`` is the full
range ``max - min``.  Six significant digits keep every measured digit that a
three-repeat campaign records while still making the file diff-stable.

Usage::

    python3 docs/paper/scripts/compose_mpi_scaling.py \\
        --raw-dir <campaign-log-dir> \\
        --host jureca-h100 --gpu-model "NVIDIA H100 PCIe 80 GB" \\
        --particles 2000000 --grid 256 \\
        --out docs/paper/data/line/h100_mpi_scaling.csv

The script refuses to guess: a missing rank file, a missing column, a
non-positive value, or conflicting metadata aborts without writing anything.
"""
import argparse
import csv
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases  # noqa: E402

RAW_REQUIRED = ("repeat", "total_s", "fft_s", "writer_s")
SERIES = ("total", "fft", "writer")
OUTPUT_COLUMNS = (
    "case_id", "host", "gpu_model", "grid_cells", "particles", "ranks",
    "repeats", "total_median_s", "total_spread_s", "fft_median_s",
    "fft_spread_s", "writer_median_s", "writer_spread_s",
)
RAW_NAME = re.compile(r"^mpi_g(?P<grid>[0-9]+)_r(?P<rank>[0-9]+)\.csv$")


class ComposeError(Exception):
    pass


def _read_rows(path):
    """Return the data rows of a raw file, skipping comment and blank lines."""
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    missing = [c for c in RAW_REQUIRED if c not in fieldnames]
    if missing:
        raise ComposeError("%s: missing column(s) %s" % (path, ", ".join(missing)))
    rows = list(reader)
    if not rows:
        raise ComposeError("%s: no data rows" % path)
    return rows


def _float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ComposeError("%s: %r is not a number" % (label, value))
    if number < 0.0 or number != number:
        raise ComposeError("%s: %r must be a finite non-negative number"
                           % (label, value))
    return number


def _int(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ComposeError("%s: %r is not an integer" % (label, value))
    if number <= 0:
        raise ComposeError("%s: %r must be positive" % (label, value))
    return number


def _meta(rows, column, fallback, label):
    values = {str(r.get(column, "")).strip() for r in rows}
    values.discard("")
    if len(values) > 1:
        raise ComposeError("%s: inconsistent %s values %s"
                           % (label, column, sorted(values)))
    if values:
        return values.pop()
    if fallback is None or str(fallback) == "":
        raise ComposeError("%s: no %s in the file and none on the command line"
                           % (label, column))
    return str(fallback)


def _fmt(number, digits=6):
    """Six significant digits, deterministic and without fixed decimals."""
    return format(float(number), ".%dg" % digits)


def _find_raw(raw_dir):
    if not os.path.isdir(raw_dir):
        raise ComposeError("raw directory does not exist: %s" % raw_dir)
    found = {}
    for root, _dirs, names in os.walk(raw_dir):
        for name in sorted(names):
            match = RAW_NAME.match(name)
            if not match:
                continue
            path = os.path.join(root, name)
            key = (int(match.group("grid")), int(match.group("rank")))
            if key in found:
                raise ComposeError("duplicate raw file for grid %d rank %d: "
                                   "%s and %s" % (key[0], key[1], found[key], path))
            found[key] = path
    if not found:
        raise ComposeError("no mpi_g<grid>_r<rank>.csv files under %s" % raw_dir)
    return found


def reduce(raw_dir, *, case_id, host=None, gpu_model=None, particles=None,
           grid=None, ranks=(1, 2, 4)):
    found = _find_raw(raw_dir)
    grids = sorted({key[0] for key in found})
    if len(grids) != 1:
        raise ComposeError("raw files mix grid sizes %s; run one grid at a time"
                           % grids)
    grid_from_files = grids[0]
    if grid is not None and int(grid) != grid_from_files:
        raise ComposeError("raw files are grid %d, but --grid is %s"
                           % (grid_from_files, grid))
    grid = grid_from_files
    wanted = sorted({int(r) for r in ranks})
    missing = [r for r in wanted if (grid, r) not in found]
    if missing:
        raise ComposeError("missing raw file(s) for grid %d rank(s) %s"
                           % (grid, ", ".join(str(r) for r in missing)))
    extra = sorted(r for (g, r) in found if g == grid and r not in wanted)
    if extra:
        raise ComposeError("unexpected rank file(s) %s; expected %s"
                           % (", ".join(str(r) for r in extra),
                              ", ".join(str(r) for r in wanted)))

    rows = []
    for rank in wanted:
        path = found[(grid, rank)]
        raw = _read_rows(path)
        repeats = [_int(r["repeat"], "%s repeat" % path) for r in raw]
        if len(set(repeats)) != len(repeats):
            raise ComposeError("%s: duplicate repeat number" % path)
        values = {s: [_float(r[s + "_s"], "%s %s_s" % (path, s)) for r in raw]
                  for s in SERIES}
        record = {
            "case_id": case_id,
            "host": _meta(raw, "host", host, path),
            "gpu_model": _meta(raw, "gpu_model", gpu_model, path),
            "grid_cells": _int(_meta(raw, "grid_cells", grid, path),
                               "%s grid_cells" % path),
            "particles": _int(_meta(raw, "particles", particles, path),
                              "%s particles" % path),
            "ranks": rank,
            "repeats": len(raw),
        }
        for stage in SERIES:
            record[stage + "_median_s"] = statistics.median(values[stage])
            record[stage + "_spread_s"] = max(values[stage]) - min(values[stage])
        rows.append(record)

    for column in ("case_id", "host", "gpu_model", "grid_cells", "particles"):
        distinct = {str(r[column]) for r in rows}
        if len(distinct) != 1:
            raise ComposeError("ranks disagree on %s: %s"
                               % (column, sorted(distinct)))
    return rows


def write_table(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["ranks"])):
            encoded = {}
            for column in OUTPUT_COLUMNS:
                value = row[column]
                if column.endswith(("_median_s", "_spread_s")):
                    encoded[column] = _fmt(value)
                else:
                    encoded[column] = value
            writer.writerow(encoded)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", required=True,
                    help="directory that holds mpi_g<grid>_r<rank>.csv")
    ap.add_argument("--out", default=cases.data("line", "h100_mpi_scaling.csv"),
                    help="reduced table path (default: %(default)s)")
    ap.add_argument("--case-id", default="h100-mpi-scaling")
    ap.add_argument("--host", default=None)
    ap.add_argument("--gpu-model", default=None)
    ap.add_argument("--particles", type=int, default=None)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--ranks", default="1,2,4",
                    help="comma-separated expected rank counts")
    a = ap.parse_args(argv)
    ranks = [int(v) for v in a.ranks.replace(" ", ",").split(",") if v]
    try:
        rows = reduce(a.raw_dir, case_id=a.case_id, host=a.host,
                      gpu_model=a.gpu_model, particles=a.particles,
                      grid=a.grid, ranks=ranks)
        write_table(rows, a.out)
    except ComposeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("wrote %s (%d rank rows)" % (a.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
