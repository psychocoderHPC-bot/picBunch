#!/usr/bin/env python3
"""Self-contained smoke test for the H100 MPI-scaling pipeline.

Draws a synthetic campaign (never a real measurement) into a temporary
directory, reduces it with ``compose_mpi_scaling.py`` and renders it with
``render_scaling.mpi_h100``.  The tree is left untouched: the synthetic table
and the PDFs live under the temporary directory and are removed on exit.

Run from the repository root::

    python3 docs/paper/scripts/selftest_mpi_h100.py

It prints one PASS/FAIL line per check and exits non-zero on any failure.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compose_mpi_scaling as compose  # noqa: E402
import render_scaling  # noqa: E402


def _raw(rank, rows):
    return (
        "# synthetic campaign file for rank %d\n"
        "repeat,total_s,fft_s,writer_s,host,gpu_model,grid_cells,particles\n"
        % rank
        + "".join("%d,%s,%s,%s,jureca-h100,NVIDIA H100 PCIe 80 GB,256,"
                  "2000000\n" % row for row in rows)
    )


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="mpi-h100-selftest-") as tmp:
        raw_dir = os.path.join(tmp, "raw")
        os.makedirs(raw_dir)
        # Synthetic numbers, one row per repeat.  The spread is deliberate.
        data = {
            1: [(1, 100.0, 10.0, 20.0), (2, 110.0, 11.0, 22.0),
                (3, 105.0, 10.5, 21.0)],
            2: [(1, 60.0, 6.0, 25.0), (2, 62.0, 6.2, 26.0),
                (3, 61.0, 6.1, 24.0)],
            4: [(1, 40.0, 4.0, 30.0), (2, 42.0, 4.2, 31.0),
                (3, 41.0, 4.1, 29.0)],
        }
        for rank, rows in data.items():
            with open(os.path.join(raw_dir, "mpi_g256_r%d.csv" % rank), "w") as fh:
                fh.write(_raw(rank, rows))

        table = os.path.join(tmp, "reduced", "h100_mpi_scaling.csv")
        try:
            rows = compose.reduce(
                raw_dir, case_id="h100-mpi-scaling", host="jureca-h100",
                gpu_model="NVIDIA H100 PCIe 80 GB", particles=2000000,
                grid=256, ranks=(1, 2, 4))
            compose.write_table(rows, table)
            ok = os.path.isfile(table)
            print("selftest compose-writes-table: %s" % ("PASS" if ok else "FAIL"))
            if not ok:
                failures.append("composer did not write the reduced table")
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print("selftest compose-writes-table: FAIL (%s)" % exc)
            failures.append("composer failed: %s" % exc)

        if os.path.isfile(table):
            with open(table) as fh:
                header = fh.readline().strip().split(",")
            want = ["case_id", "host", "gpu_model", "grid_cells", "particles",
                    "ranks", "repeats", "total_median_s", "total_spread_s",
                    "fft_median_s", "fft_spread_s", "writer_median_s",
                    "writer_spread_s"]
            ok = header == want
            print("selftest reduced-schema: %s" % ("PASS" if ok else "FAIL"))
            if not ok:
                failures.append("reduced table header does not match the schema")
            try:
                parsed = render_scaling.read_h100_table(table)
                ok = [int(r["ranks"]) for r in parsed] == [1, 2, 4]
            except Exception as exc:  # noqa: BLE001
                ok = False
                print("  reader error: %s" % exc)
            print("selftest deterministic-rank-order: %s"
                  % ("PASS" if ok else "FAIL"))
            if not ok:
                failures.append("reduced table is not ordered by rank")

        outdir = os.path.join(tmp, "out")
        try:
            render_scaling.mpi_h100(outdir, data_path=table)
            pdf = os.path.join(outdir, "F_mpi_rank_scaling_h100.pdf")
            with open(pdf, "rb") as fh:
                magic = fh.read(5)
            ok = magic == b"%PDF-" and os.path.getsize(pdf) > 1000
            print("selftest renders-valid-pdf: %s" % ("PASS" if ok else "FAIL"))
            if not ok:
                failures.append("renderer did not produce a valid PDF")
        except Exception as exc:  # noqa: BLE001
            print("selftest renders-valid-pdf: FAIL (%s)" % exc)
            failures.append("renderer failed: %s" % exc)

        missing = os.path.join(tmp, "does-not-exist", "h100_mpi_scaling.csv")
        try:
            render_scaling.mpi_h100(os.path.join(tmp, "out2"), data_path=missing)
            ok = False
        except render_scaling.PendingData:
            ok = True
        print("selftest pending-data-is-explicit: %s"
              % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("missing reduced data did not raise PendingData")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
