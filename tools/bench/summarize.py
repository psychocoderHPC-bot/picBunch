#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Summarize per-repeat benchmark rows with the pre-registered timing protocol.

Input is the wide per-repeat CSV written by ``picBunchBench --raw-csv``: one
row per warm-up or timed repeat, with the stage columns in milliseconds and an
optional ``clock_mhz`` column (filled by the external 2 s nvidia-smi sampler).
``summarize.py`` applies the protocol documented in
``docs/.../cost-and-noise.md`` (P2) and ``tools/bench/README.md``:

1. Report median, IQR (P25-P75), min and max per stage and for the total.
2. Bimodal check first: if a stage's P75/P25 > 2.5, split that stage at the
   largest adjacent gap of its sorted samples, report both cluster medians and
   counts, and report the total as two mode-totals (sum of the stage cluster
   medians). No sample is dropped in this branch.
3. Otherwise drop a repeat if its total is outside [0.7, 1.4] x the row median,
   or if its median SM clock is >10% below the row's median clock. Require at
   least 6 survivors; otherwise flag the row for a fresh 9-run set.
4. The acceptance gate is IQR/median <= 2 x the measured B0 IQR floor (pass
   ``--b0-iqr`` as a fraction, e.g. 0.015 for 1.5%), replacing the old 2.586%
   gate.

Run ``python3 tools/bench/summarize.py --self-test`` for the rule self-test.
"""
import argparse
import csv
import sys
from pathlib import Path

TOTAL = "total_ms"
BIMODAL_RATIO = 2.5
OUTLIER_LO = 0.7
OUTLIER_HI = 1.4
CLOCK_MARGIN = 0.9
MIN_SURVIVORS = 6


def percentile(sorted_values, q):
    """Linear-interpolation percentile (numpy's default method)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(sorted_values[upper]) * fraction


def stats(values):
    values = sorted(float(v) for v in values)
    p25 = percentile(values, 0.25)
    p50 = percentile(values, 0.50)
    p75 = percentile(values, 0.75)
    return {
        "n": len(values),
        "median": p50,
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "min": values[0] if values else 0.0,
        "max": values[-1] if values else 0.0,
    }


def split_at_gap(values):
    """Split sorted values at the largest adjacent gap; return (low, high)."""
    values = sorted(float(v) for v in values)
    if len(values) < 2:
        return values, []
    best_index = -1
    best_gap = -1.0
    for i in range(len(values) - 1):
        gap = values[i + 1] - values[i]
        if gap > best_gap:
            best_gap = gap
            best_index = i
    if best_index < 0 or best_gap <= 0.0:
        return values, []
    return values[: best_index + 1], values[best_index + 1 :]


def stage_columns(records):
    columns = set()
    for record in records:
        for key in record:
            if key.endswith("_ms") and key != TOTAL:
                columns.add(key)
    return sorted(columns)


def record_total(record, stages):
    if TOTAL in record and record[TOTAL] not in ("", None):
        return float(record[TOTAL])
    return sum(float(record.get(stage, 0.0) or 0.0) for stage in stages)


def analyze_row(records, stages, b0_iqr):
    """Apply the protocol to one row's timed records (list of dicts)."""
    n_timed = len(records)
    totals = [record_total(r, stages) for r in records]
    result = {"n_timed": n_timed, "stages": {}, "status": "ok", "n_used": n_timed}

    # Rule 4(a): bimodal check first, over every stage.
    bimodal = []
    for stage in stages:
        values = [float(r.get(stage, 0.0) or 0.0) for r in records]
        info = stats(values)
        if info["p25"] > 0.0:
            ratio = info["p75"] / info["p25"]
        else:
            ratio = float("inf") if info["p75"] > 0.0 else 1.0
        if ratio > BIMODAL_RATIO:
            low, high = split_at_gap(values)
            if high:
                bimodal.append(
                    {
                        "stage": stage,
                        "low": stats(low),
                        "high": stats(high),
                        "ratio": ratio,
                    }
                )

    if bimodal:
        result["status"] = "bimodal:" + ",".join(entry["stage"] for entry in bimodal)
        result["n_used"] = n_timed
        for stage in stages:
            values = [float(r.get(stage, 0.0) or 0.0) for r in records]
            result["stages"][stage] = stats(values)
        for entry in bimodal:
            result["stages"][entry["stage"]]["modes"] = {
                "low": entry["low"],
                "high": entry["high"],
                "ratio": entry["ratio"],
            }
        mode_low = 0.0
        mode_high = 0.0
        for stage in stages:
            info = result["stages"][stage]
            if "modes" in info:
                mode_low += info["modes"]["low"]["median"]
                mode_high += info["modes"]["high"]["median"]
            else:
                mode_low += info["median"]
                mode_high += info["median"]
        result["mode_low_total"] = mode_low
        result["mode_high_total"] = mode_high
        result["quotable"] = False
        result["gate_fraction"] = float("nan")
        return result

    # Rule 4(b): outlier rejection on the total and the clock guard.
    row_total_median = stats(totals)["median"]
    clocks = [r.get("clock_mhz") for r in records]
    have_clock = any(c not in (None, "") for c in clocks)
    clock_median = None
    if have_clock:
        clock_values = [float(c) for c in clocks if c not in (None, "")]
        clock_median = stats(clock_values)["median"] if clock_values else None

    keep = []
    for record, total, clock in zip(records, totals, clocks):
        if total < OUTLIER_LO * row_total_median or total > OUTLIER_HI * row_total_median:
            continue
        if have_clock and clock not in (None, ""):
            if float(clock) < CLOCK_MARGIN * clock_median:
                continue
        keep.append(record)

    result["n_used"] = len(keep)
    if len(keep) < MIN_SURVIVORS:
        result["status"] = "insufficient-survivors"
        result["quotable"] = False
        result["gate_fraction"] = float("nan")
        # Still report what survived, so the operator sees the spread.
        for stage in stages:
            result["stages"][stage] = stats([float(r.get(stage, 0.0) or 0.0) for r in keep])
        result["total"] = stats([record_total(r, stages) for r in keep])
        return result

    for stage in stages:
        result["stages"][stage] = stats([float(r.get(stage, 0.0) or 0.0) for r in keep])
    total_info = stats([record_total(r, stages) for r in keep])
    result["total"] = total_info

    if b0_iqr is None:
        result["gate_fraction"] = float("nan")
        result["quotable"] = None
    else:
        gate = 2.0 * b0_iqr
        fraction = total_info["iqr"] / total_info["median"] if total_info["median"] else float("inf")
        result["gate_fraction"] = fraction
        result["gate"] = gate
        result["quotable"] = fraction <= gate
    return result


def load_records(path):
    records = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("phase", "timed") != "timed":
                continue
            row_id = row.get("row_id") or "row"
            record = {}
            for key, value in row.items():
                if key in ("row_id", "phase"):
                    continue
                if value in (None, ""):
                    continue
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value
            records.setdefault(row_id, []).append(record)
    return records


def print_row(row_id, result):
    print(f"row {row_id}: n_timed={result['n_timed']} n_used={result['n_used']} status={result['status']}")
    for stage, info in result["stages"].items():
        print(
            f"  {stage:14s} median={info['median']:.4f} iqr={info['iqr']:.4f} "
            f"min={info['min']:.4f} max={info['max']:.4f}"
        )
        if "modes" in info:
            low, high = info["modes"]["low"], info["modes"]["high"]
            print(
                f"    modes: low n={low['n']} median={low['median']:.4f} | "
                f"high n={high['n']} median={high['median']:.4f} (P75/P25={info['modes']['ratio']:.2f})"
            )
    if "total" in result:
        info = result["total"]
        print(
            f"  {'total':14s} median={info['median']:.4f} iqr={info['iqr']:.4f} "
            f"min={info['min']:.4f} max={info['max']:.4f}"
        )
    if "mode_low_total" in result:
        print(
            f"  mode totals: low={result['mode_low_total']:.4f} "
            f"high={result['mode_high_total']:.4f} (no single number)"
        )
    if result.get("gate") is not None:
        print(
            f"  gate: IQR/median={result['gate_fraction']:.4%} vs "
            f"2xB0={result['gate']:.4%} -> {'QUOTABLE' if result['quotable'] else 'FAIL'}"
        )
    elif result.get("quotable") is None and result["status"] == "ok":
        print("  gate: not evaluated (pass --b0-iqr)")
    if result["status"] == "insufficient-survivors":
        print("  action: re-run this row once (fresh set of 9 timed repeats)")


def write_summary(path, rows, stages):
    fieldnames = ["row_id", "n_timed", "n_used", "status", "quotable", "gate_fraction"]
    for stage in stages:
        fieldnames += [f"{stage}_median", f"{stage}_iqr"]
    fieldnames += ["total_median", "total_iqr", "total_min", "total_max", "mode_low_total", "mode_high_total"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row_id, result in rows:
            flat = {
                "row_id": row_id,
                "n_timed": result["n_timed"],
                "n_used": result["n_used"],
                "status": result["status"],
                "quotable": result.get("quotable"),
                "gate_fraction": result.get("gate_fraction"),
            }
            for stage in stages:
                flat[f"{stage}_median"] = result["stages"][stage]["median"]
                flat[f"{stage}_iqr"] = result["stages"][stage]["iqr"]
            total = result.get("total")
            if total:
                flat.update(
                    total_median=total["median"],
                    total_iqr=total["iqr"],
                    total_min=total["min"],
                    total_max=total["max"],
                )
            flat["mode_low_total"] = result.get("mode_low_total")
            flat["mode_high_total"] = result.get("mode_high_total")
            writer.writerow(flat)


def run_self_test():
    """Synthetic per-repeat rows covering the four protocol branches."""

    def records(totals, fft=None, clock=None):
        out = []
        for i, total in enumerate(totals):
            record = {
                "deposit_ms": 30.0,
                "kernel_ms": 0.5,
                "fft_ms": 5.0 if fft is None else fft[i],
                "readout_ms": total - 30.0 - 0.5 - (5.0 if fft is None else fft[i]),
            }
            if clock is not None:
                record["clock_mhz"] = clock[i]
            out.append(record)
        return out

    stages = ["deposit_ms", "kernel_ms", "fft_ms", "readout_ms"]

    clean = analyze_row(records([44.0, 44.5, 43.8, 44.2, 44.1, 43.9, 44.3, 44.0, 44.4]), stages, 0.015)
    assert clean["status"] == "ok", clean["status"]
    assert clean["n_used"] == 9
    assert clean["quotable"] is True

    bimodal_fft = [5.02, 22.78, 5.02, 22.78, 5.02, 22.78, 5.02, 22.78, 22.78]
    bimodal = analyze_row(records([45.0] * 9, fft=bimodal_fft), stages, 0.015)
    assert bimodal["status"].startswith("bimodal:"), bimodal["status"]
    assert "fft_ms" in bimodal["status"], bimodal["status"]
    assert bimodal["n_used"] == 9
    modes = bimodal["stages"]["fft_ms"]["modes"]
    assert modes["low"]["n"] == 4 and modes["high"]["n"] == 5
    assert abs(modes["low"]["median"] - 5.02) < 1e-9
    assert abs(modes["high"]["median"] - 22.78) < 1e-9
    assert bimodal["mode_low_total"] < bimodal["mode_high_total"]

    outlier = records([44.0] * 9)
    outlier[4]["fft_ms"] = 5.0
    outlier[4]["readout_ms"] += 44.0  # total ~88, outside [0.7, 1.4] x 44
    outlier_result = analyze_row(outlier, stages, 0.015)
    assert outlier_result["status"] == "ok", outlier_result["status"]
    assert outlier_result["n_used"] == 8, outlier_result["n_used"]

    # Totals 40 (n=4) vs 60 (n=5): all stage ratios stay below 2.5, so this
    # exercises the outlier path, and only 5 repeats survive the drop.
    insufficient = [
        {"deposit_ms": 30.0, "kernel_ms": 0.5, "fft_ms": 5.0, "readout_ms": 4.5}
        for _ in range(4)
    ] + [
        {"deposit_ms": 40.0, "kernel_ms": 0.5, "fft_ms": 10.0, "readout_ms": 9.5}
        for _ in range(5)
    ]
    insufficient_result = analyze_row(insufficient, stages, 0.015)
    assert insufficient_result["status"] == "insufficient-survivors", insufficient_result["status"]
    assert insufficient_result["n_used"] == 5

    # Clock guard: one repeat at 50% clock is dropped.
    clocked = records([44.0] * 9, clock=[1980.0] * 9)
    clocked[2]["clock_mhz"] = 700.0
    clocked_result = analyze_row(clocked, stages, 0.015)
    assert clocked_result["n_used"] == 8, clocked_result["n_used"]

    print(
        "self-test: OK (clean, bimodal, outlier, insufficient-survivors, clock guard)"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="per-repeat CSV from picBunchBench --raw-csv")
    parser.add_argument("--b0-iqr", type=float, default=None, help="measured B0 IQR floor as a fraction (e.g. 0.015)")
    parser.add_argument("--out", help="write the per-row summary CSV here")
    parser.add_argument("--self-test", action="store_true", help="run the rule self-test")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.csv:
        parser.error("a per-repeat CSV is required unless --self-test is given")

    records = load_records(args.csv)
    if not records:
        print("summarize: no timed records found", file=sys.stderr)
        return 1
    all_stages = sorted({stage for rows in records.values() for stage in stage_columns(rows)})
    results = []
    for row_id, row_records in records.items():
        result = analyze_row(row_records, stage_columns(row_records), args.b0_iqr)
        results.append((row_id, result))
        print_row(row_id, result)
    if args.out:
        write_summary(args.out, results, all_stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
