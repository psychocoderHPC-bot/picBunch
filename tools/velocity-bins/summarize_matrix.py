#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Summarise the nextTests.md velocity-binning matrix.

Reads ``manifest.json`` and every per-run comparison JSON written by
``run_matrix.py`` and emits

  * a Markdown report with the per-case/per-K metrics, and
  * ``criteria.json`` with exactly six acceptance verdicts (section 7) plus
    the section 9 proxy check and the transverse-component view.

Every verdict is PASS, FAIL, or UNVERIFIABLE.  A metric that the measurement
pipeline does not provide is reported as UNVERIFIABLE with the missing key; it
is never replaced by an assumed value.

Usage:
  summarize_matrix.py [--manifest PATH] [--out DIR] [--markdown PATH]
                      [--criteria PATH] [--self-test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "picbunch.velocity-matrix-report/v1"
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
TRANSVERSE = ("Ey", "Ez", "By", "Bz")
FAR_FIELD_KEYS = ("far_field_rel", "far_field", "far_field_shell")
P99_KEYS = ("p99_rel", "rel_p99", "p99")
RMS_KEYS = ("rms_rel", "rel_rms", "rms")

MISSING_NOTES = {
    "far_field_rel": (
        "compare_velocity_bins.py has no 2R-offset far-field shell; its error "
        "window is nodes within 16 cells of a particle."
    ),
    "p99_rel": (
        "compare_velocity_bins.py writes only the max relative error; no 99th "
        "percentile is available."
    ),
    "rms_rel": (
        "compare_velocity_bins.py writes only the max relative error; no RMS is "
        "available."
    ),
    "in_bunch_rel": "the per-run comparison JSON has no mesh_vs_T.rel vector.",
    "in_bunch_max": "the per-run comparison JSON has no mesh_vs_T.worst_rel.",
    "startup_us": "the manifest/meta has no edge-selection startup time.",
    "nonempty_bins": "the manifest/meta has no non-empty box count.",
    "proxy_error": "the per-run comparison JSON has no model_error entry.",
    "chosen_bins": "the automatic-count K is not recoverable.",
    "output_json": "the per-run comparison JSON is missing or unreadable.",
    "wall_time_s": "a resumed run has no recoverable wall time.",
}


# ---------------------------------------------------------------------------
# loading


def _probe6(source: dict, keys: Iterable[str]) -> Optional[List[float]]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 6:
            try:
                return [float(v) for v in value]
            except (TypeError, ValueError):
                continue
    return None


def _probe_scalar(source: dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _resolve_output(manifest_path: Path, payload: dict, rel: str) -> Path:
    candidates = [manifest_path.parent / rel]
    recorded = payload.get("out_dir")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded) / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _single_run_key(payload: dict) -> Optional[str]:
    runs = payload.get("runs")
    if not isinstance(runs, dict) or not runs:
        return None
    return next(iter(runs))


def extract_run(row: dict, payload: Optional[dict]) -> dict:
    """Flatten one manifest row plus its comparison JSON into a metric dict."""
    missing = list(row.get("missing_metrics") or [])
    data = {
        "id": row.get("id"),
        "case": row.get("case"),
        "rule": row.get("rule"),
        "kind": row.get("kind"),
        "auto_count": bool(row.get("auto_count")),
        "requested_bins": row.get("requested_bins"),
        "chosen_bins": row.get("chosen_bins"),
        "in_bunch_rel": None,
        "in_bunch_max": None,
        "far_field_rel": None,
        "startup_us": row.get("startup_us"),
        "nonempty_bins": row.get("nonempty_bins"),
        "proxy_error": row.get("proxy_error"),
        "p99_rel": None,
        "rms_rel": None,
        "missing": missing,
    }
    if payload is None:
        data["missing"] = sorted(set(missing) | {"output_json"})
        return data
    key = str(data["chosen_bins"]) if data["chosen_bins"] is not None else None
    runs = payload.get("runs", {})
    if key is None or key not in runs:
        key = _single_run_key(payload)
        if key is not None:
            try:
                data["chosen_bins"] = int(key)
            except ValueError:
                pass
    if key is None or key not in runs:
        data["missing"] = sorted(set(missing) | {"output_json"})
        return data
    compare_run = runs[key]
    stats = compare_run.get("mesh_vs_T")
    if isinstance(stats, dict):
        rel = _probe6(stats, ("rel",))
        if rel is not None:
            data["in_bunch_rel"] = rel
            data["in_bunch_max"] = _probe_scalar(stats, ("worst_rel",))
            if data["in_bunch_max"] is None:
                data["in_bunch_max"] = max(rel)
        else:
            data["missing"] = sorted(set(data["missing"]) | {"in_bunch_rel"})
        far = _probe6(stats, FAR_FIELD_KEYS) or _probe6(compare_run, FAR_FIELD_KEYS)
        data["far_field_rel"] = far
        data["p99_rel"] = _probe_scalar(stats, P99_KEYS)
        data["rms_rel"] = _probe_scalar(stats, RMS_KEYS)
    else:
        data["missing"] = sorted(set(data["missing"]) | {"in_bunch_rel", "in_bunch_max"})
    if data["proxy_error"] is None:
        proxy = compare_run.get("model_error")
        if isinstance(proxy, (int, float)) and math.isfinite(float(proxy)):
            data["proxy_error"] = float(proxy)
    for key_name in ("far_field_rel", "p99_rel", "rms_rel", "startup_us",
                     "nonempty_bins", "proxy_error", "chosen_bins"):
        if data.get(key_name) is None:
            data["missing"] = sorted(set(data["missing"]) | {key_name})
    data["missing"] = sorted(set(data["missing"]))
    return data


def load_inputs(manifest_path: Path) -> Tuple[dict, List[dict]]:
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"summarize_matrix: cannot read {manifest_path}: {exc}")
    rows = payload.get("runs")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"summarize_matrix: {manifest_path} has no runs list")
    data: List[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = row.get("output")
        run_payload: Optional[dict] = None
        if isinstance(rel, str):
            path = _resolve_output(manifest_path, payload, rel)
            try:
                run_payload = json.loads(path.read_text())
            except (OSError, ValueError):
                run_payload = None
        data.append(extract_run(row, run_payload))
    return payload, data


# ---------------------------------------------------------------------------
# evaluation


def _index(data: Sequence[dict]) -> Tuple[Dict[Tuple[str, str, int], dict], Dict[str, dict]]:
    matrix: Dict[Tuple[str, str, int], dict] = {}
    auto: Dict[str, dict] = {}
    for run in data:
        if run.get("auto_count") or run.get("kind") == "auto_count":
            auto[run.get("case")] = run
        elif run.get("requested_bins") is not None:
            matrix[(run.get("case"), run.get("rule"), int(run["requested_bins"]))] = run
    return matrix, auto


def _verdict(missing: Sequence[str], outcomes: Sequence[bool]) -> str:
    if missing:
        return "UNVERIFIABLE"
    if not outcomes:
        return "UNVERIFIABLE"
    return "PASS" if all(outcomes) else "FAIL"


def _criterion(cid: int, name: str, verdict: str, driving: dict, missing: Sequence[str]) -> dict:
    return {
        "id": cid,
        "name": name,
        "verdict": verdict,
        "driving": driving,
        "missing": list(dict.fromkeys(missing)),
    }


def _max_of(run: Optional[dict], missing: List[str]) -> Optional[float]:
    if run is None:
        missing.append("run")
        return None
    value = run.get("in_bunch_max")
    if value is None:
        missing.append(f"{run.get('id')}:in_bunch_max")
        return None
    return float(value)


def criterion_control(matrix: Dict[Tuple[str, str, int], dict], bins: Sequence[int]) -> dict:
    rows: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for k in bins:
        uniform = matrix.get(("uniform", "uniform", int(k)))
        auto = matrix.get(("uniform", "auto", int(k)))
        if uniform is None:
            missing.append(f"uniform K={k} uniform-rule run")
        if auto is None:
            missing.append(f"uniform K={k} auto-rule run")
        um = _max_of(uniform, missing) if uniform is not None else None
        am = _max_of(auto, missing) if auto is not None else None
        if um is None or am is None:
            continue
        ok = am <= 1.05 * um
        rows.append({
            "K": int(k), "uniform_max": um, "auto_max": am,
            "auto_over_uniform": (am / um) if um > 0 else None, "pass": ok,
        })
        outcomes.append(ok)
    return _criterion(
        1,
        "control: auto not worse than uniform by more than 5% (all K)",
        _verdict(missing, outcomes),
        {"per_K": rows, "tolerance": 1.05},
        missing,
    )


def criterion_factor(cid: int, case: str, matrix: Dict[Tuple[str, str, int], dict]) -> dict:
    rows: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for k in (4, 8):
        uniform = matrix.get((case, "uniform", k))
        auto = matrix.get((case, "auto", k))
        if uniform is None:
            missing.append(f"{case} K={k} uniform-rule run")
        if auto is None:
            missing.append(f"{case} K={k} auto-rule run")
        um = _max_of(uniform, missing) if uniform is not None else None
        am = _max_of(auto, missing) if auto is not None else None
        if um is None or am is None:
            continue
        factor = (um / am) if am > 0 else (math.inf if um > 0 else 1.0)
        ok = um >= 2.0 * am
        rows.append({
            "K": k, "uniform_max": um, "auto_max": am,
            "uniform_over_auto": None if math.isinf(factor) else factor,
            "factor_infinite": math.isinf(factor), "pass": ok,
        })
        outcomes.append(ok)
    return _criterion(
        cid,
        f"{case}: auto at least 2x better than uniform at K=4 and K=8",
        _verdict(missing, outcomes),
        {"per_K": rows, "required_factor": 2.0},
        missing,
    )


def criterion_single(auto: Dict[str, dict]) -> dict:
    run = auto.get("single_velocity")
    missing: List[str] = []
    if run is None:
        missing.append("single_velocity automatic-count run")
        chosen = None
    else:
        chosen = run.get("chosen_bins")
        if chosen is None:
            missing.append(f"{run.get('id')}:chosen_bins")
    verdict = _verdict(missing, [chosen == 1] if chosen is not None else [])
    return _criterion(
        5,
        "single_velocity: automatic count selection returns K=1",
        verdict,
        {"chosen_bins": chosen, "expected": 1},
        missing,
    )


def criterion_monotonic(
    matrix: Dict[Tuple[str, str, int], dict], cases: Sequence[str]
) -> dict:
    per_case: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for case in cases:
        vals: List[Optional[float]] = []
        for k in (2, 4, 8):
            run = matrix.get((case, "auto", k))
            if run is None:
                missing.append(f"{case} K={k} auto-rule run")
                vals.append(None)
                continue
            value = run.get("in_bunch_max")
            if value is None:
                missing.append(f"{run.get('id')}:in_bunch_max")
            vals.append(None if value is None else float(value))
        if any(v is None for v in vals):
            per_case.append({"case": case, "K2": vals[0], "K4": vals[1],
                             "K8": vals[2], "non_increasing": None})
            continue
        ok = vals[0] >= vals[1] >= vals[2]  # type: ignore[operator]
        per_case.append({
            "case": case,
            "K2": vals[0],
            "K4": vals[1],
            "K8": vals[2],
            "delta_2_4": vals[1] - vals[0],  # type: ignore[operator]
            "delta_4_8": vals[2] - vals[1],  # type: ignore[operator]
            "non_increasing": ok,
        })
        outcomes.append(ok)
    return _criterion(
        6,
        "monotonicity (auto rule): in-bunch max relative error does not increase "
        "for K=2 -> 4 -> 8 on uniform, core_halo, exp_tail",
        _verdict(missing, outcomes),
        {"per_case": per_case, "strict_decrease_required": False},
        missing,
    )


def proxy_check(data: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in data:
        measured = run.get("in_bunch_max")
        proxy = run.get("proxy_error")
        if measured is None or proxy is None:
            continue
        ratio = (measured / proxy) if proxy > 0 else None
        flag = ratio is not None and (ratio > 2.0 or ratio < 0.5)
        rows.append({
            "id": run.get("id"),
            "case": run.get("case"),
            "rule": run.get("rule"),
            "kind": run.get("kind"),
            "K": run.get("requested_bins"),
            "chosen_bins": run.get("chosen_bins"),
            "measured_max": measured,
            "proxy_error": proxy,
            "measured_over_proxy": ratio,
            "exceeds_2x": ratio is not None and ratio > 2.0,
            "below_half": ratio is not None and ratio < 0.5,
            "flag": flag,
        })
    return rows


def transverse_view(data: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in data:
        if run.get("rule") not in ("auto", "uniform"):
            continue
        rel = run.get("in_bunch_rel")
        row = {
            "id": run.get("id"),
            "case": run.get("case"),
            "rule": run.get("rule"),
            "kind": run.get("kind"),
            "K": run.get("requested_bins"),
            "chosen_bins": run.get("chosen_bins"),
        }
        for component in TRANSVERSE:
            row[component] = rel[COMPONENTS.index(component)] if rel else None
        rows.append(row)
    return rows


def tail_diagnostics(data: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in data:
        if run.get("rule") != "auto":
            continue
        maximum = run.get("in_bunch_max")
        p99 = run.get("p99_rel")
        rms = run.get("rms_rel")
        dominated = (
            maximum is not None and p99 is not None and p99 > 0 and maximum > 10.0 * p99
        )
        rows.append({
            "id": run.get("id"),
            "case": run.get("case"),
            "kind": run.get("kind"),
            "K": run.get("requested_bins"),
            "chosen_bins": run.get("chosen_bins"),
            "max": maximum,
            "p99": p99,
            "rms": rms,
            "max_over_p99": (maximum / p99) if (maximum is not None and p99) else None,
            "near_singular_dominated": dominated,
            "missing": [key for key, value in
                        (("p99_rel", p99), ("rms_rel", rms)) if value is None],
        })
    return rows


def evaluate_criteria(data: Sequence[dict], bins: Sequence[int]) -> dict:
    matrix, auto = _index(data)
    criteria = [
        criterion_control(matrix, bins),
        criterion_factor(2, "two_cluster", matrix),
        criterion_factor(3, "core_halo", matrix),
        criterion_factor(4, "exp_tail", matrix),
        criterion_single(auto),
        criterion_monotonic(matrix, ("uniform", "core_halo", "exp_tail")),
    ]
    missing_metrics: Dict[str, dict] = {}
    for run in data:
        for key in run.get("missing", []):
            if key not in MISSING_NOTES:
                continue
            entry = missing_metrics.setdefault(
                key, {"count": 0, "note": MISSING_NOTES[key]}
            )
            entry["count"] += 1
    return {
        "criteria": criteria,
        "extra": {
            "proxy_vs_measured": proxy_check(data),
            "transverse_components": transverse_view(data),
            "tail_diagnostics": tail_diagnostics(data),
        },
        "missing_metrics": missing_metrics,
        "notes": [
            "in_bunch_max is runs[B].mesh_vs_T.worst_rel from "
            "compare_velocity_bins.py. That error window is every node within 16 "
            "cells of a particle, not strictly the bunch bounding box; the two "
            "coincide only when the bunch is small compared with 16 cells.",
            "Criterion 6 uses the auto edge rule (the rule under test), not the "
            "uniform or adaptive rule.",
            "The far-field 2R-offset shell, the 99th percentile, and the RMS of "
            "the relative error are absent from the current comparison JSON. "
            "They are reported as n/a and listed under missing_metrics.",
            "startup_us comes from PREFIX_meta_B<B>.txt bin_derive_ms, a "
            "standalone re-derivation of the edge set measured after the field "
            "solve. It is the edge-selection cost, but it is not in the "
            "comparison JSON.",
        ],
    }


# ---------------------------------------------------------------------------
# rendering


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}e}"


def _cell(value: Optional[float]) -> str:
    return fmt(value)


def render_markdown(manifest: dict, data: Sequence[dict], evaluation: dict,
                    manifest_path: Path) -> str:
    lines: List[str] = []
    lines.append("# Velocity-bin matrix report")
    lines.append("")
    lines.append(f"Manifest: `{manifest_path}`")
    counts = manifest.get("counts", {})
    lines.append(
        "Manifest rows: "
        f"{counts.get('rows', len(data))} "
        f"({counts.get('explicit', '?')} explicit + {counts.get('auto', '?')} auto); "
        f"complete: {manifest.get('complete')}."
    )
    rows = manifest.get("runs", [])
    if rows:
        keys = sorted({(run.get("case"), run.get("rule"), run.get("requested_bins"),
                        run.get("chosen_bins")) for run in data},
                      key=lambda item: (
                          str(item[0]),
                          -1 if item[2] is None else int(item[2]),
                          str(item[1]),
                      ))
        lines.append(f"Distinct (case, rule, K) combinations: {len(keys)}.")
    lines.append("")
    lines.append(
        "In-bunch errors are `runs[B].mesh_vs_T` from "
        "`compare_velocity_bins.py` (the TSC-convolved continuum reference). "
        "`n/a` means the metric is absent; it is never assumed."
    )
    lines.append("")

    lines.append("## Acceptance criteria (nextTests.md section 7)")
    lines.append("")
    lines.append("| # | criterion | verdict | driving numbers |")
    lines.append("|---|---|---|---|")
    for item in evaluation["criteria"]:
        driving = json.dumps(item["driving"], sort_keys=True)
        missing = item["missing"]
        if missing:
            driving += " | missing: " + ", ".join(missing)
        lines.append(
            f"| {item['id']} | {item['name']} | **{item['verdict']}** | "
            f"`{driving}` |"
        )
    lines.append("")

    lines.append("## Section 9 proxy vs measured (factor-2 check)")
    lines.append("")
    lines.append("| run | case | rule | K | measured max | proxy | measured/proxy | flag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in evaluation["extra"]["proxy_vs_measured"]:
        k = row["chosen_bins"] if row["kind"] == "auto_count" else row["K"]
        lines.append(
            f"| {row['id']} | {row['case']} | {row['rule']} | {k} | "
            f"{fmt(row['measured_max'])} | {fmt(row['proxy_error'])} | "
            f"{fmt(row['measured_over_proxy'])} | "
            f"{'FLAG' if row['flag'] else '-'} |"
        )
    lines.append("")
    lines.append(
        "A flag means measured/proxy is above 2 or below 0.5. "
        "Section 9 says to record both numbers and not change the rule."
    )
    lines.append("")

    lines.append("## Per-transverse-component view (Ey, Ez, By, Bz)")
    lines.append("")
    lines.append("| run | case | rule | K | Ey | Ez | By | Bz |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in evaluation["extra"]["transverse_components"]:
        k = row["chosen_bins"] if row["kind"] == "auto_count" else row["K"]
        lines.append(
            f"| {row['id']} | {row['case']} | {row['rule']} | {k} | "
            + " | ".join(_cell(row.get(component)) for component in TRANSVERSE)
            + " |"
        )
    lines.append("")
    lines.append(
        "These four components are the honest surrogate for the transverse-drift "
        "motivation; the beam-axis components are excluded."
    )
    lines.append("")

    lines.append("## Tail diagnostics (max vs 99th percentile vs RMS)")
    lines.append("")
    lines.append("| run | case | K | max | p99 | RMS | max/p99 | near-singular |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in evaluation["extra"]["tail_diagnostics"]:
        k = row["chosen_bins"] if row["kind"] == "auto_count" else row["K"]
        lines.append(
            f"| {row['id']} | {row['case']} | {k} | {fmt(row['max'])} | "
            f"{fmt(row['p99'])} | {fmt(row['rms'])} | "
            f"{fmt(row['max_over_p99'])} | "
            f"{'yes' if row['near_singular_dominated'] else '-'} |"
        )
    if any(row["missing"] for row in evaluation["extra"]["tail_diagnostics"]):
        lines.append("")
        lines.append(
            "`n/a` here is a pipeline gap: `compare_velocity_bins.py` writes only "
            "the maximum relative error, so a non-monotone max cannot be diagnosed "
            "from percentiles or RMS until that script emits them."
        )
    lines.append("")

    lines.append("## Per-case tables")
    for case in sorted({run["case"] for run in data if run.get("case")}):
        lines.append("")
        lines.append(f"### {case}")
        lines.append("")
        lines.append(
            "| rule | K | Ex | Ey | Ez | Bx | By | Bz | "
            "FF Ex | FF Ey | FF Ez | FF Bx | FF By | FF Bz | "
            "non-empty/K | startup_us | proxy | measured | measured/proxy |"
        )
        lines.append("|" + "---|" * 19)
        case_runs = [run for run in data if run.get("case") == case]
        case_runs.sort(key=lambda run: (
            0 if run.get("requested_bins") is not None else 1,
            -1 if run.get("requested_bins") is None else int(run["requested_bins"]),
            {"uniform": 0, "adaptive": 1, "auto": 2}.get(run.get("rule"), 9),
        ))
        for run in case_runs:
            rel = run.get("in_bunch_rel") or [None] * 6
            far = run.get("far_field_rel") or [None] * 6
            k_value = (run.get("chosen_bins") if run.get("kind") == "auto_count"
                       else run.get("requested_bins"))
            k_label = f"auto({k_value})" if run.get("kind") == "auto_count" else str(k_value)
            nonempty = run.get("nonempty_bins")
            nonempty_label = (
                f"{nonempty}/{k_value}" if nonempty is not None and k_value is not None
                else "n/a"
            )
            measured = run.get("in_bunch_max")
            proxy = run.get("proxy_error")
            ratio = (measured / proxy) if (measured is not None and proxy) else None
            lines.append(
                f"| {run.get('rule')} | {k_label} | "
                + " | ".join(_cell(v) for v in rel)
                + " | "
                + " | ".join(_cell(v) for v in far)
                + f" | {nonempty_label} | {fmt(run.get('startup_us'), 1)} | "
                f"{fmt(proxy)} | {fmt(measured)} | {fmt(ratio)} |"
            )
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    for note in evaluation.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Missing metrics")
    lines.append("")
    missing = evaluation["missing_metrics"]
    if not missing:
        lines.append("None.")
    else:
        lines.append("| metric | runs missing it | note |")
        lines.append("|---|---|---|")
        for key in sorted(missing):
            entry = missing[key]
            lines.append(f"| {key} | {entry['count']} | {entry.get('note', '')} |")
    lines.append("")
    lines.append(
        "These are gaps in the current measurement/comparison outputs, reported "
        "so the C++ measurement node can add them. The runner does not fabricate "
        "values."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# self-test


def _base_errors(profile: str) -> Dict[Tuple[str, str, int], float]:
    values: Dict[Tuple[str, str, int], float] = {}
    uniform = {2: 0.5, 4: 0.3, 8: 0.2}
    adaptive = {2: 0.4, 4: 0.2, 8: 0.1}
    auto = {2: 0.5, 4: 0.1, 8: 0.05}
    for k, value in uniform.items():
        values[("uniform", "uniform", k)] = value
        values[("uniform", "adaptive", k)] = adaptive[k]
        values[("uniform", "auto", k)] = value
    for case in ("two_cluster", "core_halo", "exp_tail"):
        for k, value in {2: 0.5, 4: 0.4, 8: 0.3}.items():
            values[(case, "uniform", k)] = value
        for k, value in auto.items():
            values[(case, "auto", k)] = value
        for k, value in adaptive.items():
            values[(case, "adaptive", k)] = value
    for case in ("skewed", "single_velocity"):
        for k, value in {2: 0.01, 4: 0.01, 8: 0.01}.items():
            values[(case, "uniform", k)] = value
            values[(case, "adaptive", k)] = value
            values[(case, "auto", k)] = value
    if profile == "fail":
        values.update({
            ("uniform", "auto", 2): 0.55,
            ("uniform", "auto", 4): 0.6,
            ("uniform", "auto", 8): 0.7,
            ("two_cluster", "auto", 4): 0.3,
            ("two_cluster", "auto", 8): 0.2,
            ("core_halo", "auto", 4): 0.35,
            ("core_halo", "auto", 8): 0.4,
            ("exp_tail", "auto", 4): 0.3,
            ("exp_tail", "auto", 8): 0.25,
        })
    return values


def _synthetic_data(profile: str) -> List[dict]:
    cases = ("uniform", "two_cluster", "core_halo", "exp_tail", "skewed", "single_velocity")
    errors = _base_errors(profile)
    data: List[dict] = []
    index = 0
    for case in cases:
        for rule in ("uniform", "adaptive", "auto"):
            for k in (2, 4, 8):
                value = errors[(case, rule, k)]
                data.append({
                    "id": f"{case}_{rule}_K{k}",
                    "index": index,
                    "case": case,
                    "rule": rule,
                    "kind": "matrix",
                    "auto_count": False,
                    "requested_bins": k,
                    "chosen_bins": k,
                    "in_bunch_rel": [value] * 6,
                    "in_bunch_max": value,
                    "far_field_rel": None,
                    "startup_us": 500.0,
                    "nonempty_bins": k,
                    "proxy_error": value * 1.1,
                    "p99_rel": None,
                    "rms_rel": None,
                    "missing": ["far_field_rel", "p99_rel", "rms_rel"],
                })
                index += 1
        chosen = 2
        if profile != "fail" and case == "single_velocity":
            chosen = 1
        value = errors[(case, "auto", 2)]
        data.append({
            "id": f"{case}_auto_Kauto",
            "index": index,
            "case": case,
            "rule": "auto",
            "kind": "auto_count",
            "auto_count": True,
            "requested_bins": None,
            "chosen_bins": chosen,
            "in_bunch_rel": [value] * 6,
            "in_bunch_max": value,
            "far_field_rel": None,
            "startup_us": 500.0,
            "nonempty_bins": chosen,
            "proxy_error": value * 1.1,
            "p99_rel": None,
            "rms_rel": None,
            "missing": ["far_field_rel", "p99_rel", "rms_rel"],
        })
        index += 1
    if profile == "missing":
        data = [run for run in data
                if run["id"] not in ("two_cluster_auto_K8", "single_velocity_auto_Kauto")]
    return data


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_synthetic_tree(tmp: Path, profile: str) -> Path:
    data = _synthetic_data(profile)
    runs = []
    for run in data:
        k = run["chosen_bins"]
        compare = {
            "bunch": {"sources": 1000, "gamma": 50.0},
            "runs": {
                str(k): {
                    "bins_used": k,
                    "per_axis": [k, 1, 1],
                    "model_sum": run["proxy_error"],
                    "model_error": run["proxy_error"],
                    "mesh_vs_T": {
                        "rel": run["in_bunch_rel"],
                        "worst_rel": run["in_bunch_max"],
                        "bands": [],
                        "dominant_band": None,
                    },
                    "Tb_vs_T": {"rel": run["in_bunch_rel"], "worst_rel": run["in_bunch_max"]},
                    "mesh_vs_Tb": {"rel": run["in_bunch_rel"], "worst_rel": run["in_bunch_max"]},
                    "outside_window_roundoff": 0.0,
                }
            },
        }
        name = f"{run['id']}.json"
        (tmp / name).write_text(json.dumps(compare))
        import hashlib
        digest = hashlib.sha256((tmp / name).read_bytes()).hexdigest()
        runs.append({
            "id": run["id"],
            "index": run["index"],
            "lane": 0,
            "case": run["case"],
            "rule": run["rule"],
            "kind": run["kind"],
            "auto_count": run["auto_count"],
            "requested_bins": run["requested_bins"],
            "chosen_bins": run["chosen_bins"],
            "output": name,
            "sha256": digest,
            "wall_time_s": 1.0,
            "measure_wall_s": 0.9,
            "compare_wall_s": 0.1,
            "startup_us": run["startup_us"],
            "nonempty_bins": run["nonempty_bins"],
            "proxy_error": run["proxy_error"],
            "compare_bins_key": str(k),
            "missing_metrics": run["missing"],
            "missing_metric_notes": [],
            "status": "ok",
        })
    manifest = {
        "schema": "picbunch.velocity-matrix/v1",
        "complete": True,
        "lane": 0,
        "lanes": 1,
        "bins": [2, 4, 8],
        "rules": ["uniform", "adaptive", "auto"],
        "out_dir": str(tmp),
        "counts": {"rows": len(runs),
                   "explicit": sum(1 for r in runs if not r["auto_count"]),
                   "auto": sum(1 for r in runs if r["auto_count"])},
        "runs": runs,
    }
    path = tmp / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def self_test() -> int:
    failures: List[str] = []

    def check(profile: str, expected: Dict[int, str]) -> None:
        data = _synthetic_data(profile)
        evaluation = evaluate_criteria(data, (2, 4, 8))
        got = {item["id"]: item["verdict"] for item in evaluation["criteria"]}
        try:
            _assert(len(evaluation["criteria"]) == 6, "criteria count is not 6")
            for cid, verdict in expected.items():
                _assert(got.get(cid) == verdict,
                        f"{profile}: criterion {cid} is {got.get(cid)}, expected {verdict}")
        except AssertionError as exc:
            failures.append(str(exc))

    check("pass", {1: "PASS", 2: "PASS", 3: "PASS", 4: "PASS", 5: "PASS", 6: "PASS"})
    check("fail", {1: "FAIL", 2: "FAIL", 3: "FAIL", 4: "FAIL", 5: "FAIL", 6: "FAIL"})
    check("missing", {1: "PASS", 2: "UNVERIFIABLE", 3: "PASS", 4: "PASS",
                      5: "UNVERIFIABLE", 6: "PASS"})

    # File-driven round trip: manifest + per-run JSONs -> markdown + criteria.
    with tempfile.TemporaryDirectory(prefix="velocity-summary-selftest-") as tmp_name:
        tmp = Path(tmp_name)
        try:
            manifest_path = _write_synthetic_tree(tmp, "pass")
            manifest, data = load_inputs(manifest_path)
            evaluation = evaluate_criteria(data, tuple(manifest.get("bins", [2, 4, 8])))
            markdown = render_markdown(manifest, data, evaluation, manifest_path)
            criteria = {
                "schema": SCHEMA,
                "manifest": str(manifest_path),
                "complete": manifest.get("complete"),
                "criteria": evaluation["criteria"],
                "extra": evaluation["extra"],
                "missing_metrics": evaluation["missing_metrics"],
            }
            (tmp / "report.md").write_text(markdown)
            (tmp / "criteria.json").write_text(json.dumps(criteria))
            reloaded = json.loads((tmp / "criteria.json").read_text())
            _assert(len(reloaded["criteria"]) == 6, "criteria.json does not have six entries")
            _assert(bool(markdown.strip()), "markdown report is empty")
            _assert("Acceptance criteria" in markdown, "markdown lacks the criteria section")
            _assert(all(item["verdict"] == "PASS" for item in reloaded["criteria"]),
                    "file-driven pass profile did not give six PASS verdicts")
        except (AssertionError, OSError, ValueError) as exc:
            failures.append(f"file round trip: {exc}")

    if failures:
        print("SELF-TEST FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("SELF-TEST OK: 3 profiles, 6 criteria each, file round trip")
    return 0


# ---------------------------------------------------------------------------
# main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: manifest directory)")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--criteria", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    manifest_path: Path = args.manifest
    if not manifest_path.is_file():
        print(f"summarize_matrix: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest, data = load_inputs(manifest_path)
    bins = manifest.get("bins") or sorted(
        {int(run["requested_bins"]) for run in data if run.get("requested_bins") is not None}
    )
    evaluation = evaluate_criteria(data, bins)
    markdown = render_markdown(manifest, data, evaluation, manifest_path)

    out_dir = args.out if args.out is not None else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.markdown if args.markdown is not None else out_dir / "velocity_matrix_report.md"
    criteria_path = args.criteria if args.criteria is not None else out_dir / "criteria.json"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    criteria_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown + "\n")
    payload = {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "complete": manifest.get("complete"),
        "bins": list(bins),
        "criteria": evaluation["criteria"],
        "extra": evaluation["extra"],
        "missing_metrics": evaluation["missing_metrics"],
        "notes": evaluation.get("notes", []),
    }
    criteria_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"wrote {markdown_path}")
    print(f"wrote {criteria_path}")
    for item in evaluation["criteria"]:
        print(f"criterion {item['id']}: {item['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
