#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Summarise the nextTests.md velocity-binning matrix.

Reads ``manifest.json`` and every per-run comparison JSON written by
``run_matrix.py``.  All acceptance verdicts and reported errors come from the
comparison JSON's ``metrics_v2`` block:

  ``runs[B].metrics_v2.in_bunch``    nodes inside the particle bounding box
  ``runs[B].metrics_v2.far_field``   the bounding box translated by +2 R

Each region carries ``n_nodes``, ``worst_rel``, ``worst_component`` and
``per_component{Ex..Bz}`` with ``n``, ``max_rel``, ``max_abs``,
``max_rel_node``, ``max_rel_position``, ``p99_rel`` and ``rms_rel``.

The legacy ``mesh_vs_T`` window (all nodes within 16 cells of a particle) is
*not* used for any criterion.  When ``metrics_v2`` or one of its keys is absent,
the affected verdict is UNVERIFIABLE and the missing key is printed.  A value is
never substituted.

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
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA = "picbunch.velocity-matrix-report/v2"
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
TRANSVERSE = ("Ey", "Ez", "By", "Bz")
REGIONS = ("in_bunch", "far_field")
REGION_LABELS = {"in_bunch": "in-bunch (particle bbox)", "far_field": "far-field (bbox +2R)"}
PER_COMPONENT_KEYS = (
    "max_rel", "max_abs", "max_rel_node", "max_rel_position", "p99_rel",
    "rms_rel", "n", "n_finite", "n_nonfinite",
)

MISSING_NOTES = {
    "metrics_v2": (
        "the comparison JSON has no metrics_v2 block; the bunch-box, far-field, "
        "p99 and RMS metrics are unavailable. The legacy 16-cell mesh_vs_T "
        "window is not used as a substitute."
    ),
    "in_bunch_max": "metrics_v2.in_bunch.worst_rel is absent.",
    "far_field_max": "metrics_v2.far_field.worst_rel is absent.",
    "p99_rel": "metrics_v2.*.per_component[*].p99_rel is absent.",
    "rms_rel": "metrics_v2.*.per_component[*].rms_rel is absent.",
    "startup_us": "the manifest row has no edge-selection startup time.",
    "nonempty_bins": "the manifest row has no non-empty box count.",
    "proxy_error": "the comparison JSON has no model_error entry.",
    "chosen_bins": "the automatic-count K is not recoverable.",
    "output_json": "the per-run comparison JSON is missing or unreadable.",
}


# ---------------------------------------------------------------------------
# loading


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


def _finite(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _extract_region(block: dict, name: str) -> Optional[dict]:
    region = block.get(name)
    if not isinstance(region, dict):
        return None
    per_component: Dict[str, dict] = {}
    raw = region.get("per_component")
    if isinstance(raw, dict):
        for component in COMPONENTS:
            stats = raw.get(component)
            if not isinstance(stats, dict):
                continue
            per_component[component] = {
                key: stats.get(key) for key in PER_COMPONENT_KEYS
            }
    return {
        "n_nodes": region.get("n_nodes"),
        "worst_rel": _finite(region.get("worst_rel")),
        "worst_component": region.get("worst_component"),
        "per_component": per_component,
    }


def extract_run(row: dict, payload: Optional[dict]) -> dict:
    """Flatten one manifest row plus its comparison JSON into a metric dict."""
    missing: List[str] = []
    data = {
        "id": row.get("id"),
        "case": row.get("case"),
        "rule": row.get("rule"),
        "kind": row.get("kind"),
        "auto_count": bool(row.get("auto_count")),
        "requested_bins": row.get("requested_bins"),
        "chosen_bins": row.get("chosen_bins"),
        "per_axis": row.get("per_axis"),
        "per_axis_rules": row.get("per_axis_rules"),
        "startup_us": _finite(row.get("startup_us")),
        "nonempty_bins": row.get("nonempty_bins"),
        "proxy_error": _finite(row.get("proxy_error")),
        "metrics_v2_present": False,
        "in_bunch": None,
        "far_field": None,
        "missing": missing,
    }
    if payload is None:
        missing.append("output_json")
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
        missing.append("output_json")
        return data
    compare_run = runs[key]

    block = compare_run.get("metrics_v2")
    if not isinstance(block, dict):
        missing.append("metrics_v2")
    else:
        data["metrics_v2_present"] = True
        for name in REGIONS:
            region = _extract_region(block, name)
            data[name] = region
            if region is None:
                missing.append(f"metrics_v2.{name}")
                continue
            if region["worst_rel"] is None:
                missing.append(f"metrics_v2.{name}.worst_rel")
            per_component = region["per_component"]
            if per_component:
                if any(s.get("p99_rel") is None for s in per_component.values()):
                    missing.append(f"metrics_v2.{name}.per_component.p99_rel")
                if any(s.get("rms_rel") is None for s in per_component.values()):
                    missing.append(f"metrics_v2.{name}.per_component.rms_rel")
            else:
                missing.append(f"metrics_v2.{name}.per_component")

    if data["proxy_error"] is None:
        data["proxy_error"] = _finite(compare_run.get("model_error"))
    if data["startup_us"] is None:
        missing.append("startup_us")
    if data["nonempty_bins"] is None:
        missing.append("nonempty_bins")
    if data["proxy_error"] is None:
        missing.append("proxy_error")
    if data["chosen_bins"] is None:
        missing.append("chosen_bins")
    data["missing"] = sorted(set(missing))
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


def _criterion(cid: int, name: str, verdict: str, driving: dict,
               missing: Sequence[str]) -> dict:
    return {
        "id": cid,
        "name": name,
        "verdict": verdict,
        "driving": driving,
        "missing": list(dict.fromkeys(missing)),
    }


def _region_max(run: Optional[dict], region: str, missing: List[str],
                label: str) -> Optional[float]:
    if run is None:
        missing.append(f"{label}:run")
        return None
    block = run.get(region) or {}
    value = block.get("worst_rel")
    if value is None:
        missing.append(f"{label}:metrics_v2.{region}.worst_rel")
        return None
    return float(value)


def criterion_control(matrix: Dict[Tuple[str, str, int], dict],
                      bins: Sequence[int]) -> dict:
    rows: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for k in bins:
        uniform = matrix.get(("uniform", "uniform", int(k)))
        auto = matrix.get(("uniform", "auto", int(k)))
        um = _region_max(uniform, "in_bunch", missing,
                         f"uniform/uniform_K{k}") if uniform else None
        am = _region_max(auto, "in_bunch", missing,
                         f"uniform/auto_K{k}") if auto else None
        if uniform is None:
            missing.append(f"uniform/uniform_K{k}:run")
        if auto is None:
            missing.append(f"uniform/auto_K{k}:run")
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
        "uniform: auto in-bunch max relative error within 5% of uniform (all K)",
        _verdict(missing, outcomes),
        {"per_K": rows, "tolerance": 1.05},
        missing,
    )


def criterion_factor(cid: int, case: str,
                     matrix: Dict[Tuple[str, str, int], dict]) -> dict:
    rows: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for k in (4, 8):
        uniform = matrix.get((case, "uniform", k))
        auto = matrix.get((case, "auto", k))
        um = _region_max(uniform, "in_bunch", missing,
                         f"{case}/uniform_K{k}") if uniform else None
        am = _region_max(auto, "in_bunch", missing,
                         f"{case}/auto_K{k}") if auto else None
        if uniform is None:
            missing.append(f"{case}/uniform_K{k}:run")
        if auto is None:
            missing.append(f"{case}/auto_K{k}:run")
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
        missing.append("single_velocity_auto_Kauto:run")
        chosen = None
    else:
        chosen = run.get("chosen_bins")
        if chosen is None:
            missing.append("single_velocity_auto_Kauto:chosen_bins")
    verdict = _verdict(missing, [chosen == 1] if chosen is not None else [])
    return _criterion(
        5,
        "single_velocity: automatic count selection returns K=1",
        verdict,
        {"chosen_bins": chosen, "expected": 1},
        missing,
    )


def criterion_monotonic(matrix: Dict[Tuple[str, str, int], dict],
                        cases: Sequence[str]) -> dict:
    per_case: List[dict] = []
    missing: List[str] = []
    outcomes: List[bool] = []
    for case in cases:
        values: List[Optional[float]] = []
        for k in (2, 4, 8):
            run = matrix.get((case, "auto", k))
            if run is None:
                missing.append(f"{case}/auto_K{k}:run")
                values.append(None)
                continue
            values.append(_region_max(run, "in_bunch", missing,
                                      f"{case}/auto_K{k}"))
        if any(v is None for v in values):
            per_case.append({"case": case, "K2": values[0], "K4": values[1],
                             "K8": values[2], "strictly_decreasing": None})
            continue
        ok = values[0] > values[1] > values[2]  # type: ignore[operator]
        per_case.append({
            "case": case,
            "K2": values[0],
            "K4": values[1],
            "K8": values[2],
            "delta_2_4": values[1] - values[0],  # type: ignore[operator]
            "delta_4_8": values[2] - values[1],  # type: ignore[operator]
            "strictly_decreasing": ok,
        })
        outcomes.append(ok)
    return _criterion(
        6,
        "uniform, core_halo, exp_tail: auto in-bunch max strictly decreases "
        "for K=2 -> 4 -> 8",
        _verdict(missing, outcomes),
        {"per_case": per_case, "strict_decrease_required": True},
        missing,
    )


def proxy_check(data: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in data:
        block = run.get("in_bunch") or {}
        measured = block.get("worst_rel")
        proxy = run.get("proxy_error")
        if measured is None or proxy is None:
            continue
        ratio = (measured / proxy) if proxy > 0 else None
        rows.append({
            "id": run.get("id"),
            "case": run.get("case"),
            "rule": run.get("rule"),
            "kind": run.get("kind"),
            "K": run.get("requested_bins"),
            "chosen_bins": run.get("chosen_bins"),
            "measured_in_bunch_max": measured,
            "proxy_error": proxy,
            "measured_over_proxy": ratio,
            "exceeds_2x": ratio is not None and ratio > 2.0,
            "flag": ratio is not None and ratio > 2.0,
        })
    return rows


def component_table(data: Sequence[dict]) -> List[dict]:
    """Per-transverse-component max/p99/RMS for in-bunch and far-field."""
    rows: List[dict] = []
    for run in data:
        row = {
            "id": run.get("id"),
            "case": run.get("case"),
            "rule": run.get("rule"),
            "kind": run.get("kind"),
            "K": run.get("requested_bins"),
            "chosen_bins": run.get("chosen_bins"),
            "regions": {},
        }
        for region in REGIONS:
            block = run.get(region) or {}
            per_component = block.get("per_component") or {}
            row["regions"][region] = {
                component: per_component.get(component) or {}
                for component in TRANSVERSE
            }
        rows.append(row)
    return rows


def box_count_view(data: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in data:
        k_value = (run.get("chosen_bins") if run.get("kind") == "auto_count"
                   else run.get("requested_bins"))
        nonempty = run.get("nonempty_bins")
        rows.append({
            "id": run.get("id"),
            "case": run.get("case"),
            "rule": run.get("rule"),
            "kind": run.get("kind"),
            "K": k_value,
            "nonempty_bins": nonempty,
            "nonempty_over_K": (
                (nonempty / k_value)
                if (nonempty is not None and isinstance(k_value, int) and k_value > 0)
                else None
            ),
            "per_axis": run.get("per_axis"),
            "per_axis_rules": run.get("per_axis_rules"),
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
            base = key.split(":", 1)[-1]
            if base.startswith("metrics_v2.") and base.endswith("per_component"):
                base = "metrics_v2"
            if base.endswith(".worst_rel"):
                base = "in_bunch_max" if ".in_bunch." in base else "far_field_max"
            if base.endswith("p99_rel"):
                base = "p99_rel"
            if base.endswith("rms_rel"):
                base = "rms_rel"
            entry = missing_metrics.setdefault(
                base, {"count": 0, "note": MISSING_NOTES.get(base, "")}
            )
            entry["count"] += 1
    return {
        "criteria": criteria,
        "extra": {
            "proxy_vs_measured": proxy_check(data),
            "per_transverse_component": component_table(data),
            "nonempty_vs_K": box_count_view(data),
        },
        "missing_metrics": missing_metrics,
        "notes": [
            "Every criterion uses metrics_v2.in_bunch.worst_rel, which is the "
            "maximum relative error over base nodes inside the particle bounding "
            "box. The legacy mesh_vs_T window (all nodes within 16 cells of a "
            "particle) is not used.",
            "far_field is the particle bounding box translated by +2R on every "
            "axis, clipped to the domain by compare_velocity_bins.py. A clipped "
            "or empty mask is reported as-is; it is not widened.",
            "Criterion 6 requires a strict decrease, not a non-increase.",
            "p99 and RMS are per-component metrics_v2 values; they are shown "
            "next to every max instead of being folded into one number.",
            "Criterion verdicts are always PASS, FAIL or UNVERIFIABLE. A missing "
            "key yields UNVERIFIABLE with that key listed.",
        ],
    }


# ---------------------------------------------------------------------------
# rendering


def fmt(value, digits: int = 3) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}e}"


def _component_cell(stats: dict, key: str) -> str:
    if not stats:
        return "UNVERIFIABLE(metrics_v2)"
    value = stats.get(key)
    if key in ("p99_rel", "rms_rel") and value is None:
        return f"UNVERIFIABLE({key})"
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return "(" + ",".join(str(int(v)) for v in value) + ")"
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
    geometry = manifest.get("geometry")
    if isinstance(geometry, dict):
        lines.append(
            "Geometry: cell={} m, grid={}, origin={} m, centre={} m, profile={}.".format(
                geometry.get("cell"), geometry.get("grid"),
                geometry.get("origin"), geometry.get("center"),
                manifest.get("profile", "default"))
        )
    lines.append("")
    lines.append(
        "Verdicts and numbers use `metrics_v2` from the comparison JSON: "
        "`in_bunch` is the particle bounding box, `far_field` is that box "
        "translated by +2R on every axis and clipped to the domain. "
        "`UNVERIFIABLE` means the key is absent; no value is substituted."
    )
    lines.append("")

    lines.append("## Acceptance criteria (nextTests.md section 7)")
    lines.append("")
    lines.append("| # | criterion | verdict | driving numbers |")
    lines.append("|---|---|---|---|")
    for item in evaluation["criteria"]:
        driving = json.dumps(item["driving"], sort_keys=True)
        if item["missing"]:
            driving += " | missing: " + ", ".join(item["missing"])
        lines.append(
            f"| {item['id']} | {item['name']} | **{item['verdict']}** | "
            f"`{driving}` |"
        )
    lines.append("")

    lines.append("## Non-empty boxes versus K")
    lines.append("")
    lines.append("| run | case | rule | K | non-empty | non-empty/K | per-axis | axis_edges |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in evaluation["extra"]["nonempty_vs_K"]:
        rules = row.get("per_axis_rules")
        rules_label = ",".join(rules) if rules else "UNVERIFIABLE(axis_edges)"
        nonempty = (row["nonempty_bins"] if row["nonempty_bins"] is not None
                    else "UNVERIFIABLE(nonempty_bins)")
        lines.append(
            f"| {row['id']} | {row['case']} | {row['rule']} | {row['K']} | "
            f"{nonempty} | {fmt(row['nonempty_over_K'])} | "
            f"{row['per_axis'] or 'UNVERIFIABLE(per_axis)'} | {rules_label} |"
        )
    lines.append("")
    lines.append(
        "`per_axis` is the total K distributed over x,y,z (for example "
        "`2 2 1` for K=4), never K per axis. `axis_edges` is the per-axis rule "
        "list the auto edge policy chose."
    )
    lines.append("")

    lines.append("## Section 9 proxy vs measured (flag when > 2)")
    lines.append("")
    lines.append("| run | case | rule | K | in-bunch max | proxy | measured/proxy | flag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in evaluation["extra"]["proxy_vs_measured"]:
        k = row["chosen_bins"] if row["kind"] == "auto_count" else row["K"]
        lines.append(
            f"| {row['id']} | {row['case']} | {row['rule']} | {k} | "
            f"{fmt(row['measured_in_bunch_max'])} | {fmt(row['proxy_error'])} | "
            f"{fmt(row['measured_over_proxy'])} | "
            f"{'FLAG' if row['flag'] else '-'} |"
        )
    lines.append("")
    lines.append(
        "Section 9 says to record both numbers; a flag means the measured "
        "in-bunch error exceeds twice the analytic proxy. The rule is not changed."
    )
    lines.append("")

    for region in REGIONS:
        lines.append(f"## Per-transverse-component max / p99 / RMS: {REGION_LABELS[region]}")
        lines.append("")
        header = "| run | case | rule | K |"
        divider = "|---|---|---|---|"
        for component in TRANSVERSE:
            header += f" {component} max | {component} p99 | {component} RMS |"
            divider += "---|---|---|"
        lines.append(header)
        lines.append(divider)
        for row in evaluation["extra"]["per_transverse_component"]:
            stats = row["regions"].get(region, {})
            k = row["chosen_bins"] if row["kind"] == "auto_count" else row["K"]
            cells = []
            for component in TRANSVERSE:
                values = stats.get(component) or {}
                cells.append(_component_cell(values, "max_rel"))
                cells.append(_component_cell(values, "p99_rel"))
                cells.append(_component_cell(values, "rms_rel"))
            lines.append(
                f"| {row['id']} | {row['case']} | {row['rule']} | {k} | "
                + " | ".join(cells) + " |"
            )
        appended = any(
            any(any(component is not None for component in
                    (row["regions"].get(region, {}).get(name) or {}).values())
                for name in TRANSVERSE)
            for row in evaluation["extra"]["per_transverse_component"]
        )
        if not appended:
            lines.append("")
            lines.append(
                "All cells are UNVERIFIABLE: this region has no per-component "
                "metrics (missing metrics_v2 or an empty mask)."
            )
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

    lines.append("## Interpretation notes")
    lines.append("")
    for note in evaluation.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# self-test


def _per_component(value: float) -> dict:
    out = {}
    for component in COMPONENTS:
        out[component] = {
            "n": 64, "n_finite": 64, "n_nonfinite": 0,
            "max_rel": value, "max_abs": value * 1.0e3,
            "max_rel_node": [1, 1, 1],
            "max_rel_position": [1.0e-6, 1.0e-6, 1.0e-6],
            "p99_rel": value * 0.5, "rms_rel": value * 0.25,
        }
    return out


def _metrics_v2(in_max: float, far_max: float) -> dict:
    return {
        "schema": "velocity-bin-metrics/2.0",
        "comparison": "mesh_vs_T",
        "in_bunch": {
            "n_nodes": 512, "worst_rel": in_max,
            "worst_component": "Ex",
            "per_component": _per_component(in_max),
        },
        "far_field": {
            "n_nodes": 96, "worst_rel": far_max,
            "worst_component": "Bz",
            "per_component": _per_component(far_max),
        },
    }


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
    if profile == "in_bunch_fail_far_ok":
        # Criterion 2 fails on the in-bunch metric at K=4 while the far-field
        # metric stays small: the summarizer must not let far-field rescue it.
        errors[("two_cluster", "auto", 4)] = 0.3
    data: List[dict] = []

    def add(case: str, rule: str, kind: str, requested, chosen, value: float) -> None:
        in_max = value
        far_max = 1.0e-9
        if profile == "no_metrics_v2":
            in_region = None
            far_region = None
            block_present = False
            missing = ["metrics_v2"]
        else:
            in_region = _metrics_v2(in_max, far_max)["in_bunch"]
            far_region = _metrics_v2(in_max, far_max)["far_field"]
            block_present = True
            missing = []
        data.append({
            "id": f"{case}_{rule}_K{requested if requested else 'auto'}",
            "case": case,
            "rule": rule,
            "kind": kind,
            "auto_count": kind == "auto_count",
            "requested_bins": requested,
            "chosen_bins": chosen,
            "per_axis": None,
            "per_axis_rules": None,
            "startup_us": 500.0,
            "nonempty_bins": chosen,
            "proxy_error": value * 1.1,
            "metrics_v2_present": block_present,
            "in_bunch": in_region,
            "far_field": far_region,
            "missing": missing,
        })

    for case in cases:
        for rule in ("uniform", "adaptive", "auto"):
            for k in (2, 4, 8):
                add(case, rule, "matrix", k, k, errors[(case, rule, k)])
        chosen = 2
        if profile != "fail" and case == "single_velocity":
            chosen = 1
        add(case, "auto", "auto_count", None, chosen, errors[(case, "auto", 2)])
    if profile == "missing_run":
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
                    "mesh_vs_T": {"rel": [0.9] * 6, "worst_rel": 0.9},
                }
            },
        }
        if run.get("metrics_v2_present"):
            block = _metrics_v2(0.0, 0.0)
            block["in_bunch"] = run["in_bunch"]
            block["far_field"] = run["far_field"]
            compare["runs"][str(k)]["metrics_v2"] = block
        name = f"{run['id']}.json"
        (tmp / name).write_text(json.dumps(compare))
        import hashlib
        digest = hashlib.sha256((tmp / name).read_bytes()).hexdigest()
        runs.append({
            "id": run["id"],
            "index": len(runs),
            "lane": 0,
            "case": run["case"],
            "rule": run["rule"],
            "kind": run["kind"],
            "auto_count": run["auto_count"],
            "requested_bins": run["requested_bins"],
            "chosen_bins": run["chosen_bins"],
            "per_axis": run["per_axis"],
            "per_axis_rules": run["per_axis_rules"],
            "output": name,
            "sha256": digest,
            "wall_time_s": 1.0,
            "measure_wall_s": 0.9,
            "compare_wall_s": 0.1,
            "startup_us": run["startup_us"],
            "nonempty_bins": run["nonempty_bins"],
            "proxy_error": run["proxy_error"],
            "metrics_v2": {"present": run.get("metrics_v2_present", False)},
            "compare_bins_key": str(k),
            "missing_metrics": run["missing"],
            "missing_metric_notes": [],
            "profile": "self-test",
            "status": "ok",
        })
    manifest = {
        "schema": "picbunch.velocity-matrix/v1",
        "complete": True,
        "lane": 0,
        "lanes": 1,
        "bins": [2, 4, 8],
        "rules": ["uniform", "adaptive", "auto"],
        "geometry": {"cell": [2.5e-7] * 3, "grid": [24, 16, 16],
                     "origin": [0.0] * 3, "center": [3e-6, 2e-6, 2e-6]},
        "profile": "self-test",
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
    check("missing_run", {1: "PASS", 2: "UNVERIFIABLE", 3: "PASS", 4: "PASS",
                          5: "UNVERIFIABLE", 6: "PASS"})
    # in-bunch fails while far-field stays small: verdict must come from
    # in-bunch, and far-field must still be reported.
    check("in_bunch_fail_far_ok", {2: "FAIL"})
    # no metrics_v2 at all: every criterion UNVERIFIABLE and no crash.
    # Criterion 5 reads chosen_bins from the meta/manifest, not metrics_v2, so
    # it stays verifiable; the five error-based criteria do not.
    check("no_metrics_v2", {1: "UNVERIFIABLE", 2: "UNVERIFIABLE", 3: "UNVERIFIABLE",
                            4: "UNVERIFIABLE", 5: "PASS", 6: "UNVERIFIABLE"})

    # The in-bunch/far-field split is visible in the extras.
    data = _synthetic_data("in_bunch_fail_far_ok")
    evaluation = evaluate_criteria(data, (2, 4, 8))
    try:
        crit2 = evaluation["criteria"][1]
        _assert(crit2["id"] == 2 and crit2["verdict"] == "FAIL",
                "criterion 2 does not fail on the in-bunch metric")
        row = next(r for r in evaluation["extra"]["per_transverse_component"]
                   if r["case"] == "two_cluster" and r["rule"] == "auto"
                   and r["K"] == 4)
        _assert(row["regions"]["in_bunch"]["Ey"]["max_rel"] > 0.1,
                "in-bunch Ey max missing from the transverse table")
        _assert(row["regions"]["far_field"]["Ey"]["max_rel"] < 1e-8,
                "far-field Ey max missing from the transverse table")
    except (AssertionError, StopIteration) as exc:
        failures.append(f"in_bunch_fail_far_ok extras: {exc}")

    # No metrics_v2: UNVERIFIABLE with the key, not a crash.
    evaluation = evaluate_criteria(_synthetic_data("no_metrics_v2"), (2, 4, 8))
    try:
        error_based = [item for item in evaluation["criteria"] if item["id"] != 5]
        _assert(all(item["verdict"] == "UNVERIFIABLE" for item in error_based),
                "no_metrics_v2 did not mark the error-based criteria UNVERIFIABLE")
        joined = " ".join(" ".join(item["missing"]) for item in evaluation["criteria"])
        _assert("metrics_v2" in joined,
                "no_metrics_v2 verdicts do not name the missing key")
        _assert(evaluation["extra"]["per_transverse_component"],
                "no_metrics_v2 dropped the per-component table")
    except AssertionError as exc:
        failures.append(f"no_metrics_v2: {exc}")

    # File-driven round trip: manifest + per-run JSONs -> markdown + criteria.
    with tempfile.TemporaryDirectory(prefix="velocity-summary-selftest-") as tmp_name:
        tmp = Path(tmp_name)
        try:
            manifest_path = _write_synthetic_tree(tmp, "pass")
            manifest, loaded = load_inputs(manifest_path)
            evaluation = evaluate_criteria(loaded, tuple(manifest.get("bins", [2, 4, 8])))
            markdown = render_markdown(manifest, loaded, evaluation, manifest_path)
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
            _assert("Per-transverse-component" in markdown,
                    "markdown lacks the per-transverse-component section")
            _assert(all(item["verdict"] == "PASS" for item in reloaded["criteria"]),
                    "file-driven pass profile did not give six PASS verdicts")
        except (AssertionError, OSError, ValueError) as exc:
            failures.append(f"file round trip: {exc}")

    if failures:
        print("SELF-TEST FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("SELF-TEST OK: 5 profiles, 6 metrics_v2 criteria each, "
          "in-bunch/far-field split, boundary case without metrics_v2, "
          "file round trip")
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
        "geometry": manifest.get("geometry"),
        "profile": manifest.get("profile"),
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
