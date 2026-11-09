#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""End-to-end wall-time harness for picBunch, including the openPMD writer.

The harness runs the shipped CLI once per warm-up/repeat, measures wall time
around the whole process (open, compute, write, finalize, exit), parses the
stage timings the CLI prints, verifies the written series with openPMD's Python
bindings, and writes raw and summary CSV files.  See README.md for the schema.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGE_NAMES = ["reader", "deposit", "kernel", "fft", "transpose", "readout", "writer", "total"]
STAGE_HEADERS = [name + "_ms" for name in STAGE_NAMES]

RUN_FIELDS = [
    "case",
    "backend",
    "io_backend",
    "engine",
    "backend_match",
    "repeats",
    "repeat",
    "phase",
    "ranks",
    "host",
    "binary_sha256",
    "source_commit",
    "compiled_backend",
    "cuda_version",
    "adios2_version",
    "openpmd_version",
    "env_file",
    "wall_ms",
    *STAGE_HEADERS,
    "near_shell_ms",
    "setup_ms",
    "output_bytes",
    "status",
    "notes",
]

SUMMARY_FIELDS = [
    "case",
    "backend",
    "io_backend",
    "engine",
    "backend_match",
    "repeats",
    "host",
    "binary_sha256",
    "source_commit",
    "compiled_backend",
    "cuda_version",
    "adios2_version",
    "openpmd_version",
    "env_file",
    "wall_median_ms",
    "wall_min_ms",
    "wall_max_ms",
    "wall_stddev_ms",
    *[name + "_median_ms" for name in STAGE_NAMES],
    "near_shell_median_ms",
    "setup_median_ms",
    "status",
    "notes",
]

PLAN_FIELDS = ["case", "backend", "phase", "index", "estimate_s", "input", "input_present"]


def eprint(*args):
    print(*args, file=sys.stderr)


def env_wrap(cmd, env_file):
    """Run *cmd* after sourcing an optional environment file by path."""
    if not env_file:
        return cmd
    script = 'set -e; if [ -n "$1" ]; then source "$1"; fi; shift; exec "$@"'
    return ["bash", "-c", script, "_", str(env_file), *cmd]


def run_wrapped(cmd, env_file, timeout=None, env=None):
    return subprocess.run(
        env_wrap(cmd, env_file),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def sha256_file(path):
    if not path or not Path(path).is_file():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_environment(env_file, python):
    script = r'''
set -e
if [ -n "$1" ]; then source "$1"; fi
printf 'ADIOS2_PREFIX=%s\n' "${ADIOS2_PREFIX:-}"
printf 'OPENPMD_PREFIX=%s\n' "${OPENPMD_PREFIX:-}"
printf 'HDF5_ROOT=%s\n' "${HDF5_ROOT:-}"
if command -v bpls >/dev/null 2>&1; then
    printf 'BPLS=%s\n' "$(bpls --version 2>&1 | head -1)"
fi
if command -v nvcc >/dev/null 2>&1; then
    printf 'NVCC=%s\n' "$(nvcc --version 2>&1 | grep -o 'release [0-9.]*' | head -1 | cut -d' ' -f2)"
fi
if command -v mpirun >/dev/null 2>&1; then
    printf 'MPIRUN=%s\n' "$(mpirun --version 2>&1 | head -1)"
fi
'''
    info = {}
    try:
        result = run_wrapped(["bash", "-c", script, "_", str(env_file) if env_file else ""], None)
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
    except Exception as exc:
        info["probe_error"] = str(exc)

    try:
        result = run_wrapped(
            [python, "-c", "import openpmd_api; print(openpmd_api.__version__)"],
            env_file,
            timeout=60,
        )
        if result.returncode == 0:
            info["OPENPMD"] = result.stdout.strip().splitlines()[-1]
    except Exception:
        pass
    return info


def probe_binary(binary, env_file):
    info = {}
    if not binary or not Path(binary).is_file():
        return info
    try:
        result = run_wrapped([binary, "--help"], env_file, timeout=60)
        match = re.search(r"Compiled backend:\s*(\w+)", result.stdout + result.stderr)
        if match:
            info["compiled_backend"] = match.group(1)
    except Exception:
        pass
    return info


def parse_stages(text):
    stages = {}
    pattern = re.compile(
        r"^\s*(reader|deposit|kernel|fft|transpose|readout|writer|total)\s+"
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            stages[match.group(1)] = float(match.group(2))
    return stages


def parse_near_shell(text):
    match = re.search(
        r"near shell wall \[rank 0\]:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*ms",
        text,
    )
    return float(match.group(1)) if match else ""


def directory_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def inspect_output(path, expected_backend, python, env_file):
    info = {"io_backend": "", "engine": "", "backend_match": "", "verify": ""}
    path = Path(path)
    if not path.exists():
        return info
    if path.is_dir():
        names = set(os.listdir(path))
        if {"md.0", "mmd.0"} <= names:
            info["engine"] = "BP5"
        elif "md.0" in names:
            info["engine"] = "BP4-or-other-ADIOS2"
    try:
        result = run_wrapped([python, str(HERE / "inspect_series.py"), str(path)], env_file, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            info["verify"] = payload.get("verify", "")
            info["io_backend"] = payload.get("backend", "")
            info["iterations"] = payload.get("iterations", "")
            info["meshes"] = ",".join(payload.get("meshes", []))
            if payload.get("error"):
                info["verify_error"] = payload["error"]
        else:
            info["verify"] = "inspector-failed"
    except Exception as exc:
        info["verify"] = "inspector-error"
        info["verify_error"] = str(exc)
    expected = {"bp5": "ADIOS2", "h5": "HDF5"}[expected_backend]
    if info["io_backend"]:
        info["backend_match"] = "yes" if info["io_backend"] == expected else "no"
    return info


def number(value):
    if isinstance(value, float):
        return "%.12g" % value
    return str(value)


def triple(value):
    if isinstance(value, (list, tuple)):
        return ",".join(number(v) for v in value)
    return str(value)


def build_command(case, binary, input_dir, output):
    command = [
        binary,
        "--input",
        str(Path(input_dir) / case["input"]),
        "--cell-size",
        triple(case["cell_size"]),
        "--grid",
        triple(case["grid"]),
        "--origin",
        triple(case["origin"]),
        "--velocity-bins",
        str(case.get("velocity_bins", 1)),
        "--tolerance",
        number(case.get("tolerance", 1e-3)),
        "--shape",
        str(case.get("shape", "tsc")),
        "--output",
        str(output),
    ]
    near = case.get("near_shell", "none")
    if near != "none":
        command += [
            "--near-shell",
            near,
            "--near-shell-radius",
            number(case["near_shell_radius"]),
            "--near-shell-tile",
            str(case.get("near_shell_tile", 4)),
            "--near-shell-fast-order",
            str(case.get("near_shell_fast_order", 5)),
        ]
    return command


def expand_cases(cases_file, selection, overrides):
    data = json.loads(Path(cases_file).read_text())
    campaigns = data["campaigns"]
    cases = data["cases"]
    if selection in (None, "", "all"):
        names = campaigns["e2e"] + campaigns["cost"]
    elif selection == "tiny":
        names = campaigns["tiny"]
    else:
        names = []
        for token in selection.split(","):
            token = token.strip()
            if not token:
                continue
            if token in campaigns:
                names.extend(campaigns[token])
            elif token in cases:
                names.append(token)
            else:
                raise SystemExit(f"unknown case or campaign: {token}")
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    resolved = []
    for name in seen:
        case = dict(cases[name])
        case["_name"] = name
        if overrides.get("ranks"):
            case["ranks"] = overrides["ranks"]
        resolved.append(case)
    return resolved


def plan_rows(cases, backends, warmup, repeat, input_dir):
    rows = []
    for case in cases:
        for backend in backends:
            for index in range(1, warmup + 1):
                rows.append(
                    {
                        "case": case["_name"],
                        "backend": backend,
                        "phase": "warmup",
                        "index": index,
                        "estimate_s": case["estimate_s"],
                        "input": case["input"],
                        "input_present": "yes"
                        if (Path(input_dir) / case["input"]).is_file()
                        else "no",
                    }
                )
            for index in range(1, repeat + 1):
                rows.append(
                    {
                        "case": case["_name"],
                        "backend": backend,
                        "phase": "repeat",
                        "index": index,
                        "estimate_s": case["estimate_s"],
                        "input": case["input"],
                        "input_present": "yes"
                        if (Path(input_dir) / case["input"]).is_file()
                        else "no",
                    }
                )
    return rows


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def summarize(raw_rows, planned_repeats):
    groups = {}
    for row in raw_rows:
        if row.get("phase") != "repeat":
            continue
        groups.setdefault((row["case"], row["backend"]), []).append(row)
    summary = []
    for (case_name, backend), rows in groups.items():
        ok = [row for row in rows if row["status"] == "ok"]
        base = dict(rows[0])
        base.pop("repeat", None)
        base.pop("phase", None)
        base["repeats"] = planned_repeats
        base["status"] = "ok" if len(ok) == planned_repeats else ("partial" if ok else "failed")
        if ok:
            for stage in STAGE_NAMES:
                values = [float(row[stage + "_ms"]) for row in ok if row[stage + "_ms"] != ""]
                base[stage + "_median_ms"] = "%.6f" % statistics.median(values) if values else ""
            near = [float(row["near_shell_ms"]) for row in ok if row["near_shell_ms"] != ""]
            base["near_shell_median_ms"] = "%.6f" % statistics.median(near) if near else ""
            setup = [float(row["setup_ms"]) for row in ok if row["setup_ms"] != ""]
            base["setup_median_ms"] = "%.6f" % statistics.median(setup) if setup else ""
            walls = [float(row["wall_ms"]) for row in ok]
            base["wall_median_ms"] = "%.6f" % statistics.median(walls)
            base["wall_min_ms"] = "%.6f" % min(walls)
            base["wall_max_ms"] = "%.6f" % max(walls)
            base["wall_stddev_ms"] = "%.6f" % statistics.stdev(walls) if len(walls) > 1 else ""
        summary.append(base)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="picBunch executable to measure")
    parser.add_argument("--input-dir", default=".", help="directory holding the campaign input files")
    parser.add_argument("--out", default="e2e-out", help="output directory for CSV, logs and series")
    parser.add_argument("--cases", default="all", help="campaign name, comma-separated cases, 'all' or 'tiny'")
    parser.add_argument("--cases-file", default=str(HERE / "cases.json"))
    parser.add_argument("--backend", choices=["bp5", "h5", "both"], default="bp5")
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats after warm-up")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs")
    parser.add_argument("--ranks", type=int, help="override the MPI rank count of every case")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES for single-rank runs")
    parser.add_argument("--env-file", help="environment file to source by path before every run")
    parser.add_argument("--python", default="python3", help="interpreter that can import openpmd_api")
    parser.add_argument("--source-dir", help="picBunch source tree (records the git commit)")
    parser.add_argument("--build-script", help="optional script run once before measuring")
    parser.add_argument("--timeout", type=float, default=7200.0, help="per-run timeout in seconds")
    parser.add_argument("--keep-output", action="store_true", help="keep the openPMD series after inspection")
    parser.add_argument("--plan", action="store_true", help="print the run plan and exit")
    parser.add_argument("--tiny", action="store_true", help="one tiny GPU0 run: --cases tiny --repeat 1 --warmup 0")
    args = parser.parse_args()

    if args.tiny:
        args.cases = "tiny"
        args.repeat = 1
        args.warmup = 0
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.backend == "both":
        backends = ["bp5", "h5"]
    else:
        backends = [args.backend]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = expand_cases(args.cases_file, args.cases, {"ranks": args.ranks})

    env_info = probe_environment(args.env_file, args.python)
    host = socket.gethostname()
    source_commit = ""
    if args.source_dir:
        try:
            source_commit = subprocess.run(
                ["git", "-C", args.source_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            pass

    manifest = {
        "host": host,
        "utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(args.binary) if args.binary else "",
        "binary_sha256": sha256_file(args.binary),
        "source_commit": source_commit,
        "env_file": str(args.env_file) if args.env_file else "",
        "environment": env_info,
        "cases": [case["_name"] for case in cases],
        "backends": backends,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }

    if args.plan:
        rows = plan_rows(cases, backends, args.warmup, args.repeat, args.input_dir)
        write_csv(out_dir / "plan.csv", PLAN_FIELDS, rows)
        total = sum(float(row["estimate_s"]) for row in rows)
        print(f"# plan for {len(cases)} case(s), backends={','.join(backends)}")
        print(f"# host={host}  out={out_dir}")
        print(f"{'case':<22} {'backend':<8} {'phase':<8} {'idx':>3} {'est_s':>10} input")
        for row in rows:
            print(
                f"{row['case']:<22} {row['backend']:<8} {row['phase']:<8} {row['index']:>3} "
                f"{float(row['estimate_s']):>10.2f} {row['input']} "
                f"({row['input_present']})"
            )
        print(f"# estimated total wall time: {total:.1f} s ({total / 3600.0:.2f} h)")
        missing = sorted({row["input"] for row in rows if row["input_present"] == "no"})
        if missing:
            print("# missing inputs (generate with tools/e2e/gen_inputs.py or point --input-dir):")
            for name in missing:
                print(f"#   {name}")
        (out_dir / "plan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0

    if not args.binary:
        raise SystemExit("--binary is required unless --plan is used")
    if not Path(args.binary).is_file():
        raise SystemExit(f"binary not found: {args.binary}")

    if args.build_script:
        eprint(f"building via {args.build_script}")
        build = run_wrapped(["bash", str(args.build_script)], args.env_file, timeout=args.timeout)
        (out_dir / "build.log").write_text(build.stdout + build.stderr)
        if build.returncode != 0:
            eprint(build.stdout[-4000:])
            eprint(build.stderr[-4000:])
            raise SystemExit("build script failed")

    binary_info = probe_binary(args.binary, args.env_file)
    manifest["binary_sha256"] = sha256_file(args.binary)
    manifest["compiled_backend"] = binary_info.get("compiled_backend", "")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    child_env = os.environ.copy()
    child_env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    child_env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    if args.device is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    common = {
        "host": host,
        "binary_sha256": manifest["binary_sha256"],
        "source_commit": source_commit,
        "compiled_backend": binary_info.get("compiled_backend", ""),
        "cuda_version": env_info.get("NVCC", ""),
        "adios2_version": env_info.get("BPLS", ""),
        "openpmd_version": env_info.get("OPENPMD", ""),
        "env_file": str(args.env_file) if args.env_file else "",
        "repeats": args.repeat,
    }

    raw_rows = []
    for case in cases:
        name = case["_name"]
        for backend in backends:
            extension = ".bp" if backend == "bp5" else ".h5"
            series_dir = out_dir / "series" / name / backend
            log_dir = out_dir / "logs" / name / backend
            series_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            for phase, count in (("warmup", args.warmup), ("repeat", args.repeat)):
                for index in range(1, count + 1):
                    output = series_dir / f"{phase}-{index}{extension}"
                    if output.is_dir():
                        shutil.rmtree(output)
                    elif output.exists():
                        output.unlink()
                    command = build_command(case, args.binary, args.input_dir, output)
                    ranks = int(case.get("ranks", 1))
                    if ranks > 1:
                        command = [
                            "mpirun",
                            "-np",
                            str(ranks),
                            "--bind-to",
                            "none",
                            *command,
                        ]
                    eprint(f"[{name}/{backend}/{phase}-{index}] starting")
                    started = time.monotonic()
                    timed_out = False
                    try:
                        result = run_wrapped(
                            command,
                            args.env_file,
                            timeout=args.timeout,
                            env=child_env,
                        )
                    except subprocess.TimeoutExpired as exc:
                        timed_out = True
                        result = subprocess.CompletedProcess(
                            command,
                            returncode=124,
                            stdout=(exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                            stderr=(exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                        )
                    wall_ms = (time.monotonic() - started) * 1000.0
                    (log_dir / f"{phase}-{index}.stdout.log").write_text(result.stdout)
                    (log_dir / f"{phase}-{index}.stderr.log").write_text(result.stderr)
                    stages = parse_stages(result.stdout)
                    inspect = inspect_output(output, backend, args.python, args.env_file)
                    status = "ok" if result.returncode == 0 else ("timeout" if timed_out else "failed")
                    notes = " ".join(case.get("notes", "").split())
                    if status != "ok":
                        tail = (result.stderr.strip().splitlines() or [""])[-1]
                        notes = (notes + " | error: " + tail).strip(" |")
                    if inspect.get("verify") not in ("openpmd", ""):
                        notes = (notes + " | verify: " + str(inspect.get("verify"))).strip(" |")
                    if inspect.get("verify_error"):
                        notes = (notes + " | verify-error: " + str(inspect["verify_error"])).strip(" |")
                    row = {
                        "case": name,
                        "backend": backend,
                        "io_backend": inspect.get("io_backend", ""),
                        "engine": inspect.get("engine", ""),
                        "backend_match": inspect.get("backend_match", ""),
                        "repeats": args.repeat,
                        "repeat": index,
                        "phase": phase,
                        "ranks": ranks,
                        **common,
                        "wall_ms": "%.6f" % wall_ms,
                        "near_shell_ms": parse_near_shell(result.stderr),
                        "setup_ms": "",
                        "output_bytes": directory_bytes(output),
                        "status": status,
                        "notes": notes,
                    }
                    for stage in STAGE_NAMES:
                        row[stage + "_ms"] = "%.6f" % stages[stage] if stage in stages else ""
                    if "total" in stages:
                        row["setup_ms"] = "%.6f" % (wall_ms - stages["total"])
                    raw_rows.append(row)
                    if phase == "repeat":
                        eprint(
                            f"[{name}/{backend}] {index}/{count} wall={wall_ms:.1f} ms "
                            f"total={row['total_ms'] or '?'} writer={row['writer_ms'] or '?'} "
                            f"{status}"
                        )
                    if not args.keep_output and output.exists():
                        if output.is_dir():
                            shutil.rmtree(output)
                        else:
                            output.unlink()

    write_csv(out_dir / "runs.csv", RUN_FIELDS, raw_rows)
    summary = summarize(raw_rows, args.repeat)
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
    print(f"wrote {out_dir / 'runs.csv'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    failed = [row for row in raw_rows if row["status"] != "ok"]
    if failed:
        eprint(f"{len(failed)} run(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
