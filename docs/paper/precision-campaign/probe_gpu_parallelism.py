#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Short scheduler probe: sample ``nvidia-smi`` while four cases run.

The probe starts a 5 Hz sampler, runs ``run_campaign.py`` over a set of
mesh-heavy cases in a *new* attempt directory, stops the sampler and writes
``gpu_parallelism.json`` next to the campaign root.  It answers one question:
do the four GPU workers actually overlap, and how much SM utilisation do the
mesh jobs reach?

Usage::

    probe_gpu_parallelism.py --root <new-attempt-root> --bin <build>/picBunch \\
        --driver <campaign>/bin/exact_field_driver \\
        --harness-commit <sha> [--cases line-gamma,line-w4b,multivel,hires-g10]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_campaign.py")
DEFAULT_CASES = "line-gamma,line-w4b,multivel,hires-g10"


def sample_once():
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            apps.strip().replace("\n", "|"),
            gpus.strip().replace("\n", "|"))


def sampler(path, stop):
    with open(path, "w") as handle:
        while not stop.is_set():
            handle.write("%s;%s;%s\n" % sample_once())
            handle.flush()
            time.sleep(0.2)


def summarize(path):
    uuid_index = {}
    counts = {}
    per_gpu = {}
    samples = 0
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(";")
            if len(parts) != 3:
                continue
            samples += 1
            for item in parts[2].split("|"):
                fields = [f.strip() for f in item.split(",")]
                if len(fields) >= 4 and fields[0].isdigit():
                    uuid_index[fields[1]] = int(fields[0])
                    util = int(fields[2].rstrip(" %") or 0)
                    mem = int(fields[3].split()[0] or 0)
                    per_gpu.setdefault(int(fields[0]), []).append((util, mem))
            active = {uuid_index.get(item.split(",")[0].strip())
                      for item in parts[1].split("|") if item.strip()}
            active.discard(None)
            counts[len(active)] = counts.get(len(active), 0) + 1
    summary = {
        "samples": samples,
        "max_simultaneous_gpus": max(counts) if counts else 0,
        "simultaneous_histogram": {str(k): v for k, v in sorted(counts.items())},
        "per_gpu": {},
    }
    for index, rows in sorted(per_gpu.items()):
        utils = [u for u, _ in rows]
        active = [(u, m) for u, m in rows if m > 0]
        busy = [(u, m) for u, m in rows if u >= 10]
        summary["per_gpu"][str(index)] = {
            "peak_util_pct": max(utils) if utils else 0,
            "samples_active": len(active),
            "active_seconds_estimate": 0.2 * len(active),
            "samples_util_ge_10pct": len(busy),
            "seconds_util_ge_10pct_estimate": 0.2 * len(busy),
            "mean_util_active_pct": (
                sum(u for u, _ in active) / len(active) if active else 0.0),
            "peak_memory_mib": max((m for _, m in rows), default=0),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--driver", default="")
    parser.add_argument("--harness-commit", default="")
    parser.add_argument("--build-commit", default="00fb8f7fb22a53748636c70e32bbdb8dd85c23c4")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--ref-workers", type=int, default=8)
    parser.add_argument("--ref-threads", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)
    log = os.path.join(args.root, "gpu_samples.log")
    stop = threading.Event()
    thread = threading.Thread(target=sampler, args=(log, stop), daemon=True)
    thread.start()
    cmd = [sys.executable, RUNNER, "--root", args.root, "--bin", args.bin,
           "--only", args.cases,
           "--git-commit", args.harness_commit,
           "--build-commit", args.build_commit,
           "--ref-workers", str(args.ref_workers),
           "--ref-threads", str(args.ref_threads)]
    if args.driver:
        cmd += ["--driver", args.driver]
    started = time.time()
    rc = subprocess.call(cmd)
    stop.set()
    thread.join()
    summary = summarize(log)
    summary["wall_s"] = time.time() - started
    summary["cases"] = args.cases.split(",")
    campaign = os.path.join(args.root, "prov", "campaign_summary.json")
    if os.path.isfile(campaign):
        with open(campaign) as handle:
            run = json.load(handle)
        summary["case_slots"] = run.get("case_slots")
        summary["case_wall_s"] = {
            key: value.get("wall_s") for key, value in run.get("cases", {}).items()}
    with open(os.path.join(args.root, "gpu_parallelism.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
