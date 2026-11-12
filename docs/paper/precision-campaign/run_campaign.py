#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Driver for the picBunch precision measurement campaign.

The driver enumerates every precision case in ``cases.json`` (w4b, gamma-error
distribution, hires gp sweep, validation slices, large-domain slices and
multivel; the scaling/performance figures are deliberately absent) and records
one provenance record per case.

Execution uses two independent pools.  Four GPU workers each own one
``CUDA_VISIBLE_DEVICES`` slot and pull mesh jobs from a shared work queue, so
every mesh that is ready runs as soon as a GPU is free.  A separate pool runs
the CPU (OpenMP) reference jobs for different cases concurrently; references
and meshes of the same case start together, because the reference has no
dependency on the mesh.  A case is reduced once all of its mesh and reference
jobs have finished.  The old launcher bound one case at a time to each slot and
ran that slot's mesh and reference sequentially, which left the GPU idle for
the whole reference.

Usage on the H100 node::

    ./run_campaign.py --root /home/ubuntu/picbunch-work-20261112/campaign \\
        --bin <build>/picBunch --estimate
    ./run_campaign.py --root ... --bin ... --pilot --only var1b-1mrad,hires-g10
    ./run_campaign.py --root ... --bin ...              # full campaign

Every mesh runs in one process pinned with ``CUDA_VISIBLE_DEVICES=<slot>``
(picBunch has no ``--device`` flag; the CUDA runtime then exposes exactly one
device per process).  The default reference pool is 8 jobs x 16 OpenMP threads
(128 of the 144 cores), which leaves headroom for the mesh processes.

A case's estimated wall is ``mesh_s + ref_s + reduce_s`` from ``cases.json``.
The reference estimate is recalibrated from ``--ref-rate`` (seconds per
sample-point times source, the upper-bound cost model).  The pilot run measures
the real mesh and reference times and writes ``pilot_report.json`` with the
measured/estimated ratio.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CASES_JSON = os.path.join(HERE, "cases.json")
GEN_INPUTS = os.path.join(HERE, "gen_inputs.py")
DRIVER_SRC = os.path.join(HERE, "reference", "exact_field_driver.cpp")
DRIVER_INCLUDE = os.path.join(HERE, "reference", "refinc")
YEE = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
       (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
COMP = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_cases():
    with open(CASES_JSON) as handle:
        return json.load(handle)


def n_sources(input_h5):
    import h5py
    with h5py.File(input_h5, "r") as handle:
        return int(handle["data/0/particles/e/position/x"].shape[0])


def plane_size(plane, n):
    return n[0] * n[1] if plane == "xy" else n[1] * n[2]


def reference_cost(case, root, rate):
    """Upper-bound reference wall: sample points x sources x rate."""
    total = 0.0
    for run in case["runs"]:
        ref = run.get("reference") or case.get("reference")
        if not ref:
            continue
        cell = run["mesh"]["cell"]
        n = run["mesh"]["grid"]
        if ref["mode"] == "rings":
            points = len(ref["rings"]) * len(ref["axes"]) * 6
        else:
            points = sum(plane_size(p, n) for p in ref["planes"]) * 6
        sources = n_sources(os.path.join(root, "inputs", run["input"] + ".h5"))
        total += points * sources * rate
    return total


def case_cost(case, root, rate):
    est = case.get("est", {})
    mesh_s = float(est.get("mesh_s", 0.0))
    reduce_s = float(est.get("reduce_s", 0.0))
    ref_s = float(est.get("ref_s", 0.0))
    if os.path.isdir(os.path.join(root, "inputs")):
        try:
            ref_s = max(ref_s, reference_cost(case, root, rate))
        except (OSError, KeyError):
            pass
    return mesh_s + ref_s + reduce_s


def schedule(cases, root, rate, gpu_count=4):
    """Longest-processing-time-first assignment to identical GPU slots."""
    order = sorted(cases, key=lambda c: case_cost(c, root, rate), reverse=True)
    loads = [0.0] * gpu_count
    slots = [[] for _ in range(gpu_count)]
    for case in order:
        index = min(range(gpu_count), key=lambda i: (loads[i], i))
        slots[index].append(case)
        loads[index] += case_cost(case, root, rate)
    return slots, loads


def _directions():
    """Axis, face-diagonal and cube-diagonal unit directions."""
    dirs = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            vec = [0.0, 0.0, 0.0]
            vec[axis] = sign
            dirs.append(tuple(vec))
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    vec = [0.0, 0.0, 0.0]
                    vec[i] = si / math.sqrt(2.0)
                    vec[j] = sj / math.sqrt(2.0)
                    dirs.append(tuple(vec))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                dirs.append((sx / math.sqrt(3.0), sy / math.sqrt(3.0),
                             sz / math.sqrt(3.0)))
    return dirs


def reference_points(run, ref):
    """Return ``(points_cell, comp, d_cells, ijk)`` for one reference stage."""
    cell = [float(v) for v in run["mesh"]["cell"]]
    n = [int(v) for v in run["mesh"]["grid"]]
    origin = [float(v) for v in run["mesh"].get("origin", [0.0, 0.0, 0.0])]
    points = []
    comps = []
    d_cells = []
    ijk = []
    if ref["mode"] == "rings":
        lattice = float(ref["lattice_cell_m"])
        centre = [float(v) for v in ref["centre_cells"]]
        source = [centre[i] * lattice for i in range(3)]
        source_cell = [source[i] / cell[i] for i in range(3)]
        base = 1e-7
        seen = set()
        for ring in ref["rings"]:
            distance = ring * base
            for comp in range(6):
                for direction in _directions():
                    target = [distance * direction[a] for a in range(3)]
                    delta = [int(round(target[a] / cell[a] - YEE[comp][a]))
                             for a in range(3)]
                    node = [source_cell[a] + delta[a] + YEE[comp][a]
                            for a in range(3)]
                    if any(node[a] < 0.0 or node[a] > n[a] - 1 for a in range(3)):
                        continue
                    key = (comp, tuple(delta))
                    if key in seen:
                        continue
                    seen.add(key)
                    actual = math.sqrt(sum((delta[a] + YEE[comp][a]) ** 2
                                           * cell[a] ** 2 for a in range(3))) / base
                    if abs(actual - ring) > 0.35:
                        continue
                    points.append(node)
                    comps.append(comp)
                    d_cells.append(actual)
                    ijk.append((-1, -1, -1))
    else:
        for plane, fixed in ref["planes"].items():
            if plane == "xy":
                z = int(fixed)
                for j in range(n[1]):
                    for i in range(n[0]):
                        base_ijk = (i, j, z)
                        for comp in range(6):
                            points.append([base_ijk[a] + YEE[comp][a] for a in range(3)])
                            comps.append(comp)
                            d_cells.append(-1.0)
                            ijk.append(base_ijk)
            else:
                x = int(fixed)
                for k in range(n[2]):
                    for j in range(n[1]):
                        base_ijk = (x, j, k)
                        for comp in range(6):
                            points.append([base_ijk[a] + YEE[comp][a] for a in range(3)])
                            comps.append(comp)
                            d_cells.append(-1.0)
                            ijk.append(base_ijk)
    return points, comps, d_cells, ijk


def ensure_driver(path):
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["g++", "-std=c++20", "-O3", "-march=native", "-fopenmp",
           "-I", DRIVER_INCLUDE, DRIVER_SRC, "-o", path]
    subprocess.run(cmd, check=True)
    return path


def build_reference(root, case, run, driver, order, max_level, tol, threads):
    input_h5 = os.path.join(root, "inputs", run["input"] + ".h5")
    sources_bin = os.path.join(root, "inputs", run["input"] + ".sources.bin")
    cell = [float(v) for v in run["mesh"]["cell"]]
    points, comps, d_cells, ijk = reference_points(run, run["reference"])
    stage = os.path.join(root, "ref", case["id"] + "__" + run["name"])
    os.makedirs(os.path.dirname(stage), exist_ok=True)
    points_bin = stage + ".points.bin"
    out_bin = stage + ".bin"
    arr = [(points[i][0], points[i][1], points[i][2], comps[i])
           for i in range(len(points))]
    with open(points_bin, "wb") as handle:
        import numpy as np
        np.asarray(arr, dtype=np.float64).tofile(handle)
    cmd = [driver, "--samples", sources_bin, points_bin, out_bin,
           str(tol), str(cell[0]), str(cell[1]), str(cell[2]),
           str(max_level), str(order)]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import numpy as np
    vals = np.fromfile(out_bin, dtype=np.float64)
    np.savez(stage + ".npz",
             point_m=np.asarray([(points[i][0] * cell[0],
                                  points[i][1] * cell[1],
                                  points[i][2] * cell[2]) for i in range(len(points))]),
             comp=np.asarray(comps, dtype=np.int64),
             d_cells=np.asarray(d_cells, dtype=np.float64),
             ijk=np.asarray(ijk, dtype=np.int64),
             vals=vals, order=np.asarray(order), n=np.asarray(run["mesh"]["grid"]),
             cell=np.asarray(cell))
    return stage + ".npz", cmd


def mesh_command(binary, root, run, out_path):
    mesh = run["mesh"]
    cmd = [binary, "--input", os.path.join(root, "inputs", run["input"] + ".h5"),
           "--cell-size", ",".join("%g" % v for v in mesh["cell"]),
           "--grid", ",".join(str(int(v)) for v in mesh["grid"]),
           "--origin", ",".join("%g" % v for v in mesh.get("origin", [0, 0, 0])),
           "--velocity-bins", str(int(mesh.get("bins", 1))),
           "--tolerance", "1e-3", "--shape", "tsc"]
    near = mesh.get("near_shell")
    if near:
        cmd += ["--near-shell", near.get("mode", "fast"),
                "--near-shell-radius", str(near.get("radius", 3)),
                "--near-shell-tile", str(near.get("tile", 4)),
                "--near-shell-fast-order", str(near.get("fast_order", 5))]
    cmd += list(mesh.get("extra", []))
    cmd += ["--output", out_path]
    return cmd


def resolve(root, path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(root, path)


def input_centroid_m(root, run):
    import h5py
    import numpy as np
    with h5py.File(os.path.join(root, "inputs", run["input"] + ".h5"), "r") as handle:
        species = handle["data/0/particles/e"]
        pos = []
        for axis in "xyz":
            node = species["position"][axis]
            pos.append(np.asarray(node[:], dtype=float)
                       * float(node.attrs.get("unitSI", 1.0)))
        weight = species["weighting"]["value"]
        w = np.asarray(weight[:], dtype=float) * float(weight.attrs.get("unitSI", 1.0))
    pos = np.stack(pos, axis=1)
    return (pos * w[:, None]).sum(axis=0) / w.sum()


def slice_reduce_spec(case, root, run, pic_path, ref_path, host, picbunch):
    spec = dict(case["reduce"])
    mesh = run["mesh"]
    spec.update({
        "pic": pic_path, "ref": ref_path,
        "n": [int(v) for v in mesh["grid"]],
        "cell": [float(v) for v in mesh["cell"]],
        "origin": [float(v) for v in mesh.get("origin", [0.0, 0.0, 0.0])],
        "input_h5": os.path.join(root, "inputs", run["input"] + ".h5"),
        "host": host, "picbunch": picbunch,
        "plane_indices": {},
        "out": resolve(root, case["reduce"]["out"]),
        "ingest": resolve(root, case["reduce"].get("ingest")),
        "summary": os.path.join(root, "prov", "reduced", case["id"] + ".json"),
    })
    ref = run.get("reference") or case.get("reference")
    for plane, fixed in ref["planes"].items():
        spec["plane_indices"][plane] = int(fixed)
    if case["reduce"]["style"] == "large1":
        spec["centre"] = input_centroid_m(root, run).tolist()
    return spec


def line_reduce_spec(case, root, run_paths, refs):
    spec = dict(case["reduce"])
    mode = spec["mode"]
    if mode in ("w4b", "gamma"):
        units = []
        for unit in spec["units"]:
            run = unit["run"]
            entry = dict(unit)
            entry["pic"] = run_paths[run]
            entry["ref"] = refs[run]
            units.append(entry)
        spec["units"] = units
        spec["out"] = resolve(root, case["reduce"]["out"])
        spec["ingest"] = resolve(root, case["reduce"]["ingest"])
    else:
        spec["fields_e5"] = {b: run_paths["e5-b%d" % b] for b in
                             (1, 4, 16, 64, 256)}
        spec["fields_a5"] = {b: run_paths["a5-b%d" % b] for b in
                             (1, 4, 16, 64, 256)}
        spec["input_e5"] = resolve(root, case["reduce"]["input_e5"])
        spec["input_a5"] = resolve(root, case["reduce"]["input_a5"])
        spec["panel_e5"] = resolve(root, case["reduce"]["panel_e5"])
        spec["panel_a5"] = resolve(root, case["reduce"]["panel_a5"])
        spec["panel_out_e5"] = resolve(root, case["reduce"]["panel_out_e5"])
        spec["panel_out_a5"] = resolve(root, case["reduce"]["panel_out_a5"])
        spec["fields_out"] = resolve(root, case["reduce"]["fields_out"])
        spec["out"] = resolve(root, case["reduce"]["out"])
        spec["ingest_dir"] = os.path.join(root, "reduced", "multivel")
    spec["summary"] = os.path.join(root, "prov", "reduced", case["id"] + ".json")
    return spec


def _ref_of(case, run):
    return run.get("reference") or case.get("reference")


def _mesh_env(args, slot):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot))
    lib = args.lib_path
    if lib:
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"]
                                        if env.get("LD_LIBRARY_PATH") else "")
    return env


def _case_record(st, args, host, picbunch, spec, outputs, reduce_command):
    case = st["case"]
    slots = sorted(st["mesh_slots"])
    if len(slots) == 1:
        slot_field = slots[0]
        binding = "CUDA_VISIBLE_DEVICES=%d" % slots[0]
    else:
        slot_field = slots
        binding = "CUDA_VISIBLE_DEVICES=%s" % ",".join(str(s) for s in slots)
    commands = sorted(st["commands"] + ([reduce_command] if reduce_command else []),
                      key=lambda item: item["stage"])
    refs = st["ref_paths"]
    return {
        "case_id": case["id"], "family": case["family"],
        "status": "failed" if st["failures"] else "done",
        # Both commits are recorded so the binary and the harness can be
        # rebuilt independently; ``git_commit`` is kept as the harness alias.
        "harness_commit": args.git_commit,
        "build_commit": args.build_commit,
        "git_commit": args.git_commit,
        "binary": {"path": args.bin, "sha256": sha256_file(args.bin)},
        "config_sha256": json_sha(case),
        "host": host, "gpu_slot": slot_field, "gpu_binding": binding,
        "start_utc": st["start_utc"], "end_utc": utcnow(),
        "wall_s": time.time() - st["start_mono"],
        "exit_code": 0 if not st["failures"] else 1,
        "failures": st["failures"],
        "commands": commands,
        "outputs": [{"path": os.path.relpath(p, args.root), "sha256": sha256_file(p)}
                    for p in outputs if os.path.isfile(p)],
        "reference": None if not refs else {
            "driver": args.driver,
            "driver_sha256": sha256_file(args.driver),
            "mode": (_ref_of(case, case["runs"][0]) or {}).get("mode"),
            "pilot": bool(args.pilot)},
        "versions": args.versions,
    }


def finalize_case(st, args, lock, host, picbunch):
    """Run the reduction once every mesh and reference job of a case is done."""
    case = st["case"]
    root = args.root
    case_dir = os.path.join(root, "work", case["id"])
    os.makedirs(case_dir, exist_ok=True)
    outputs = [st["mesh_paths"][run["name"]] for run in case["runs"]
               if run["name"] in st["mesh_paths"]]
    outputs += [st["ref_paths"][run["name"]] for run in case["runs"]
                if run["name"] in st["ref_paths"]]
    spec = None
    reduce_command = None
    if st["failures"]:
        st["failures"].append({"stage": "reduce", "exit": 1,
                               "reason": "skipped after mesh/reference failure"})
    else:
        try:
            run_paths = {run["name"]: st["mesh_paths"][run["name"]]
                         for run in case["runs"]}
            refs = {run["name"]: st["ref_paths"][run["name"]]
                    for run in case["runs"] if run["name"] in st["ref_paths"]}
            if case["reduce"]["script"] == "reduce_slices.py":
                run = case["runs"][0]
                spec = slice_reduce_spec(case, root, run, run_paths[run["name"]],
                                         refs[run["name"]], host, picbunch)
            else:
                spec = line_reduce_spec(case, root, run_paths, refs)
            spec_path = os.path.join(case_dir, "reduce_spec.json")
            with open(spec_path, "w") as handle:
                json.dump(spec, handle, indent=1)
            cmd = [sys.executable, os.path.join(HERE, case["reduce"]["script"]),
                   "--spec", spec_path]
            reduce_command = {"stage": "reduce", "argv": cmd, "env": {}}
            with open(os.path.join(args.root, "logs", case["id"] + ".reduce.log"),
                      "w") as log:
                procedure = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if procedure.returncode != 0:
                st["failures"].append({"stage": "reduce",
                                       "exit": procedure.returncode})
        except (OSError, KeyError, subprocess.SubprocessError) as exc:
            st["failures"].append({"stage": "reduce", "error": str(exc)})
    if spec is not None:
        for key in ("out", "ingest"):
            path = spec.get(key)
            if path and os.path.isfile(path):
                outputs.append(path)
        if case["reduce"]["script"] == "reduce_line.py" and case["reduce"]["mode"] == "multivel":
            for key in ("panel_out_e5", "panel_out_a5"):
                path = spec.get(key)
                if path and os.path.isfile(path):
                    outputs.append(path)
            for b in (16, 64):
                for cs in ("e5", "a5"):
                    path = os.path.join(spec["fields_out"],
                                        "multivel_fields_%s_b%d.npz" % (cs, b))
                    if os.path.isfile(path):
                        outputs.append(path)
    record = _case_record(st, args, host, picbunch, spec, outputs, reduce_command)
    os.makedirs(os.path.join(root, "prov", "records"), exist_ok=True)
    record_path = os.path.join(root, "prov", "records", case["id"] + ".json")
    with open(record_path, "w") as handle:
        json.dump(record, handle, indent=1)
    with lock:
        with open(os.path.join(root, "prov", "index.jsonl"), "a") as handle:
            handle.write(json.dumps({"case_id": case["id"],
                                     "status": record["status"],
                                     "record": record_path}) + "\n")
    st["record"] = record
    print("[complete] %-20s %s %s %.1fs" %
          (case["id"], record["status"], record["gpu_binding"],
           record["wall_s"]), flush=True)


def execute(cases, args, log_dir):
    """Run meshes on a four-GPU work queue while references run on a CPU pool.

    The previous launcher bound each case to one slot and ran that slot's mesh
    and reference sequentially, so a GPU idled for every reference.  Here the
    mesh and reference jobs of every case are independent queue items: four GPU
    workers each own one ``CUDA_VISIBLE_DEVICES`` slot and pull the next mesh,
    while a separate pool evaluates references for different cases in parallel.
    A case is reduced once all of its meshes and references have finished.
    """
    root = args.root
    os.makedirs(log_dir, exist_ok=True)
    host = socket.gethostname()
    picbunch = "%s (sha256 %s)" % (args.bin, sha256_file(args.bin))
    lock = threading.Lock()
    states = {}
    run_lookup = {}
    mesh_jobs = queue.Queue()
    ref_jobs = queue.Queue()
    for case in cases:
        state = {"case": case, "pending": 0, "mesh_paths": {}, "ref_paths": {},
                 "mesh_slots": set(), "commands": [], "failures": [],
                 "start_utc": None, "start_mono": None,
                 "reduce_started": False, "record": None}
        for run in case["runs"]:
            run_lookup[(case["id"], run["name"])] = (case, run)
            state["pending"] += 1
            mesh_jobs.put((case["id"], run["name"]))
            if _ref_of(case, run):
                state["pending"] += 1
                ref_jobs.put((case["id"], run["name"]))
        states[case["id"]] = state
    reduce_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    reduce_futures = []

    def note_start(state, start_utc, start_mono):
        if state["start_mono"] is None or start_mono < state["start_mono"]:
            state["start_utc"] = start_utc
            state["start_mono"] = start_mono

    def maybe_reduce(case_id):
        state = states[case_id]
        if state["pending"] == 0 and not state["reduce_started"]:
            state["reduce_started"] = True
            reduce_futures.append(reduce_pool.submit(
                finalize_case, state, args, lock, host, picbunch))

    def finish(case_id, kind, run_name, path, slot, start_utc, start_mono,
               error, command):
        with lock:
            state = states[case_id]
            note_start(state, start_utc, start_mono)
            state["commands"].append(command)
            if slot is not None:
                state["mesh_slots"].add(slot)
            if kind == "mesh":
                state["mesh_paths"][run_name] = path
            else:
                state["ref_paths"][run_name] = path
            if error is not None:
                state["failures"].append(error)
            state["pending"] -= 1
            maybe_reduce(case_id)

    def gpu_worker(slot):
        while True:
            item = mesh_jobs.get()
            if item is None:
                return
            case_id, run_name = item
            case, run = run_lookup[(case_id, run_name)]
            start_utc, start_mono = utcnow(), time.time()
            out_h5 = os.path.join(root, "run", case_id, run_name + ".h5")
            os.makedirs(os.path.dirname(out_h5), exist_ok=True)
            cmd = mesh_command(args.bin, root, run, out_h5)
            error = None
            try:
                with open(os.path.join(log_dir, case_id + "__" + run_name + ".mesh.log"),
                          "w") as log:
                    procedure = subprocess.run(cmd, env=_mesh_env(args, slot),
                                               stdout=log, stderr=subprocess.STDOUT)
                if procedure.returncode != 0:
                    error = {"stage": "mesh:" + run_name,
                             "exit": procedure.returncode}
            except OSError as exc:
                error = {"stage": "mesh:" + run_name, "error": str(exc)}
            finish(case_id, "mesh", run_name, out_h5, slot, start_utc, start_mono,
                   error, {"stage": "mesh:" + run_name, "argv": cmd,
                           "env": {"CUDA_VISIBLE_DEVICES": str(slot)}})

    def ref_worker():
        while True:
            item = ref_jobs.get()
            if item is None:
                return
            case_id, run_name = item
            case, run = run_lookup[(case_id, run_name)]
            ref = _ref_of(case, run)
            start_utc, start_mono = utcnow(), time.time()
            order = (args.pilot_order if args.pilot else None) or int(ref["order"])
            level = (args.pilot_level if (args.pilot and args.pilot_level)
                     else int(ref["max_level"]))
            tol = (args.pilot_tol if (args.pilot and args.pilot_tol)
                   else float(ref["tol"]))
            ref_npz = None
            cmd = []
            error = None
            try:
                ref_npz, cmd = build_reference(root, case, run, args.driver,
                                               order, level, tol, args.ref_threads)
            except (OSError, subprocess.CalledProcessError) as exc:
                error = {"stage": "reference:" + run_name, "error": str(exc)}
            finish(case_id, "reference", run_name, ref_npz, None, start_utc,
                   start_mono, error,
                   {"stage": "reference:" + run_name, "argv": cmd,
                    "env": {"OMP_NUM_THREADS": str(args.ref_threads)}})

    gpu_threads = [threading.Thread(target=gpu_worker, args=(slot,), daemon=True)
                   for slot in range(args.gpus)]
    ref_threads = [threading.Thread(target=ref_worker, daemon=True)
                   for _ in range(args.ref_workers)]
    for thread in gpu_threads + ref_threads:
        thread.start()
    for _ in gpu_threads:
        mesh_jobs.put(None)
    for _ in ref_threads:
        ref_jobs.put(None)
    for thread in gpu_threads:
        thread.join()
    for thread in ref_threads:
        thread.join()
    reduce_pool.shutdown(wait=True)
    for future in reduce_futures:
        future.result()
    return [state["record"] for state in states.values() if state["record"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ubuntu/picbunch-work-20261112/campaign")
    parser.add_argument("--bin", default="")
    parser.add_argument("--driver", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-order", type=int, default=8)
    parser.add_argument("--pilot-level", type=int, default=4)
    parser.add_argument("--pilot-tol", type=float, default=1e-2)
    parser.add_argument("--ref-rate", type=float, default=2e-5,
                        help="reference seconds per sample-point x source")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--ref-workers", type=int, default=8,
                        help="concurrent reference jobs (different cases overlap)")
    parser.add_argument("--ref-threads", type=int, default=16,
                        help="OMP_NUM_THREADS per reference job; "
                             "8x16 leaves headroom on the 144-core node")
    parser.add_argument("--lib-path",
                        default="/home/ubuntu/opt/openPMD-0.17.1-adios2-2.12.1-r2/lib:"
                                "/home/ubuntu/build-adios2-2.12.1/lib")
    parser.add_argument("--git-commit", "--harness-commit", dest="git_commit",
                        default="",
                        help="harness source commit (recorded as harness_commit)")
    parser.add_argument("--build-commit", default="00fb8f7fb22a53748636c70e32bbdb8dd85c23c4",
                        help="commit the picBunch binary was built from")
    args = parser.parse_args()

    doc = load_cases()
    cases = doc["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise SystemExit("unknown case(s): " + ", ".join(sorted(missing)))
    if args.pilot and not args.only:
        args.only = "line-w4b,var1b-1mrad,hires-g10,multivel"
        wanted = set(args.only.split(","))
        cases = [c for c in doc["cases"] if c["id"] in wanted]

    if args.estimate:
        slots, loads = schedule(cases, args.root, args.ref_rate, args.gpus)
        print("%-20s %-10s %8s %8s %8s %8s  slot" %
              ("case", "family", "mesh_s", "ref_s", "reduce_s", "total_s"))
        for index, slot in enumerate(slots):
            for case in slot:
                est = case.get("est", {})
                ref = 0.0
                try:
                    ref = reference_cost(case, args.root, args.ref_rate)
                except (OSError, KeyError):
                    ref = float(est.get("ref_s", 0.0))
                total = float(est.get("mesh_s", 0)) + max(ref, float(est.get("ref_s", 0))) \
                    + float(est.get("reduce_s", 0))
                print("%-20s %-10s %8.0f %8.0f %8.0f %8.0f  %d" %
                      (case["id"], case["family"], float(est.get("mesh_s", 0)),
                       max(ref, float(est.get("ref_s", 0))),
                       float(est.get("reduce_s", 0)), total, index))
        print("\nper-slot estimated wall (s):", ["%.0f" % v for v in loads])
        print("estimated campaign wall (s): %.0f (%.1f h)" % (max(loads), max(loads) / 3600.0))
        print("out-of-scope (not scheduled):", doc["campaign"]["excluded_out_of_scope"])
        return

    if not args.bin or not os.path.isfile(args.bin):
        raise SystemExit("--bin must point at a built picBunch executable")
    args.driver = ensure_driver(args.driver or os.path.join(args.root, "bin", "exact_field_driver"))
    if not args.git_commit:
        try:
            args.git_commit = subprocess.check_output(
                ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            args.git_commit = "unknown"
    args.versions = {"cuda": os.environ.get("CUDA_VERSION", "13.4.92"),
                     "gcc": "13.3.0", "openpmd": "0.17.1"}
    os.makedirs(args.root, exist_ok=True)
    os.makedirs(os.path.join(args.root, "inputs"), exist_ok=True)
    inputs = sorted({run["input"] for case in cases for run in case["runs"]})
    subprocess.run([sys.executable, GEN_INPUTS, "--out-dir",
                    os.path.join(args.root, "inputs"), "--only", ",".join(inputs)],
                   check=True)
    # panel inputs for the multivel particle panels
    panels = sorted({case["reduce"].get("panel_e5") for case in cases
                     if case["reduce"].get("mode") == "multivel"} |
                    {case["reduce"].get("panel_a5") for case in cases
                     if case["reduce"].get("mode") == "multivel"})
    if panels:
        subprocess.run([sys.executable, GEN_INPUTS, "--out-dir",
                        os.path.join(args.root, "inputs"),
                        "--only", "mvpanel-e5,mvpanel-a5"], check=True)

    started = time.time()
    log_dir = os.path.join(args.root, "logs")
    results = execute(cases, args, log_dir)
    wall = time.time() - started
    summary = {"wall_s": wall,
               "scheduler": "GPU mesh work queue (one CUDA_VISIBLE_DEVICES slot "
                            "per worker) + concurrent reference pool",
               "gpus": args.gpus, "ref_workers": args.ref_workers,
               "ref_threads": args.ref_threads,
               "harness_commit": args.git_commit,
               "build_commit": args.build_commit,
               "binary_sha256": sha256_file(args.bin),
               "case_slots": {r["case_id"]: r["gpu_slot"] for r in results},
               "cases": {r["case_id"]: {"status": r["status"],
                                        "wall_s": r["wall_s"]}
                         for r in results},
               "pilot": bool(args.pilot)}
    with open(os.path.join(args.root, "prov", "campaign_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    if args.pilot:
        with open(os.path.join(args.root, "prov", "pilot_report.json"), "w") as handle:
            json.dump(summary, handle, indent=1)
    failed = [r["case_id"] for r in results if r["status"] != "done"]
    print("campaign wall %.1fs; failed: %s" % (wall, failed or "none"))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
