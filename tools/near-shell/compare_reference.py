#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Compare measure_near_shell output with the independent reference.

Uses indfield.py (different panelisation and coordinate transform) from the
pr24 reference tree.  Reports per-component worst relative error for the total
field, the mesh-only field outside the shell, and the seam excess across the
near/far transition.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

OFFSETS = np.array([
    (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
KC = 8.9875517873681764e9
C = 299792458.0
QE = -1.602176634e-19


def read_meta(prefix):
    m = {}
    with open(prefix + "_meta.txt") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            try:
                m[p[0]] = [float(x) for x in p[1:]] if len(p) > 2 else float(p[1])
            except ValueError:
                m[p[0]] = p[1]
    return m


def load_field(prefix, suffix, n):
    a = np.fromfile(prefix + suffix, dtype='<f8')
    return a.reshape(6, n[2], n[1], n[0])


def truth_at(ref, pos, u, cell):
    e, b = ref.integrate_point(pos, np.zeros(3), u, cell[0], cell[1], cell[2],
                               order=12, refine=3)
    return np.concatenate([e, b]) * np.array([KC * QE] * 3 + [KC / C * QE] * 3)


def relerr(a, b):
    scale = np.maximum(np.max(np.abs(b), axis=-1, keepdims=True), 1e-300)
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-6 * scale)
    return np.where(np.abs(b) > 1e-9 * scale, rel, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('--indfield-dir', required=True)
    ap.add_argument('--gpu-truth-dir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--truth-cache', default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.indfield_dir)
    import indfield as ref

    m = read_meta(args.prefix)
    n = [int(x) for x in m['grid']]
    origin = np.array(m['origin']); cell = np.array(m['cell'])
    u = np.array(m['u']); src = np.array(m['source'])
    radius = m['radius']; tile = int(m['tile'])
    mesh = load_field(args.prefix, '_mesh.bin', n)
    total = load_field(args.prefix, '_total.bin', n)

    idx = np.stack(np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                               indexing='ij'), axis=-1)
    # max-norm distance in each axis's own cell unit
    dist = np.max(np.abs((origin + idx * cell - src) / cell), axis=-1)

    truth = np.empty((6, n[2], n[1], n[0]))
    cached = False
    if args.truth_cache and os.path.exists(args.truth_cache):
        truth = np.load(args.truth_cache)['truth']
        cached = True
    elif args.gpu_truth_dir:
        sys.path.insert(0, args.gpu_truth_dir)
        import cupy as cp
        cp.cuda.Device(int(os.environ.get('NS1_GPU', '2'))).use()
        from gate import gpu_truth
        for c in range(6):
            pos_cell = (origin + idx * cell - src) / cell + OFFSETS[c]
            f = gpu_truth(pos_cell.reshape(-1, 3), u, cell, order=12, refine=3)
            f = f * np.array([KC * QE] * 3 + [KC / C * QE] * 3)
            truth[c] = f[:, c].reshape(n[0], n[1], n[2]).transpose(2, 1, 0)
        if args.truth_cache:
            np.savez(args.truth_cache, truth=truth)
    else:
        for c in range(6):
            pos = origin + idx * cell + OFFSETS[c] * cell
            for m, p in enumerate(pos.reshape(-1, 3)):
                e, b = ref.integrate_point(p, np.zeros(3), u,
                                           cell[0], cell[1], cell[2], order=12, refine=3)
                f = np.concatenate([e, b]) * np.array([KC * QE] * 3 + [KC / C * QE] * 3)
                truth[c].flat[m] = f[c]
            truth[c] = truth[c].reshape(n[0], n[1], n[2]).transpose(2, 1, 0)

    # near flag per node reproduces the tile selection for a single source.
    src_idx = (src - origin) / cell
    near = np.zeros((n[0], n[1], n[2]), dtype=bool)
    for c in range(6):
        p = idx + OFFSETS[c]
        lo = (np.floor(p / tile) * tile)
        hi = lo + tile - 1
        near_c = np.all((src_idx >= lo - radius) & (src_idx <= hi + radius), axis=-1)
        if c == 0:
            near = near_c.transpose(2, 1, 0)

    out = {}
    tr = truth.reshape(6, -1)
    to = total.reshape(6, -1)
    me = mesh.reshape(6, -1)
    d = dist.transpose(2, 1, 0).reshape(-1)
    nr = near.reshape(-1)
    re = relerr(to.T, tr.T).T
    out['total_per_comp'] = re.max(axis=1).tolist()
    out['total_worst'] = float(re.max())
    out['total_near_worst'] = float(re[:, nr].max())
    out['total_far_worst'] = float(re[:, ~nr].max()) if np.any(~nr) else 0.0
    rm = relerr(me.T, tr.T).T
    out['mesh_far_per_comp'] = rm[:, ~nr].max(axis=1).tolist() if np.any(~nr) else [0] * 6
    out['mesh_far_worst'] = float(rm[:, ~nr].max()) if np.any(~nr) else 0.0
    out['effective_radius_cells'] = float(d[nr].max()) if np.any(nr) else 0.0

    # Seam: adjacent nodes along each axis where the near flag changes.
    seam = np.zeros((6, 3))
    scale = np.maximum(np.max(np.abs(truth), axis=(1, 2, 3)), 1e-300)
    for c in range(6):
        for a in range(3):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[a] = slice(0, -1)
            sl1[a] = slice(1, None)
            m0 = total[c][tuple(sl0)]
            m1 = total[c][tuple(sl1)]
            t0 = truth[c][tuple(sl0)]
            t1 = truth[c][tuple(sl1)]
            flag0 = near[tuple(sl0)]
            flag1 = near[tuple(sl1)]
            cross = flag0 != flag1
            if np.any(cross):
                jump_model = np.abs(m0 - m1)[cross]
                jump_ref = np.abs(t0 - t1)[cross]
                excess = np.abs(jump_model - jump_ref)
                seam[c, a] = excess.max() / scale[c]
    out['seam_per_comp'] = seam.max(axis=1).tolist()
    out['seam_worst'] = float(seam.max())
    out['radius'] = radius
    out['tile'] = tile
    out['pairs'] = m['pairs']
    out['truth_cached'] = cached
    out['mesh_seconds'] = m['mesh_seconds']
    out['near_shell_seconds'] = m['near_shell_seconds']
    import json
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
