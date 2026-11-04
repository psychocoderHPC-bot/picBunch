# GPU1 — exact in-support near-field on V100 (run pr24-20261102)

## Kernel (3 lines)
- One thread per (source, staggered-node, component) pair; double-precision tensor
  Gauss-Legendre quadrature of the TSC-convolved moving point-charge field
  F_c = ∫ W(s) Kpoint_c(R0−s) ds^3 over [-1.5,1.5]^3 (cell=1) — same work/pair as the CPU reference.
- TSC support split at knots {−1.5,−0.5,0.5,1.5}; in-support pairs split the anchor panel at R0 and
  map each panel s=c+(b−c)τ^5 (Jacobian (b−c)·5τ^4 cancels the 1/r^2 singularity) — the independent
  reference's scheme; each panel is bisected 2^refine times (order/refine are CLI flags).

## V100-SXM2-32GB throughput (saturated, γ=10 axial u=(0,0,9.95))
- In-support (singular, 4 panels/axis): (8,0) 6.1e5 pairs/s (2.0e10 nodes/s); (8,1) 7.8e4; (8,2) ~1e4.
- Band 1.5–4 cells (non-singular, 3 panels/axis): (8,0) ~9e5 pairs/s (1.2e10 nodes/s); (8,1) 1.1e5; (8,2) ~1e4.
- H100 not measured. Node rate is only ~3× the CPU reference (~2e10 vs ~6.8e9 nodes/s), because the
  kernel is double-precision / sqrt-bound, not FLOP-bound.

## Accuracy (worst per-component rel err vs converged ref o10r3==o10r4, γ=10)
- In-support: (8,0) ~9e-4 (marginal); (8,1) 2.6e-5; (8,2) 8e-13.   Band: (8,0) 6.6e-4; (8,1) 3.5e-5; (8,2) 4e-6.
- The GPU kernel reproduces the independent reference to 2.3e-8 at every (order, refine) (anchor + bisection verified).

## 300 s projection (2e6 particles; in-support 1.7e8, 3-cell 1.7e9, 4-cell 3.4e9 pairs)
- support-only: (8,0) ~280 s; (8,1) ~2200 s; (8,2) ~1.8e4 s.
- 3-cell shell: (8,0) ~2000 s; (8,1) ~1.6e4 s; (8,2) ~1.7e5 s.
- 4-cell shell: (8,0) ~3900 s; (8,1) ~3.1e4 s; (8,2) ~3.4e5 s.

## 300 s verdict at 1e-3
- REACHABLE only for the support-only shell at the MARGINAL (8,0) config (~280 s, ~9e-4 worst).
- NOT reachable at a safe 1e-3 config: support-only (8,1) ~2200 s (7× over); the realistic 3–4-cell
  shells (mesh seam at h=1/4) are ~7–100× over on one V100. CPU ref 3.1e-4 s/pair ≈ 14.6 h for 1.7e8;
  the GPU is ~3× faster at matched accuracy, so 4× V100 → ~550 s support-only (8,1), still >300 s.

## Recommended config
order 8, refine 1 (safe ~3e-5). 300 s needs support-only + refine 0 (marginal) and/or multi-GPU/H100;
the 3–4-cell exact shell is out of reach at 1e-3 on a single node — let the mesh carry the band beyond ~1.5 cells.

## Repro (branch experimental/gpu-near)
gpu_near.cu (kernel+driver: --kernel --band --order --refine), acc2.py (accuracy vs converged ref),
verify_ref.py (GPU==ref at matched config), ref_check.py, diag.py. nvcc -O3 -arch=sm_70 -std=c++20
-expt-relaxed-constexpr -I <alpaka> -lhwloc -ldl.
