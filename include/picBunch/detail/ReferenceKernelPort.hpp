// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Standalone port of the validated stationary in-support evaluator from
// psychocoderHPC-bot/picongpu, PR #24 head 2baa2566
// (commits 18b959c "neuralBunch reference: analytic in-support Coulomb field"
// and ec48c52), file
// include/picongpu/particles/neuralBunch/ReferenceKernel.hpp.
//
// Only the two self-contained host routines are reproduced here:
//   box_field                    analytic singular part (divergence theorem,
//                                six-face Gauss over the source support box)
//   remainder_support_integral   smooth remainder, octant split at the anchor
//                                t = R0/cells so the integrable 1/r singularity
//                                sits on an octant corner
//
// The upstream in-support path subtracts the isotropic Coulomb box field only
// for u = 0.  The relativistic kernel is anisotropic, so this decomposition is
// not valid for moving sources; in_support_stationary() is therefore documented
// as stationary-only even though it accepts a velocity argument (the upstream
// code takes the same shape, it simply never calls this branch for u != 0).
//
// No alpaka, PMacc, BLAS or FFT dependency: std + <cmath> only.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace picBunch::refkernel {
struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;

  double &operator[](int i) { return (&x)[i]; }

  double const &operator[](int i) const { return (&x)[i]; }
};

inline Vec3 operator+(Vec3 a, Vec3 const &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(Vec3 a, Vec3 const &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }

inline Vec3 operator*(double s, Vec3 a) { return a * s; }

inline Vec3 operator/(Vec3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }

inline double dot(Vec3 const &a, Vec3 const &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 const &a, Vec3 const &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

//! PIConGPU TSC assignment function; argument in cell units.
inline double tsc_shape(double x) {
  double const ax = std::abs(x);
  if (ax < 0.5)
    return 0.75 - ax * ax;
  if (ax < 1.5)
    return 0.5 * (1.5 - ax) * (1.5 - ax);
  return 0.0;
}

//! Stable present-time point kernel (per-unit-charge fields).
inline void point_field(Vec3 const &R, Vec3 const &u, Vec3 &E, Vec3 &B) {
  double const uu2 = dot(u, u);
  double const inv_g2 = 1.0 / (1.0 + uu2);

  if (uu2 == 0.0) {
    double const r2 = dot(R, R);
    double const denom = r2 * std::sqrt(r2);
    E = R / denom;
    B = Vec3{0.0, 0.0, 0.0};
    return;
  }

  double const speed = std::sqrt(uu2);
  Vec3 const bhat = u / speed;
  double const r_par = dot(R, bhat);
  Vec3 const r_perp = R - r_par * bhat;
  double const D = r_par * r_par + dot(r_perp, r_perp) * inv_g2;
  double const denom = D * std::sqrt(D);

  E = R * (inv_g2 / denom);
  Vec3 const beta = u / std::sqrt(1.0 + uu2);
  B = cross(beta, E);
}

//! Gauss-Legendre nodes/weights on [-1, 1], computed at run time.
inline void gauss_legendre(int n, std::vector<double> &nodes,
                           std::vector<double> &weights) {
  if (n < 1)
    throw std::invalid_argument("Gauss order must be positive");
  nodes.assign(n, 0.0);
  weights.assign(n, 0.0);
  int const m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(3.141592653589793238462643383279502884 * (i + 0.75) /
                        (n + 0.5));
    double pp = 0.0;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double p1 = 1.0;
      double p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        double const p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      double const z1 = z;
      z = z1 - p1 / pp;
      if (std::abs(z - z1) <= 1e-15)
        break;
    }
    nodes[i] = -z;
    nodes[n - 1 - i] = z;
    double const w = 2.0 / ((1.0 - z * z) * pp * pp);
    weights[i] = w;
    weights[n - 1 - i] = w;
  }
}

//! Coulomb field at an interior point of the physical TSC support box:
//!     J(x_obs) = int_Box (x_obs - p)/|x_obs - p|^3 d^3p
//! where Box is [x_src +/- 1.5 cells].  By the divergence theorem the volume
//! integral becomes a smooth surface integral over the six faces.  Upstream
//! evaluates each face with an n-by-n Gauss rule (the "asinh face terms" of
//! the round-9 description are the same analytic surface integral; this
//! revision of the file uses the six-face Gauss variant).
inline Vec3 box_field(Vec3 const &x_obs, Vec3 const &x_src, Vec3 const &cells,
                      int n) {
  double const ext[3] = {1.5 * cells.x, 1.5 * cells.y, 1.5 * cells.z};
  double const lo[3] = {x_src.x - ext[0], x_src.y - ext[1], x_src.z - ext[2]};
  double const hi[3] = {x_src.x + ext[0], x_src.y + ext[1], x_src.z + ext[2]};
  std::vector<double> xn, wn;
  gauss_legendre(n, xn, wn);
  Vec3 out{};
  for (int axis = 0; axis < 3; ++axis)
    for (int sgn : {-1, 1}) {
      double const fixed = (sgn < 0) ? lo[axis] : hi[axis];
      double const df = x_obs[axis] - fixed;
      int const o0 = (axis + 1) % 3;
      int const o1 = (axis + 2) % 3;
      double const h0 = 0.5 * (hi[o0] - lo[o0]);
      double const m0 = 0.5 * (hi[o0] + lo[o0]);
      double const h1 = 0.5 * (hi[o1] - lo[o1]);
      double const m1 = 0.5 * (hi[o1] + lo[o1]);
      double P = 0.0;
      for (int i = 0; i < n; ++i) {
        double const v0 = m0 + h0 * xn[i];
        double const d0 = v0 - x_obs[o0];
        for (int j = 0; j < n; ++j) {
          double const v1 = m1 + h1 * xn[j];
          double const d1 = v1 - x_obs[o1];
          P += (h0 * h1) * wn[i] * wn[j] /
               std::sqrt(d0 * d0 + d1 * d1 + df * df);
        }
      }
      out[axis] += sgn * P;
    }
  return out;
}

//! Smooth-remainder quadrature for the analytic-singularity subtraction:
//!     int_S (W(s) - w_offset) K(s) d^3s ,  K(s) = (R0 - s*cells)/|...|^3
//! The support box [-1.5, 1.5]^3 is split at t (the singularity) into up to
//! eight octants, so the integrable 1/r singularity (one power milder than
//! the bare 1/r^2 kernel, because W(s) - w_offset vanishes at s = t) sits at
//! an octant corner.  Plain Gauss on each octant converges algebraically;
//! the n vs 2n order pair is a genuine error estimate at the anchor.
inline void remainder_support_integral(Vec3 const &R0, Vec3 const &u,
                                       Vec3 const &cells, int n,
                                       double w_offset, Vec3 &Eout,
                                       Vec3 &Bout) {
  double const tb[3] = {R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
  std::array<std::vector<std::array<double, 2>>, 3> seg;
  for (int a = 0; a < 3; ++a) {
    if (tb[a] > -1.5)
      seg[a].push_back({-1.5, tb[a]});
    if (tb[a] < 1.5)
      seg[a].push_back({tb[a], 1.5});
  }
  std::vector<double> xn, wn;
  gauss_legendre(n, xn, wn);
  std::array<double, 6> acc{};
  for (auto const &px : seg[0])
    for (auto const &py : seg[1])
      for (auto const &pz : seg[2]) {
        double const hx = 0.5 * (px[1] - px[0]);
        double const mx = 0.5 * (px[1] + px[0]);
        double const hy = 0.5 * (py[1] - py[0]);
        double const my = 0.5 * (py[1] + py[0]);
        double const hz = 0.5 * (pz[1] - pz[0]);
        double const mz = 0.5 * (pz[1] + pz[0]);
        for (int i = 0; i < n; ++i) {
          double const sx = mx + hx * xn[i];
          double const wx = hx * wn[i];
          for (int j = 0; j < n; ++j) {
            double const sy = my + hy * xn[j];
            double const wxy = wx * hy * wn[j];
            for (int k = 0; k < n; ++k) {
              double const sz = mz + hz * xn[k];
              double const W = tsc_shape(sx) * tsc_shape(sy) * tsc_shape(sz);
              double const Ww = W - w_offset;
              if (Ww == 0.0)
                continue;
              Vec3 const R{R0.x - sx * cells.x, R0.y - sy * cells.y,
                           R0.z - sz * cells.z};
              Vec3 E, B;
              point_field(R, u, E, B);
              double const ww = wxy * hz * wn[k] * Ww;
              acc[0] += ww * E.x;
              acc[1] += ww * E.y;
              acc[2] += ww * E.z;
              acc[3] += ww * B.x;
              acc[4] += ww * B.y;
              acc[5] += ww * B.z;
            }
          }
        }
      }
  Eout = {acc[0], acc[1], acc[2]};
  Bout = {acc[3], acc[4], acc[5]};
}

//! Number of point-kernel evaluations in one box_field call at order n.
inline std::size_t box_nodes(int n) {
  return 6u * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
}

//! Number of nonzero octants used by the split at t (1..8).
inline int remainder_octants(Vec3 const &R0, Vec3 const &cells) {
  double const tb[3] = {R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
  int octants = 1;
  for (int a = 0; a < 3; ++a) {
    int const segments = (tb[a] > -1.5 ? 1 : 0) + (tb[a] < 1.5 ? 1 : 0);
    octants *= segments;
  }
  return octants;
}

//! Number of point-kernel evaluations in one remainder call at order n.
inline std::size_t remainder_nodes(Vec3 const &R0, Vec3 const &cells, int n) {
  return static_cast<std::size_t>(remainder_octants(R0, cells)) *
         static_cast<std::size_t>(n) * static_cast<std::size_t>(n) *
         static_cast<std::size_t>(n);
}

struct InSupportResult {
  Vec3 E{};
  Vec3 B{};
  double error = 0.0;
  bool converged = false;
  int box_order = 0;
  int remainder_order = 0;
  std::size_t box_node_count = 0;
  std::size_t remainder_node_count = 0;
  std::size_t total_node_count = 0;
};

//! Stationary in-support evaluator: analytic singular box plus smooth
//! remainder, exactly the 2baa2566 decomposition with a caller-chosen fixed
//! box order and remainder order.
//!
//! VALID FOR u = 0 ONLY.  The isotropic Coulomb box does not subtract the
//! 1/gamma beaming peak of a moving source (T123: relative floor 0.15-0.42
//! at gamma 10/100).  The velocity argument is kept so callers can reproduce
//! that failure explicitly.
inline InSupportResult in_support_stationary(Vec3 const &R0, Vec3 const &u,
                                             Vec3 const &cells, int box_order,
                                             int remainder_order) {
  Vec3 const t{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
  double const Wt = tsc_shape(t.x) * tsc_shape(t.y) * tsc_shape(t.z);
  Vec3 const S = box_field(R0, Vec3{}, cells, box_order) *
                 (Wt / (cells.x * cells.y * cells.z));
  Vec3 R, B;
  remainder_support_integral(R0, u, cells, remainder_order, Wt, R, B);
  InSupportResult out;
  out.E = S + R;
  out.B = B; // exactly zero for a stationary source
  out.box_order = box_order;
  out.remainder_order = remainder_order;
  out.box_node_count = box_nodes(box_order);
  out.remainder_node_count = remainder_nodes(R0, cells, remainder_order);
  out.total_node_count = out.box_node_count + out.remainder_node_count;
  return out;
}

//! Upstream default policy: fixed box order, remainder order doubled from
//! 16 until the n vs 2n Richardson estimate is below tol or n reaches 512.
inline InSupportResult
in_support_stationary_adaptive(Vec3 const &R0, Vec3 const &u, Vec3 const &cells,
                               int box_order = 256, double tol = 1e-7,
                               int n_start = 16, int n_max = 512) {
  Vec3 const t{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
  double const Wt = tsc_shape(t.x) * tsc_shape(t.y) * tsc_shape(t.z);
  Vec3 const S = box_field(R0, Vec3{}, cells, box_order) *
                 (Wt / (cells.x * cells.y * cells.z));

  int n = n_start;
  Vec3 E1, B1, E2, B2;
  remainder_support_integral(R0, u, cells, n, Wt, E1, B1);
  double err = 0.0;
  for (;;) {
    int const n2 = std::min(2 * n, n_max);
    remainder_support_integral(R0, u, cells, n2, Wt, E2, B2);
    std::array<double, 6> a{E1.x, E1.y, E1.z, B1.x, B1.y, B1.z};
    std::array<double, 6> b{E2.x, E2.y, E2.z, B2.x, B2.y, B2.z};
    double max_diff = 0.0;
    double max_val = 0.0;
    for (int i = 0; i < 6; ++i) {
      max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
      max_val = std::max(max_val, std::abs(a[i]));
      max_val = std::max(max_val, std::abs(b[i]));
    }
    err = max_diff / (1.0 + max_val);
    if (err <= tol || n2 >= n_max) {
      InSupportResult out;
      out.E = S + E2;
      out.B = B2;
      out.error = err;
      out.converged = err <= tol;
      out.box_order = box_order;
      out.remainder_order = n2;
      out.box_node_count = box_nodes(box_order);
      out.remainder_node_count = remainder_nodes(R0, cells, n2);
      out.total_node_count = out.box_node_count + out.remainder_node_count;
      return out;
    }
    E1 = E2;
    B1 = B2;
    n = n2;
  }
}

} // namespace picBunch::refkernel
