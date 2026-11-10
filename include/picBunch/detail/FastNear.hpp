// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Fast near-field evaluator.
//
// Ported from psychocoderHPC-bot/picongpu tools/closed-form/ (branch
// experimental/closed-form, commit b3699dff1ca6e2685292597c2c91974e25c6c76f,
// gpu_extract.cu / cf1_core.py).  The closed form in this header is the CPU
// reference of the same algorithm the CUDA tool runs.
//
// In-support field of one source for a TSC-convolved uniformly moving charge:
//
//   F(R0) = W(s*) * E_box(R0) + int (S(s) - W(s*)) K(R0 - s*cells) ds^3
//
// s* = R0/cells, W the separable TSC shape, E_box the exact divergence-theorem
// field of the unit-density support box (24 asinh calls, O(1)), and K the point
// kernel.  The remainder is plain tensor Gauss-Legendre on the subtracted
// integrand over the TSC knot panels, each axis split at s* when interior.
//
// The asinh primitive T(A,B,u) uses |u| in the zero branch.  The un-fixed form
// fired for every u < 0 and produced 30-400x wrong fields for points outside a
// box face.
//
// Moving sources: axial beta has the exact gamma-stretched box.  Oblique
// in-support pairs are integrated directly in the Lorentz-contracted
// d = M R frame, where the kernel becomes the isotropic Coulomb field: the
// support box maps to a parallelepiped and the TSC shape to a piecewise
// polynomial along each ray.  The radial integral is exact with four Gauss
// nodes per knot interval and the angular rule is polar-aligned with beta,
// so the beamed tube is resolved without the former Duffy sliver problem.
// The exact constant-density polyhedron field is kept as
// detail::exactObliqueBox (closed-form face/edge terms) for validation and
// as the analytic reference for the d-space integral.

#pragma once

#include "picBunch/detail/NearReference.hpp"
#include "picBunch/detail/ObliqueOrder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <mutex>
#include <vector>

namespace picBunch::pbfast
{
    using nbref::Vec3;

    struct Settings
    {
        //! Gauss order per knot/s* panel of the subtracted remainder.
        int order = 5;
        //! Rest-frame octant order of the oblique singular box (8*order^3 nodes).
        int obliqueOrder = 8;
        //! Power-transform exponent of the oblique singular box.
        int obliquePower = 5;
        //! Per-pair error budget source.  The oblique in-support path picks the
        //! controlled quadrature when the octant/Duffy order-pair residual
        //! exceeds a fraction of this value.
        double tolerance = 1e-3;
        //! Which TSC support cell classes use the exact s-space evaluator.
        //! The support is split at the TSC knots, so the source-relative cell
        //! of an observer is ``floor(R0/cell + 1/2)`` per axis:
        //!  - 2: every in-support pair (the historical all-exact routing),
        //!  - 1: the observer-containing cell plus the six face-adjacent cells,
        //!  - 0: only the observer-containing cell (production hybrid default),
        //!  - -1: never (legacy quadrature only).
        //! Out-of-support high-gamma pairs always use the exact evaluator.
        int exactCellReach = 0;
    };

    /*! Resolve the automatic exact-cell sentinel for one output tile.
     *
     * -2 selects the observer cell at the default 1e-3 tolerance and the full
     * support at tolerance <= 1e-4.  A gamma-100 tile is always full exact,
     * because its contracted tube is unresolvable on the legacy quadrature.
     * An explicit reach (including -1 legacy) otherwise wins.
     */
    inline int resolveExactCellReach(int requested, double tolerance, double tileMaxGamma)
    {
        int reach = requested;
        if(reach == -2)
            reach = (tolerance <= 1.0e-4) ? 2 : 0;
        if(isExactGamma(tileMaxGamma))
            reach = 2;
        return reach;
    }

    namespace detail
    {
        inline double tsc(double x)
        {
            return nbref::tsc_shape(x);
        }

        //! int_0^A int_0^B 1/sqrt(u^2 + X^2 + Y^2) dX dY.  A, B > 0.
        inline double Tfun(double A, double B, double u)
        {
            if(std::abs(u) < 1e-300)
                return A * std::asinh(B / A) + B * std::asinh(A / B);
            return A * std::asinh(B / std::sqrt(u * u + A * A))
                + B * std::asinh(A / std::sqrt(u * u + B * B))
                - u * std::atan(A * B / (u * std::sqrt(A * A + u * u + B * B)));
        }

        //! Coulomb field component c at p of the unit-density box [lo,hi]^3.
        //! Only called for p inside the box, so every edge length is positive.
        inline double boxField8T(double const p[3], double const lo[3], double const hi[3], int c)
        {
            double E = 0.0;
            if(c == 0)
            {
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(double sy : ey)
                    for(double sz : ez)
                        E += Tfun(sy, sz, hi[0] - p[0]) - Tfun(sy, sz, p[0] - lo[0]);
            }
            else if(c == 1)
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(double sx : ex)
                    for(double sz : ez)
                        E += Tfun(sx, sz, hi[1] - p[1]) - Tfun(sx, sz, p[1] - lo[1]);
            }
            else
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                for(double sx : ex)
                    for(double sy : ey)
                        E += Tfun(sx, sy, hi[2] - p[2]) - Tfun(sx, sy, p[2] - lo[2]);
            }
            return E;
        }

        struct Rule
        {
            std::vector<double> x, w;
        };

        inline Rule makeRule(int n)
        {
            Rule r;
            r.x.assign(static_cast<std::size_t>(n), 0.0);
            r.w.assign(static_cast<std::size_t>(n), 0.0);
            int const m = (n + 1) / 2;
            for(int i = 0; i < m; ++i)
            {
                double z = std::cos(3.141592653589793238462643383279502884 * (i + 0.75) / (n + 0.5));
                double pp = 0.0;
                for(int it = 0; it < 100; ++it)
                {
                    double p1 = 1.0;
                    double p2 = 0.0;
                    for(int j = 1; j <= n; ++j)
                    {
                        double const p3 = p2;
                        p2 = p1;
                        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                    }
                    pp = n * (z * p1 - p2) / (z * z - 1.0);
                    double const z1 = z;
                    z = z1 - p1 / pp;
                    if(std::abs(z - z1) <= 1e-15)
                        break;
                }
                r.x[static_cast<std::size_t>(i)] = -z;
                r.x[static_cast<std::size_t>(n - 1 - i)] = z;
                double const w = 2.0 / ((1.0 - z * z) * pp * pp);
                r.w[static_cast<std::size_t>(i)] = w;
                r.w[static_cast<std::size_t>(n - 1 - i)] = w;
            }
            return r;
        }

        inline Rule const& rule(int n)
        {
            static Rule const r4 = makeRule(4);
            static Rule const r5 = makeRule(5);
            static Rule const r6 = makeRule(6);
            static Rule const r8 = makeRule(8);
            static Rule const r10 = makeRule(10);
            static Rule const r12 = makeRule(12);
            // The allowed oblique orders are precomputed so a tile never pays
            // for Gauss-table generation on the evaluation path.
            if(n == 4)
                return r4;
            if(n == 5)
                return r5;
            if(n == 6)
                return r6;
            if(n == 8)
                return r8;
            if(n == 10)
                return r10;
            if(n == 12)
                return r12;
            static std::mutex mutex;
            static std::map<int, Rule> cache;
            std::lock_guard<std::mutex> lock(mutex);
            return cache.emplace(n, makeRule(n)).first->second;
        }

        //! M = n n^T + (1/gamma)(I - n n^T): the Lorentz contraction of the
        //! Liénard kernel, |M R|^2 = R^2(1-beta^2) + (R.beta)^2.
        inline Vec3 applyMmin(Vec3 const& v, Vec3 const& n, double gam)
        {
            double const vn = nbref::dot(v, n);
            return v + (1.0 / gam - 1.0) * (v - vn * n);
        }

        //! M^{-1} = n n^T + gamma (I - n n^T).
        inline Vec3 applyMplus(Vec3 const& v, Vec3 const& n, double gam)
        {
            double const vn = nbref::dot(v, n);
            return v + (gam - 1.0) * (v - vn * n);
        }

        //! True when beta is aligned with one coordinate axis (the exact
        //! gamma-stretched box branch).
        inline bool isAxialBeta(Vec3 const& u, Vec3 const& betaH)
        {
            if(nbref::dot(u, u) == 0.0)
                return false;
            for(int a = 0; a < 3; ++a)
                if(std::abs(std::abs(betaH[a]) - 1.0) < 1e-12)
                    return true;
            return false;
        }

        /*! Singular support-box field in Lorentz-contracted coordinates.
         *
         * In d-space the Liénard kernel is |M R|^{-3} R.  With d = M R the
         * Jacobian cancels the (1-beta^2) prefactor and the integrand becomes
         * M^{-1} d / |d|^3, a plain Coulomb field.  The support box of the
         * observer is split into the eight corner boxes around R = 0; each is
         * in turn split into the six tetrahedra of a cube and integrated with
         * the collapsed (Duffy) tetrahedral map
         *
         *     d(u,v,w) = u [ (1-v)(1-w) A + v(1-w) B + w C ],
         *     jac     = u^2 |det(Q, dQ/dv, dQ/dw)|,
         *
         * which turns the 1/|d|^2 vertex singularity into a smooth integrand.
         * This is the controlled variant used only when the octant residual
         * exceeds the per-pair budget.
         */
        inline void boxTermDuffy(
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u,
            int order,
            Vec3& E,
            Vec3& B)
        {
            double const uu2 = nbref::dot(u, u);
            if(uu2 == 0.0)
            {
                E = Vec3{};
                B = Vec3{};
                return;
            }
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const betaH = u / std::sqrt(uu2);
            double const Lp[3] = {
                1.5 * cells.x - R0.x,
                1.5 * cells.y - R0.y,
                1.5 * cells.z - R0.z};
            double const Ln[3] = {
                1.5 * cells.x + R0.x,
                1.5 * cells.y + R0.y,
                1.5 * cells.z + R0.z};
            Rule const& gl = rule(order);
            Vec3 acc{};
            int const perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
            for(int sx = -1; sx <= 1; sx += 2)
                for(int sy = -1; sy <= 1; sy += 2)
                    for(int sz = -1; sz <= 1; sz += 2)
                    {
                        double const L[3] = {
                            sx > 0 ? Lp[0] : Ln[0],
                            sy > 0 ? Lp[1] : Ln[1],
                            sz > 0 ? Lp[2] : Ln[2]};
                        if(L[0] <= 1e-14 || L[1] <= 1e-14 || L[2] <= 1e-14)
                            continue;
                        Vec3 const e[3] = {
                            applyMmin(Vec3{sx * L[0], 0.0, 0.0}, betaH, gam),
                            applyMmin(Vec3{0.0, sy * L[1], 0.0}, betaH, gam),
                            applyMmin(Vec3{0.0, 0.0, sz * L[2]}, betaH, gam)};
                        for(int pi = 0; pi < 6; ++pi)
                        {
                            Vec3 const A = e[perm[pi][0]];
                            Vec3 const Bv = A + e[perm[pi][1]];
                            Vec3 const C = A + e[perm[pi][1]] + e[perm[pi][2]];
                            for(int iu = 0; iu < order; ++iu)
                            {
                                double const tu = 0.5 * gl.x[static_cast<std::size_t>(iu)] + 0.5;
                                double const wu = 0.5 * gl.w[static_cast<std::size_t>(iu)];
                                for(int iv = 0; iv < order; ++iv)
                                {
                                    double const tv = 0.5 * gl.x[static_cast<std::size_t>(iv)] + 0.5;
                                    double const wv = 0.5 * gl.w[static_cast<std::size_t>(iv)];
                                    for(int iw = 0; iw < order; ++iw)
                                    {
                                        double const tw = 0.5 * gl.x[static_cast<std::size_t>(iw)] + 0.5;
                                        double const ww = 0.5 * gl.w[static_cast<std::size_t>(iw)];
                                        Vec3 const Q = (1.0 - tv) * (1.0 - tw) * A
                                            + tv * (1.0 - tw) * Bv + tw * C;
                                        Vec3 const Qv = (1.0 - tw) * (Bv - A);
                                        Vec3 const Qw = -(1.0 - tv) * A - tv * Bv + C;
                                        double const det = nbref::dot(Q, nbref::cross(Qv, Qw));
                                        Vec3 const dp = tu * Q;
                                        double const rho = std::sqrt(nbref::dot(dp, dp));
                                        if(rho < 1e-300)
                                            continue;
                                        Vec3 const R = applyMplus(dp, betaH, gam);
                                        double const jac = tu * tu * std::abs(det) * wu * wv * ww;
                                        acc = acc - (jac / (rho * rho * rho)) * R;
                                    }
                                }
                            }
                        }
                    }
            double const invVolume = 1.0 / (cells.x * cells.y * cells.z);
            E = invVolume * acc;
            B = nbref::cross(u / gam, E);
        }

        inline double norm3(Vec3 const& v)
        {
            return std::sqrt(nbref::dot(v, v));
        }

        constexpr double kBoxPi = 3.141592653589793238462643383279502884;

        //! Stable Lorentz contraction v_par + k v_perp.  Never subtracts O(1)
        //! terms, so it stays accurate for gamma ~ 1e2 and beyond.
        inline Vec3 applyContract(Vec3 const& v, Vec3 const& n, double k)
        {
            double const vn = nbref::dot(v, n);
            return vn * n + k * (v - vn * n);
        }

        //! Primitive d*J(l) of a parallelogram edge contribution to the face
        //! integral I = int_face dA/|x|.  J(l) = asinh(l/a) + (h/d) atan(h l/(d r)),
        //! a^2 = h^2 + d^2, r^2 = a^2 + l^2.  Ordinary atan (not atan2) keeps the
        //! branch continuous for d < 0.
        inline double boxEdgePrimitive(double l, double hh, double d)
        {
            double const a = std::sqrt(hh * hh + d * d);
            double const r = std::sqrt(a * a + l * l);
            double const asinhTerm = d * std::asinh(l / a);
            double atanTerm;
            if(d == 0.0)
                atanTerm = (l > 0.0 ? 0.5 * kBoxPi : (l < 0.0 ? -0.5 * kBoxPi : 0.0)) * hh;
            else
                atanTerm = hh * std::atan(hh * l / (d * r));
            return asinhTerm + atanTerm;
        }

        //! Closed-form int_face dA/|x| over a parallelogram face with outward
        //! unit normal n.  q must be in cyclic order.  The interior disk term
        //! 2*pi*|h| is removed when the origin projects inside the face.
        inline double boxFaceIntegral(Vec3 const q[4], Vec3 const& n)
        {
            double const h = nbref::dot(q[0], n);
            double const hh = std::abs(h);
            double I = 0.0;
            for(int i = 0; i < 4; ++i)
            {
                Vec3 const a = q[i];
                Vec3 const b = q[(i + 1) % 4];
                Vec3 const e = b - a;
                double const L = norm3(e);
                Vec3 const m = e / L;
                Vec3 const n2 = nbref::cross(m, n);
                double const d = nbref::dot(a, n2);
                double const la = nbref::dot(a, m);
                double const lb = nbref::dot(b, m);
                I += boxEdgePrimitive(lb, hh, d) - boxEdgePrimitive(la, hh, d);
            }
            Vec3 const foot = h * n;
            Vec3 u = q[1] - q[0];
            u = u / norm3(u);
            Vec3 const w = nbref::cross(n, u);
            bool inside = false;
            int j = 3;
            for(int i = 0; i < 4; ++i)
            {
                double const xi = nbref::dot(q[i] - foot, u);
                double const yi = nbref::dot(q[i] - foot, w);
                double const xj = nbref::dot(q[j] - foot, u);
                double const yj = nbref::dot(q[j] - foot, w);
                if(((yi > 0.0) != (yj > 0.0)) && (0.0 < (xj - xi) * (0.0 - yi) / (yj - yi) + xi))
                    inside = !inside;
                j = i;
            }
            if(inside)
                I -= 2.0 * kBoxPi * hh;
            return I;
        }


        // ------------------------------------------------------------------
        // Exact closed-form polynomial box field (Stage 1: oblique in-support)
        // ------------------------------------------------------------------
        //
        // Integral over one parallelepiped P = dc + { sum_a t_a e_a,
        // t_a in [-1/2,1/2] } of p(y) d/|d|^3, y = d - dc, p a polynomial of
        // total degree <= 6.  The volume recurrence (divergence theorem)
        // consumes *dc-centred* face moments F_beta(f) = int_face y^beta/|d| dA.
        // The foot-centred face monomials F_ab are only the primitive used to
        // evaluate those moments; the dc shift is carried by the expansion of
        // y = (h n - dc) + xi u + eta v.  This is what closes (4.2).

        //! Compact monomial bookkeeping for total degree <= 6 in 3 variables.
        struct PolyIdx
        {
            int idx[7][7][7];
            constexpr PolyIdx() : idx{}
            {
                int n = 0;
                for(int d = 0; d <= 6; ++d)
                    for(int a = 0; a <= d; ++a)
                        for(int b = 0; b <= d - a; ++b)
                        {
                            int const c = d - a - b;
                            idx[a][b][c] = n++;
                        }
            }
        };
        inline constexpr PolyIdx polyIdx{};
        static constexpr int kPolyTerms = 84;

        struct Poly6
        {
            double c[kPolyTerms];
        };

        inline void polyZero(Poly6& p)
        {
            for(int i = 0; i < kPolyTerms; ++i)
                p.c[i] = 0.0;
        }

        inline void polyAdd(Poly6& p, int a, int b, int c, double v)
        {
            if(a + b + c <= 6)
                p.c[polyIdx.idx[a][b][c]] += v;
        }

        //! r = p*q, total degree <= 6.
        inline void polyMul(Poly6 const& p, Poly6 const& q, Poly6& r)
        {
            polyZero(r);
            for(int a = 0; a <= 6; ++a)
                for(int b = 0; b <= 6 - a; ++b)
                    for(int c = 0; c <= 6 - a - b; ++c)
                    {
                        double const pv = p.c[polyIdx.idx[a][b][c]];
                        if(pv == 0.0)
                            continue;
                        for(int a2 = 0; a2 <= 6 - a; ++a2)
                            for(int b2 = 0; b2 <= 6 - a - a2; ++b2)
                                for(int c2 = 0; c2 <= 6 - a - a2 - b2; ++c2)
                                {
                                    double const qv = q.c[polyIdx.idx[a2][b2][c2]];
                                    if(qv == 0.0)
                                        continue;
                                    int const aa = a + a2;
                                    int const bb = b + b2;
                                    int const cc = c + c2;
                                    if(aa + bb + cc <= 6)
                                        r.c[polyIdx.idx[aa][bb][cc]] += pv * qv;
                                }
                    }
        }

        inline void bivarZero(double R[7][7])
        {
            for(int i = 0; i < 7; ++i)
                for(int j = 0; j < 7; ++j)
                    R[i][j] = 0.0;
        }

        //! R = (a + b xi + c eta)^n in the bivariate (xi,eta) basis, n <= 6.
        inline void linPowBivar(double a, double b, double c, int n, double R[7][7])
        {
            bivarZero(R);
            R[0][0] = 1.0;
            for(int s = 0; s < n; ++s)
            {
                double T[7][7];
                bivarZero(T);
                for(int i = 0; i <= 6; ++i)
                    for(int j = 0; j <= 6 - i; ++j)
                    {
                        double const v = R[i][j];
                        if(v == 0.0)
                            continue;
                        T[i][j] += v * a;
                        if(i + j < 6)
                        {
                            T[i + 1][j] += v * b;
                            T[i][j + 1] += v * c;
                        }
                    }
                for(int i = 0; i < 7; ++i)
                    for(int j = 0; j < 7; ++j)
                        R[i][j] = T[i][j];
            }
        }

        inline void bivarMul(double const A[7][7], double const B[7][7], double R[7][7])
        {
            bivarZero(R);
            for(int a = 0; a <= 6; ++a)
                for(int b = 0; b <= 6 - a; ++b)
                {
                    double const av = A[a][b];
                    if(av == 0.0)
                        continue;
                    for(int c = 0; c <= 6 - a - b; ++c)
                        for(int d = 0; d <= 6 - a - b - c; ++d)
                            R[a + c][b + d] += av * B[c][d];
                }
        }

        //! Coefficients of (a0 + a1 t)^n, n <= 5.
        inline void linPow1d(double a0, double a1, int n, double c[6])
        {
            for(int k = 0; k <= 5; ++k)
                c[k] = 0.0;
            for(int k = 0; k <= n; ++k)
            {
                double b = 1.0;
                for(int i = 0; i < k; ++i)
                    b *= static_cast<double>(n - i) / static_cast<double>(i + 1);
                double p0 = 1.0;
                for(int i = 0; i < n - k; ++i)
                    p0 *= a0;
                double p1 = 1.0;
                for(int i = 0; i < k; ++i)
                    p1 *= a1;
                c[k] = b * p0 * p1;
            }
        }

        //! W[k] = int_0^1 t^k rho(t) dt, rho = sqrt(A t^2 + B t + C), k = 0..5.
        inline void edgeWeights(
            double xi0,
            double eta0,
            double dxi,
            double deta,
            double Q,
            double W[6])
        {
            double const A = dxi * dxi + deta * deta;
            double const B = 2.0 * (xi0 * dxi + eta0 * deta);
            double const C = Q + xi0 * xi0 + eta0 * eta0;
            double const t0 = -B / (2.0 * A);
            double const D = xi0 * deta - eta0 * dxi;
            double s2 = (A * Q + D * D) / (A * A);
            if(s2 < 0.0)
                s2 = 0.0;
            double const sqA = std::sqrt(A);
            double const u0 = -t0;
            double const u1 = 1.0 - t0;
            double const xi1 = xi0 + dxi;
            double const eta1 = eta0 + deta;
            double const rho0 = std::sqrt(C);
            double const rho1 = std::sqrt(Q + xi1 * xi1 + eta1 * eta1);
            double I0;
            if(s2 > 0.0)
            {
                double const ss = std::sqrt(s2);
                I0 = 0.5 * (u1 * rho1 - u0 * rho0)
                    + 0.5 * s2 * sqA * (std::asinh(u1 / ss) - std::asinh(u0 / ss));
            }
            else
                I0 = 0.5 * (u1 * rho1 - u0 * rho0);
            W[0] = I0;
            W[1] = (rho1 * rho1 * rho1 - rho0 * rho0 * rho0) / (3.0 * A)
                - (B / (2.0 * A)) * W[0];
            for(int k = 2; k <= 5; ++k)
            {
                double const bnd = (rho1 * rho1 * rho1) / ((k + 2.0) * A);
                W[k] = bnd
                    - ((2.0 * k + 1.0) * B / (2.0 * (k + 2.0) * A)) * W[k - 1]
                    - ((k - 1.0) * C / ((k + 2.0) * A)) * W[k - 2];
            }
        }

        //! boxEdgePrimitive with the removable 0/0 (h=0, origin foot on the
        //! edge) taken as its limit zero.
        inline double boxEdgePrimitiveSafe(double l, double hh, double d)
        {
            if(hh == 0.0 && d == 0.0)
                return 0.0;
            return boxEdgePrimitive(l, hh, d);
        }

        //! boxFaceIntegral regularised for the degenerate face-through-origin
        //! edge (the original returns 0*inf there).
        inline double boxFaceIntegralSafe(Vec3 const q[4], Vec3 const& n)
        {
            double const h = nbref::dot(q[0], n);
            double const hh = std::abs(h);
            double I = 0.0;
            for(int i = 0; i < 4; ++i)
            {
                Vec3 const a = q[i];
                Vec3 const b = q[(i + 1) % 4];
                Vec3 const e = b - a;
                double const L = norm3(e);
                Vec3 const m = e / L;
                Vec3 const n2 = nbref::cross(m, n);
                double const d = nbref::dot(a, n2);
                double const la = nbref::dot(a, m);
                double const lb = nbref::dot(b, m);
                I += boxEdgePrimitiveSafe(lb, hh, d) - boxEdgePrimitiveSafe(la, hh, d);
            }
            Vec3 const foot = h * n;
            Vec3 u = q[1] - q[0];
            u = u / norm3(u);
            Vec3 const w = nbref::cross(n, u);
            bool inside = false;
            int j = 3;
            for(int i = 0; i < 4; ++i)
            {
                double const xi = nbref::dot(q[i] - foot, u);
                double const yi = nbref::dot(q[i] - foot, w);
                double const xj = nbref::dot(q[j] - foot, u);
                double const yj = nbref::dot(q[j] - foot, w);
                if(((yi > 0.0) != (yj > 0.0))
                   && (0.0 < (xj - xi) * (0.0 - yi) / (yj - yi) + xi))
                    inside = !inside;
                j = i;
            }
            if(inside)
                I -= 2.0 * kBoxPi * hh;
            return I;
        }

        struct FaceData
        {
            Vec3 n;
            double h = 0.0;
            double Q = 0.0;
            Vec3 u;
            Vec3 v;
            double Fab[7][7];
            //! Per-axis bivariate powers (w0_i + xi u_i + eta v_i)^n, n <= 6.
            double P[3][7][7][7];
        };

        //! Fill fd.P[axis][n] with the bivariate powers of the three linear
        //! forms whose product gives (d-dc)^nu on the face.
        inline void prepFacePowers(FaceData& fd, Vec3 const& dc)
        {
            double const w0[3] = {
                fd.h * fd.n.x - dc.x,
                fd.h * fd.n.y - dc.y,
                fd.h * fd.n.z - dc.z};
            double lin[3][3];
            for(int i = 0; i < 3; ++i)
            {
                lin[i][0] = w0[i];
                lin[i][1] = fd.u[i];
                lin[i][2] = fd.v[i];
            }
            for(int i = 0; i < 3; ++i)
            {
                double R[7][7];
                bivarZero(R);
                R[0][0] = 1.0;
                for(int n = 0; n <= 6; ++n)
                {
                    for(int a = 0; a <= 6; ++a)
                        for(int b = 0; b <= 6 - a; ++b)
                            fd.P[i][n][a][b] = R[a][b];
                    if(n == 6)
                        break;
                    double T[7][7];
                    bivarZero(T);
                    for(int a = 0; a <= 6; ++a)
                        for(int b = 0; b <= 6 - a; ++b)
                        {
                            double const val = R[a][b];
                            if(val == 0.0)
                                continue;
                            T[a][b] += val * lin[i][0];
                            if(a + b < 6)
                            {
                                T[a + 1][b] += val * lin[i][1];
                                T[a][b + 1] += val * lin[i][2];
                            }
                        }
                    for(int a = 0; a < 7; ++a)
                        for(int b = 0; b < 7; ++b)
                            R[a][b] = T[a][b];
                }
            }
        }

        //! Foot-centred F_ab = int_face xi^a eta^b / rho dA, a+b <= 6.
        inline void faceMonomials(Vec3 const q[4], Vec3 const& n, double h, double Fab[7][7])
        {
            for(int a = 0; a < 7; ++a)
                for(int b = 0; b < 7; ++b)
                    Fab[a][b] = 0.0;
            double const Q = h * h;
            Vec3 const foot = h * n;
            Vec3 u = q[1] - q[0];
            u = u / norm3(u);
            Vec3 const v = nbref::cross(n, u);
            double xi[4], eta[4];
            for(int l = 0; l < 4; ++l)
            {
                Vec3 const w = q[l] - foot;
                xi[l] = nbref::dot(w, u);
                eta[l] = nbref::dot(w, v);
            }
            Fab[0][0] = boxFaceIntegralSafe(q, n);

            double U[7][7];
            double V[7][7];
            for(int a = 0; a < 7; ++a)
                for(int b = 0; b < 7; ++b)
                {
                    U[a][b] = 0.0;
                    V[a][b] = 0.0;
                }
            for(int l = 0; l < 4; ++l)
            {
                int const m = (l + 1) % 4;
                double const x0 = xi[l];
                double const y0 = eta[l];
                double const dx = xi[m] - xi[l];
                double const dy = eta[m] - eta[l];
                double W[6];
                edgeWeights(x0, y0, dx, dy, Q, W);
                for(int a = 1; a <= 6; ++a)
                    for(int b = 0; b <= 6 - a; ++b)
                    {
                        double c1[6], c2[6], c[6];
                        linPow1d(x0, dx, a - 1, c1);
                        linPow1d(y0, dy, b, c2);
                        for(int k = 0; k <= 5; ++k)
                            c[k] = 0.0;
                        for(int i = 0; i <= 5; ++i)
                            for(int j = 0; i + j <= 5; ++j)
                                c[i + j] += c1[i] * c2[j];
                        double s = 0.0;
                        for(int k = 0; k <= 5; ++k)
                            s += c[k] * W[k];
                        U[a][b] += dy * s;
                    }
                for(int b = 1; b <= 6; ++b)
                    for(int a = 0; a <= 6 - b; ++a)
                    {
                        double c1[6], c2[6], c[6];
                        linPow1d(x0, dx, a, c1);
                        linPow1d(y0, dy, b - 1, c2);
                        for(int k = 0; k <= 5; ++k)
                            c[k] = 0.0;
                        for(int i = 0; i <= 5; ++i)
                            for(int j = 0; i + j <= 5; ++j)
                                c[i + j] += c1[i] * c2[j];
                        double s = 0.0;
                        for(int k = 0; k <= 5; ++k)
                            s += c[k] * W[k];
                        V[a][b] += -dx * s;
                    }
            }
            for(int deg = 1; deg <= 6; ++deg)
            {
                if(deg == 1)
                    Fab[0][1] = V[0][1];
                else
                    Fab[0][deg] = (2.0 * V[0][deg]
                                   - (deg - 1.0) * (Q * Fab[0][deg - 2] + U[2][deg - 2]))
                        / (deg + 1.0);
                for(int a = 2; a <= deg; a += 2)
                    Fab[a][deg - a] = (U[a][deg - a]
                                       - (a - 1.0)
                                           * (Q * Fab[a - 2][deg - a]
                                              + Fab[a - 2][deg - a + 2]))
                        / a;
                Fab[1][deg - 1] = U[1][deg - 1];
                for(int a = 3; a <= deg; a += 2)
                    Fab[a][deg - a] = (U[a][deg - a]
                                       - (a - 1.0)
                                           * (Q * Fab[a - 2][deg - a]
                                              + Fab[a - 2][deg - a + 2]))
                        / a;
            }
        }

        //! int_face (d-dc)^nu / |d| dA in foot-centred F_ab.
        inline double faceMono(FaceData const& fd, int const nu[3])
        {
            double T[7][7], R[7][7];
            bivarMul(fd.P[0][nu[0]], fd.P[1][nu[1]], T);
            bivarMul(T, fd.P[2][nu[2]], R);
            double sum = 0.0;
            for(int a = 0; a <= 6; ++a)
                for(int b = 0; b <= 6 - a; ++b)
                    sum += R[a][b] * fd.Fab[a][b];
            return sum;
        }

        //! Build the six outward face planes of P = dc +- 0.5 e_a.
        inline void polyBoxFaces(Vec3 const& dc, Vec3 const e[3], Vec3 q[6][4], Vec3 nrm[6], double h[6])
        {
            Vec3 V[2][2][2];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        V[a][b][c] = dc + (a ? 0.5 : -0.5) * e[0]
                            + (b ? 0.5 : -0.5) * e[1] + (c ? 0.5 : -0.5) * e[2];
            static int const FI[6][4][3] = {
                {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
            for(int f = 0; f < 6; ++f)
            {
                for(int v = 0; v < 4; ++v)
                    q[f][v] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]];
                Vec3 nn = nbref::cross(q[f][1] - q[f][0], q[f][2] - q[f][0]);
                nn = nn / norm3(nn);
                Vec3 const cen = (q[f][0] + q[f][1] + q[f][2] + q[f][3]) / 4.0;
                if(nbref::dot(cen - dc, nn) < 0.0)
                {
                    Vec3 const t0 = q[f][0], t1 = q[f][1], t2 = q[f][2], t3 = q[f][3];
                    q[f][0] = t3;
                    q[f][1] = t2;
                    q[f][2] = t1;
                    q[f][3] = t0;
                    nn = nbref::cross(q[f][1] - q[f][0], q[f][2] - q[f][0]);
                    nn = nn / norm3(nn);
                }
                nrm[f] = nn;
                h[f] = nbref::dot(q[f][0], nn);
            }
        }

        //! Accumulate dc-centred h_f F_beta(f) and the vector sum n_f F_beta(f).
        inline void faceMomentSums(
            Vec3 const& dc,
            Vec3 const e[3],
            double hF[kPolyTerms],
            Vec3 nF[kPolyTerms])
        {
            for(int i = 0; i < kPolyTerms; ++i)
            {
                hF[i] = 0.0;
                nF[i] = Vec3{};
            }
            Vec3 q[6][4];
            Vec3 nrm[6];
            double h[6];
            polyBoxFaces(dc, e, q, nrm, h);
            for(int f = 0; f < 6; ++f)
            {
                FaceData fd;
                fd.n = nrm[f];
                fd.h = h[f];
                fd.Q = h[f] * h[f];
                Vec3 u = q[f][1] - q[f][0];
                fd.u = u / norm3(u);
                fd.v = nbref::cross(fd.n, fd.u);
                faceMonomials(q[f], fd.n, fd.h, fd.Fab);
                prepFacePowers(fd, dc);
                for(int a = 0; a <= 6; ++a)
                    for(int b = 0; b <= 6 - a; ++b)
                        for(int c = 0; c <= 6 - a - b; ++c)
                        {
                            int const nu[3] = {a, b, c};
                            double const Fm = faceMono(fd, nu);
                            int const id = polyIdx.idx[a][b][c];
                            hF[id] += fd.h * Fm;
                            nF[id] = nF[id] + fd.n * Fm;
                        }
            }
        }

        //! I_beta = int_P (d-dc)^beta/|d| dV for |beta| <= 5 (components <= 2).
        inline void volumeFromFaceMoments(Vec3 const& dc, double const hF[kPolyTerms], double I[kPolyTerms])
        {
            for(int i = 0; i < kPolyTerms; ++i)
                I[i] = 0.0;
            for(int deg = 0; deg <= 5; ++deg)
                for(int a = 0; a <= deg; ++a)
                    for(int b = 0; b <= deg - a; ++b)
                    {
                        int const c = deg - a - b;
                        int const id = polyIdx.idx[a][b][c];
                        double val;
                        if(deg == 0)
                            val = 0.5 * hF[id];
                        else
                        {
                            double sub = 0.0;
                            if(a > 0)
                                sub += a * dc.x * I[polyIdx.idx[a - 1][b][c]];
                            if(b > 0)
                                sub += b * dc.y * I[polyIdx.idx[a][b - 1][c]];
                            if(c > 0)
                                sub += c * dc.z * I[polyIdx.idx[a][b][c - 1]];
                            val = (hF[id] - sub) / (deg + 2.0);
                        }
                        I[id] = val;
                    }
        }

        //! Exposed for the I_100@dc!=0 unit test.
        inline void boxVolumeI(Vec3 const& dc, Vec3 const e[3], double Iout[3][3][3])
        {
            double hF[kPolyTerms];
            Vec3 nF[kPolyTerms];
            faceMomentSums(dc, e, hF, nF);
            double I[kPolyTerms];
            volumeFromFaceMoments(dc, hF, I);
            for(int a = 0; a <= 2; ++a)
                for(int b = 0; b <= 2; ++b)
                    for(int c = 0; c <= 2; ++c)
                        Iout[a][b][c] = I[polyIdx.idx[a][b][c]];
        }

        //! J_j = int_P p(y) d_j/|d|^3 dV, y = d - dc, box dc +- 0.5 e_a.
        inline void polyJ(Vec3 const& dc, Vec3 const e[3], Poly6 const& p, Vec3& J)
        {
            double hF[kPolyTerms];
            Vec3 nF[kPolyTerms];
            faceMomentSums(dc, e, hF, nF);
            double I[kPolyTerms];
            volumeFromFaceMoments(dc, hF, I);
            J = Vec3{};
            for(int a = 0; a <= 6; ++a)
                for(int b = 0; b <= 6 - a; ++b)
                    for(int c = 0; c <= 6 - a - b; ++c)
                    {
                        int const id = polyIdx.idx[a][b][c];
                        double const av = p.c[id];
                        if(av == 0.0)
                            continue;
                        Vec3 vol{};
                        if(a > 0)
                            vol.x = a * I[polyIdx.idx[a - 1][b][c]];
                        if(b > 0)
                            vol.y = b * I[polyIdx.idx[a][b - 1][c]];
                        if(c > 0)
                            vol.z = c * I[polyIdx.idx[a][b][c - 1]];
                        J = J + (nF[id] * (-1.0) + vol) * av;
                    }
        }

        //! hF-only face moments for |beta| <= 5 (the volume recurrence in polyJ
        //! never needs degree 6).  The -nF surface term is omitted: summed over
        //! the 27 continuous TSC pieces it cancels exactly (p vanishes on the
        //! outer boundary), so polyBoxField can drop it.
        inline void faceMomentSumsH(
            Vec3 const& dc,
            Vec3 const e[3],
            double hF[kPolyTerms])
        {
            for(int i = 0; i < kPolyTerms; ++i)
                hF[i] = 0.0;
            Vec3 q[6][4];
            Vec3 nrm[6];
            double h[6];
            polyBoxFaces(dc, e, q, nrm, h);
            for(int f = 0; f < 6; ++f)
            {
                FaceData fd;
                fd.n = nrm[f];
                fd.h = h[f];
                fd.Q = h[f] * h[f];
                Vec3 u = q[f][1] - q[f][0];
                fd.u = u / norm3(u);
                fd.v = nbref::cross(fd.n, fd.u);
                faceMonomials(q[f], fd.n, fd.h, fd.Fab);
                prepFacePowers(fd, dc);
                for(int a = 0; a <= 5; ++a)
                    for(int b = 0; b <= 5 - a; ++b)
                        for(int c = 0; c <= 5 - a - b; ++c)
                        {
                            int const nu[3] = {a, b, c};
                            double const Fm = faceMono(fd, nu);
                            int const id = polyIdx.idx[a][b][c];
                            hF[id] += fd.h * Fm;
                        }
            }
        }

        //! Volume part of J only (no surface term); see faceMomentSumsH.
        inline void polyJVol(Vec3 const& dc, Vec3 const e[3], Poly6 const& p, Vec3& J)
        {
            double hF[kPolyTerms];
            faceMomentSumsH(dc, e, hF);
            double I[kPolyTerms];
            volumeFromFaceMoments(dc, hF, I);
            J = Vec3{};
            for(int a = 0; a <= 6; ++a)
                for(int b = 0; b <= 6 - a; ++b)
                    for(int c = 0; c <= 6 - a - b; ++c)
                    {
                        int const id = polyIdx.idx[a][b][c];
                        double const av = p.c[id];
                        if(av == 0.0)
                            continue;
                        Vec3 vol{};
                        if(a > 0)
                            vol.x = a * I[polyIdx.idx[a - 1][b][c]];
                        if(b > 0)
                            vol.y = b * I[polyIdx.idx[a][b - 1][c]];
                        if(c > 0)
                            vol.z = c * I[polyIdx.idx[a][b][c - 1]];
                        J = J + vol * av;
                    }
        }

        //! One box in source-index space: s in s_c +- halfwidth, d = M(R0 - s*cells).
        inline void boxPolyJ(
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u,
            Poly6 const& p,
            Vec3 const& sc,
            Vec3 const& halfwidth,
            Vec3& J)
        {
            double const uu2 = nbref::dot(u, u);
            Vec3 dc;
            Vec3 e[3];
            Vec3 const scCells{sc.x * cells.x, sc.y * cells.y, sc.z * cells.z};
            if(uu2 == 0.0)
            {
                dc = R0 - scCells;
                e[0] = Vec3{2.0 * halfwidth.x * cells.x, 0.0, 0.0};
                e[1] = Vec3{0.0, 2.0 * halfwidth.y * cells.y, 0.0};
                e[2] = Vec3{0.0, 0.0, 2.0 * halfwidth.z * cells.z};
            }
            else
            {
                double const gam = std::sqrt(1.0 + uu2);
                Vec3 const n = u / std::sqrt(uu2);
                auto const M = [&](Vec3 const& v) { return applyContract(v, n, 1.0 / gam); };
                dc = M(R0 - scCells);
                e[0] = M(Vec3{2.0 * halfwidth.x * cells.x, 0.0, 0.0});
                e[1] = M(Vec3{0.0, 2.0 * halfwidth.y * cells.y, 0.0});
                e[2] = M(Vec3{0.0, 0.0, 2.0 * halfwidth.z * cells.z});
            }
            polyJ(dc, e, p, J);
        }

        //! One axis factor q_k(s_a(y)) as a Poly6.
        inline void axisQuadratic(
            double Ac,
            double Bc,
            double Cc,
            double sca,
            double cell,
            double gam,
            Vec3 const& n,
            int axis,
            Poly6& q)
        {
            polyZero(q);
            double const l0 = sca;
            double l[3];
            for(int m = 0; m < 3; ++m)
            {
                double const d = (m == axis ? 1.0 : 0.0);
                l[m] = -(gam * d + (1.0 - gam) * n[axis] * n[m]) / cell;
            }
            polyAdd(q, 0, 0, 0, Ac * l0 * l0 + Bc * l0 + Cc);
            for(int m = 0; m < 3; ++m)
            {
                int ex[3] = {0, 0, 0};
                ex[m] = 1;
                polyAdd(q, ex[0], ex[1], ex[2], (2.0 * Ac * l0 + Bc) * l[m]);
            }
            for(int m = 0; m < 3; ++m)
                for(int k = 0; k < 3; ++k)
                {
                    int ex[3] = {0, 0, 0};
                    ex[m] += 1;
                    ex[k] += 1;
                    if(m < k)
                        polyAdd(q, ex[0], ex[1], ex[2], 2.0 * Ac * l[m] * l[k]);
                    else if(m == k)
                        polyAdd(q, ex[0], ex[1], ex[2], Ac * l[m] * l[k]);
                }
        }

        // ------------------------------------------------------------------
        // Exact source-frame (s-space) scalar-potential moments.
        // ------------------------------------------------------------------
        // The TSC density in source-index space is a tensor product of three
        // per-axis quadratics, i.e. 27 monomials s^beta with beta_a <= 2.  For
        // one cell B (image of a unit index cube, so an axis-aligned physical
        // box) the field is
        //
        //   E = (1/V) G^{-1} [ int_B f grad_R p dR - oint p f n dA ],
        //   f = 1/|M R|,  G = I + (gamma^2-1) n n^T = gamma^2 M^2.
        //
        // Both pieces reduce to face integrals of f times bivariate
        // polynomials of degree <= 4.  The volume term uses the divergence
        // identity  div(R^gamma R f) = (|gamma|+2) R^gamma f  on each
        // axis-aligned cell.  Faces are whitened to |z|^2 + qmin and evaluated
        // with the verified Euclidean face recurrence (degree 4).  Over the
        // full support the p-weighted surface terms cancel exactly (p is
        // continuous and vanishes on the outer boundary), so the 27-cell sum
        // only needs the volume term.
        inline double sIp(double x, int n)
        {
            double r = 1.0;
            for(int i = 0; i < n; ++i)
                r *= x;
            return r;
        }

        //! W[k] = int_0^1 t^k rho(t) dt, rho = sqrt(A t^2+B t+C), k <= 3.
        //! A and sqrt(A) are hoisted by the caller: opposite edges of the
        //! whitened parallelogram share them.  Only k <= 3 is consumed by the
        //! degree-4 face moments sFaceM needs.
        inline void sEdgeW4(
            double xi0,
            double eta0,
            double dxi,
            double deta,
            double Q,
            double A,
            double invA,
            double sqA,
            double W[4])
        {
            double const inv2A = 0.5 * invA;
            double const B = 2.0 * (xi0 * dxi + eta0 * deta);
            double const C = Q + xi0 * xi0 + eta0 * eta0;
            double const t0 = -B * inv2A;
            double const Dd = xi0 * deta - eta0 * dxi;
            double s2 = (A * Q + Dd * Dd) / (A * A);
            if(s2 < 0.0)
                s2 = 0.0;
            double const u0 = -t0;
            double const u1 = 1.0 - t0;
            double const xi1 = xi0 + dxi;
            double const eta1 = eta0 + deta;
            double const rho0 = std::sqrt(C);
            double const rho1 = std::sqrt(Q + xi1 * xi1 + eta1 * eta1);
            double I0;
            if(s2 > 0.0)
            {
                double const ss = std::sqrt(s2);
                I0 = 0.5 * (u1 * rho1 - u0 * rho0)
                    + 0.5 * s2 * sqA * (std::asinh(u1 / ss) - std::asinh(u0 / ss));
            }
            else
                I0 = 0.5 * (u1 * rho1 - u0 * rho0);
            double const r1c = rho1 * rho1 * rho1;
            W[0] = I0;
            W[1] = (r1c - rho0 * rho0 * rho0) * (invA / 3.0) - (B * inv2A) * W[0];
            for(int k = 2; k <= 3; ++k)
            {
                double const ik = invA / (k + 2.0);
                W[k] = r1c * ik
                    - ((2.0 * k + 1.0) * B * inv2A / (k + 2.0)) * W[k - 1]
                    - ((k - 1.0) * C) * ik * W[k - 2];
            }
        }

        //! Foot-centred face moments F_ab = int_face xi^a eta^b / rho dA for
        //! a <= 2, a+b <= 4: exactly what sFaceM consumes.  The parallelogram
        //! edge contributions are assembled straight from the signed edge
        //! moments MY/MW instead of filling the full U/V tensors.
        inline void sFaceFab4(Vec3 const q[4], Vec3 const& n, double h, double Fab[5][5])
        {
            for(int a = 0; a <= 2; ++a)
                for(int b = 0; b <= 4; ++b)
                    Fab[a][b] = 0.0;
            double const Q = h * h;
            Vec3 const foot = h * n;
            Vec3 u = q[1] - q[0];
            u = u / std::sqrt(nbref::dot(u, u));
            Vec3 const v = nbref::cross(n, u);
            double xi[4], eta[4];
            for(int l = 0; l < 4; ++l)
            {
                Vec3 const w = q[l] - foot;
                xi[l] = nbref::dot(w, u);
                eta[l] = nbref::dot(w, v);
            }
            double A[4], invA[4];
            for(int l = 0; l < 4; ++l)
            {
                int const m = (l + 1) % 4;
                double const dx = xi[m] - xi[l], dy = eta[m] - eta[l];
                A[l] = dx * dx + dy * dy;
                invA[l] = 1.0 / A[l];
            }
            double sqA[4];
            sqA[0] = std::sqrt(A[0]);
            sqA[1] = std::sqrt(A[1]);
            sqA[2] = sqA[0];
            sqA[3] = sqA[1];
            {
                double const hh = std::abs(h);
                double Ibase = 0.0;
                for(int l = 0; l < 4; ++l)
                {
                    int const m = (l + 1) % 4;
                    double const ex = xi[m] - xi[l], ey = eta[m] - eta[l];
                    double const invL = 1.0 / std::sqrt(ex * ex + ey * ey);
                    double const mx = ex * invL, my = ey * invL;
                    double const d = xi[l] * my - eta[l] * mx;
                    double const la = xi[l] * mx + eta[l] * my;
                    double const lb = la + 1.0 / invL;
                    Ibase += boxEdgePrimitive(lb, hh, d) - boxEdgePrimitive(la, hh, d);
                }
                bool inside = false;
                int j = 3;
                for(int i = 0; i < 4; ++i)
                {
                    double const xiI = xi[i], yi = eta[i];
                    double const xj = xi[j], yj = eta[j];
                    if(((yi > 0.0) != (yj > 0.0))
                       && (0.0 < (xj - xiI) * (0.0 - yi) / (yj - yi) + xiI))
                        inside = !inside;
                    j = i;
                }
                if(inside)
                    Ibase -= 2.0 * kBoxPi * hh;
                Fab[0][0] = Ibase;
            }
            double U1[4] = {0.0, 0.0, 0.0, 0.0};
            double U2[3] = {0.0, 0.0, 0.0};
            double V0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for(int l = 0; l < 4; ++l)
            {
                int const m = (l + 1) % 4;
                double const xa = xi[l], ya = eta[l];
                double const dx = xi[m] - xi[l], dy = eta[m] - eta[l];
                double W[4];
                sEdgeW4(xa, ya, dx, dy, Q, A[l], invA[l], sqA[l], W);
                double PY[4][4] = {};
                PY[0][0] = 1.0;
                for(int nn = 1; nn <= 3; ++nn)
                    for(int k = 0; k <= nn; ++k)
                        PY[nn][k] = (k <= nn - 1 ? PY[nn - 1][k] * ya : 0.0)
                            + (k >= 1 ? PY[nn - 1][k - 1] * dy : 0.0);
                double MY[4], MW[3];
                for(int nn = 0; nn <= 3; ++nn)
                {
                    double sum = 0.0;
                    for(int j = 0; j <= nn; ++j)
                        sum += PY[nn][j] * W[j];
                    MY[nn] = sum;
                }
                for(int nn = 0; nn <= 2; ++nn)
                {
                    double sum = 0.0;
                    for(int j = 0; j <= nn; ++j)
                        sum += PY[nn][j] * W[j + 1];
                    MW[nn] = sum;
                }
                for(int b = 0; b <= 3; ++b)
                    U1[b] += dy * MY[b];
                for(int b = 0; b <= 2; ++b)
                    U2[b] += dy * (xa * MY[b] + dx * MW[b]);
                for(int b = 1; b <= 4; ++b)
                    V0[b] += -dx * MY[b - 1];
            }
            for(int deg = 1; deg <= 4; ++deg)
            {
                if(deg == 1)
                    Fab[0][1] = V0[1];
                else
                    Fab[0][deg] = (2.0 * V0[deg]
                                   - (deg - 1.0) * (Q * Fab[0][deg - 2] + U2[deg - 2]))
                        / (deg + 1.0);
                Fab[1][deg - 1] = U1[deg - 1];
                if(deg >= 2)
                    Fab[2][deg - 2]
                        = (U2[deg - 2] - (Q * Fab[0][deg - 2] + Fab[0][deg])) / 2.0;
            }
        }

        //! Per-pair face geometry of one box axis (independent of the cell).
        struct SAxis
        {
            int a, b, c;
            double Hxx, Hxy, Hyy, detH;
            double L11, L21, L22, detL;
            double ax, bx, by;
            double uhx, uhy, qc;
        };

        inline void sPrepAxis(SAxis& X, int a, Vec3 const& n, double s)
        {
            int const b = (a + 1) % 3;
            int const c = (a + 2) % 3;
            double G[3][3];
            double const nn[3] = {n.x, n.y, n.z};
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    G[i][j] = (i == j ? 1.0 : 0.0) + s * nn[i] * nn[j];
            X.a = a;
            X.b = b;
            X.c = c;
            X.Hxx = G[b][b];
            X.Hxy = G[b][c];
            X.Hyy = G[c][c];
            X.detH = X.Hxx * X.Hyy - X.Hxy * X.Hxy;
            double const gab = G[a][b], gac = G[a][c];
            X.uhx = -(X.Hyy * gab - X.Hxy * gac) / X.detH;
            X.uhy = -(-X.Hxy * gab + X.Hxx * gac) / X.detH;
            X.qc = G[a][a] + gab * X.uhx + gac * X.uhy;
            X.L11 = std::sqrt(X.Hxx);
            X.L21 = X.Hxy / X.L11;
            X.L22 = std::sqrt(X.Hyy - X.L21 * X.L21);
            X.detL = X.L11 * X.L22;
            X.ax = 1.0 / X.L11;
            X.bx = -X.L21 / (X.L11 * X.L22);
            X.by = 1.0 / X.L22;
        }

        //! M[p][q] = int_face R_b^p R_c^q / |M R| dA for one face of the cell.
        inline void sFaceM(
            SAxis const& X,
            double h,
            Vec3 const& Rc,
            Vec3 const& C,
            double gam,
            double M[3][3])
        {
            double const qmin = h * h * X.qc;
            double const hh = std::sqrt(qmin > 0.0 ? qmin : 0.0);
            double const u0x = h * X.uhx, u0y = h * X.uhy;
            int const b = X.b, c = X.c;
            double const xlo = Rc[b] - 0.5 * C[b], xhi = Rc[b] + 0.5 * C[b];
            double const ylo = Rc[c] - 0.5 * C[c], yhi = Rc[c] + 0.5 * C[c];
            double const xy[4][2] = {{xlo, ylo}, {xhi, ylo}, {xhi, yhi}, {xlo, yhi}};
            Vec3 qz[4];
            for(int i = 0; i < 4; ++i)
            {
                double const dx = xy[i][0] - u0x, dy = xy[i][1] - u0y;
                qz[i] = Vec3{X.L11 * dx + X.L21 * dy, X.L22 * dy, hh};
            }
            double Fab[5][5];
            sFaceFab4(qz, Vec3{0.0, 0.0, 1.0}, hh, Fab);
            double const ax = X.ax, bx = X.bx, by = X.by;
            double const ux[3] = {1.0, u0x, u0x * u0x};
            double const uy[3] = {1.0, u0y, u0y * u0y};
            double const Ax[3] = {1.0, ax, ax * ax};
            double const Bx[3] = {1.0, bx, bx * bx};
            double const By[3] = {1.0, by, by * by};
            static int const CB[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
            for(int p = 0; p <= 2; ++p)
                for(int q = 0; q <= 2; ++q)
                {
                    double sum = 0.0;
                    for(int i = 0; i <= p; ++i)
                    {
                        double const ci = CB[p][i] * ux[p - i];
                        for(int k = 0; k <= i; ++k)
                        {
                            double const ck = CB[i][k] * Ax[i - k] * Bx[k];
                            for(int j = 0; j <= q; ++j)
                                sum += ci * ck * (CB[q][j] * uy[q - j] * By[j])
                                    * Fab[i - k][k + j];
                        }
                    }
                    M[p][q] = gam / X.detL * sum;
                }
        }

        //! Exact in-support TSC field in source frame (27 cells, 27 monomials).
        //! Internal knot planes are shared by adjacent cells: the face moments
        //! Distance below which a source-index coordinate is treated as a knot.
        inline constexpr double kKnotDetect = 1.0e-6;
        //! One-sided offset used by the exact evaluator on a knot plane.  The
        //! s-space field is continuous across every TSC knot, but the face
        //! recurrence divides by the origin-to-face distance and is singular
        //! there; the one-sided limit removes the removable degeneracy.
        inline constexpr double kKnotEvalSnapLowGamma = 1.0e-9;
        inline constexpr double kKnotEvalSnapHighGamma = 1.0e-7;
        inline constexpr double kKnotHighGammaThreshold = 20.0;

        inline double knotEvalSnap(double const gamma)
        {
            return gamma >= kKnotHighGammaThreshold ? kKnotEvalSnapHighGamma
                                                    : kKnotEvalSnapLowGamma;
        }
        //! Gate-only offset: the conditioning check rejects points closer than
        //! 1e-6 to a knot, so the gate nudges knot geometries past that guard.
        inline constexpr double kKnotGateSnap = 1.0e-5;

        inline double snapKnotCoordinate(double const s, double const snap)
        {
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};
            for(int k = 0; k < 4; ++k)
                if(std::abs(s - knots[k]) < kKnotDetect)
                    return knots[k] + (knots[k] > 0.0 ? -snap : snap);
            return s;
        }

        inline Vec3 snapKnotOrigin(Vec3 const& R0, Vec3 const& cells, double const snap)
        {
            return Vec3{
                cells.x * snapKnotCoordinate(R0.x / cells.x, snap),
                cells.y * snapKnotCoordinate(R0.y / cells.y, snap),
                cells.z * snapKnotCoordinate(R0.z / cells.z, snap)};
        }

        //! M[p][q] depend only on geometry, so they are evaluated once per
        //! 3x4x3x3 = 108 unique face-plane/transverse-cell combination instead
        //! of once per cell (162 evaluations).
        inline Vec3 polyBoxFieldS(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = uu2 == 0.0 ? Vec3{} : u / std::sqrt(uu2);
            double const s = gam * gam - 1.0;
            SAxis AX[3];
            for(int a = 0; a < 3; ++a)
                sPrepAxis(AX[a], a, n, s);
            static double const qA[3] = {0.5, -1.0, 0.5};
            static double const qB[3] = {1.5, 0.0, -1.5};
            static double const qC[3] = {1.125, 0.75, 1.125};
            Vec3 const Ra = snapKnotOrigin(R0, cells, knotEvalSnap(gam));
            double const rr[3] = {Ra.x, Ra.y, Ra.z};
            double const cc[3] = {cells.x, cells.y, cells.z};
            // Shared face moments: [axis][plane 0..3][sb+1][sc+1][p][q].
            double Mf[3][4][3][3][3][3];
            double hf[3][4];
            for(int a = 0; a < 3; ++a)
            {
                int const b = (a + 1) % 3, c = (a + 2) % 3;
                for(int pl = 0; pl < 4; ++pl)
                {
                    double const sp = -1.5 + static_cast<double>(pl);
                    double const h = rr[a] - cc[a] * sp;
                    hf[a][pl] = h;
                    for(int sb = -1; sb <= 1; ++sb)
                        for(int sc = -1; sc <= 1; ++sc)
                        {
                            Vec3 Rc{};
                            (&Rc.x)[b] = rr[b] - cc[b] * static_cast<double>(sb);
                            (&Rc.x)[c] = rr[c] - cc[c] * static_cast<double>(sc);
                            sFaceM(AX[a], h, Rc, cells, gam, Mf[a][pl][sb + 1][sc + 1]);
                        }
                }
            }
            Vec3 Iv{};
            for(int sx = -1; sx <= 1; ++sx)
                for(int sy = -1; sy <= 1; ++sy)
                    for(int sz = -1; sz <= 1; ++sz)
                    {
                        int const sc[3] = {sx, sy, sz};
                        double pc[3][3];
                        for(int a = 0; a < 3; ++a)
                        {
                            int const k = sc[a] + 1;
                            double const ic = 1.0 / cc[a], t = rr[a] * ic;
                            pc[a][0] = qA[k] * t * t + qB[k] * t + qC[k];
                            pc[a][1] = -2.0 * qA[k] * t * ic - qB[k] * ic;
                            pc[a][2] = qA[k] * ic * ic;
                        }
                        double pB[27];
                        for(int i = 0; i < 3; ++i)
                            for(int j = 0; j < 3; ++j)
                                for(int k = 0; k < 3; ++k)
                                    pB[i * 9 + j * 3 + k] = pc[0][i] * pc[1][j] * pc[2][k];
                        double Wt[27] = {};
                        for(int a = 0; a < 3; ++a)
                        {
                            int const b = (a + 1) % 3, c = (a + 2) % 3;
                            for(int sg = 0; sg < 2; ++sg)
                            {
                                int const pl = (sg == 0) ? (sc[a] + 2) : (sc[a] + 1);
                                double const h = hf[a][pl];
                                double const hp[3] = {h, h * h, h * h * h};
                                double const sign = sg ? 1.0 : -1.0;
                                double const(*M)[3] = Mf[a][pl][sc[b] + 1][sc[c] + 1];
                                for(int i = 0; i < 3; ++i)
                                    for(int j = 0; j < 3; ++j)
                                        for(int k = 0; k < 3; ++k)
                                        {
                                            int const be[3] = {i, j, k};
                                            Wt[i * 9 + j * 3 + k]
                                                += sign * hp[be[a]] * M[be[b]][be[c]];
                                        }
                            }
                        }
                        for(int a = 0; a < 3; ++a)
                        {
                            double acc = 0.0;
                            for(int i = 0; i < 3; ++i)
                                for(int j = 0; j < 3; ++j)
                                    for(int k = 0; k < 3; ++k)
                                    {
                                        int const be[3] = {i, j, k};
                                        double const pv = pB[i * 9 + j * 3 + k];
                                        if(be[a] == 0 || pv == 0.0)
                                            continue;
                                        int g[3] = {i, j, k};
                                        g[a]--;
                                        int const dg = g[0] + g[1] + g[2];
                                        double const W
                                            = Wt[g[0] * 9 + g[1] * 3 + g[2]] / (dg + 2.0);
                                        acc += be[a] * pv * W;
                                    }
                            (&Iv.x)[a] += acc;
                        }
                    }
            Vec3 const vn = nbref::dot(Iv, n) * n;
            Vec3 const E = ((Iv - vn) + vn / (gam * gam))
                * (1.0 / (cells.x * cells.y * cells.z));
            return E;
        }

        //! Reference d-space path (kept for equivalence tests).
        inline Vec3 polyBoxFieldD(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = uu2 == 0.0 ? Vec3{} : u / std::sqrt(uu2);
            auto const M = [&](Vec3 const& v) { return applyContract(v, n, 1.0 / gam); };
            static double const qA[3] = {0.5, -1.0, 0.5};
            static double const qB[3] = {1.5, 0.0, -1.5};
            static double const qC[3] = {1.125, 0.75, 1.125};
            Vec3 Jtot{};
            for(int sx = -1; sx <= 1; ++sx)
                for(int sy = -1; sy <= 1; ++sy)
                    for(int sz = -1; sz <= 1; ++sz)
                    {
                        int const kx = sx + 1;
                        int const ky = sy + 1;
                        int const kz = sz + 1;
                        Poly6 qx, qy, qz, qxy, p;
                        axisQuadratic(qA[kx], qB[kx], qC[kx], sx, cells.x, gam, n, 0, qx);
                        axisQuadratic(qA[ky], qB[ky], qC[ky], sy, cells.y, gam, n, 1, qy);
                        axisQuadratic(qA[kz], qB[kz], qC[kz], sz, cells.z, gam, n, 2, qz);
                        polyMul(qx, qy, qxy);
                        polyMul(qxy, qz, p);
                        Vec3 const scCells{sx * cells.x, sy * cells.y, sz * cells.z};
                        Vec3 const dc = M(R0 - scCells);
                        Vec3 e[3];
                        if(uu2 == 0.0)
                        {
                            e[0] = Vec3{cells.x, 0.0, 0.0};
                            e[1] = Vec3{0.0, cells.y, 0.0};
                            e[2] = Vec3{0.0, 0.0, cells.z};
                        }
                        else
                        {
                            e[0] = M(Vec3{cells.x, 0.0, 0.0});
                            e[1] = M(Vec3{0.0, cells.y, 0.0});
                            e[2] = M(Vec3{0.0, 0.0, cells.z});
                        }
                        Vec3 Jp;
                        polyJVol(dc, e, p, Jp);
                        Jtot = Jtot + Jp;
                    }
            Vec3 const Eraw = applyContract(Jtot, n, gam); // M^{-1}
            double const invV = 1.0 / (cells.x * cells.y * cells.z);
            return Vec3{Eraw.x * invV, Eraw.y * invV, Eraw.z * invV};
        }

        //! Exact TSC in-support field, evaluated in the source frame.
        inline Vec3 polyBoxField(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            return polyBoxFieldS(R0, cells, u);
        }

        /*! Geometric validity gate for the exact polynomial path.
         *
         * Rejects: non-finite input, beta = 0 (static path), an observer on or
         * too close to the outer support boundary or to an internal TSC knot
         * (origin on a piece face/edge), and near-degenerate piece geometry
         * (a face of the piece containing the origin passes too close to it).
         * The origin is always inside the union of the 27 pieces for in-support
         * pairs; only the piece containing s* has it strictly interior, and its
         * minimum face distance is the conditioning quantity.
         */
        inline bool polyBoxUsable(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            if(!(uu2 > 1e-24) || !std::isfinite(uu2))
                return false;
            if(!(cells.x > 0.0 && cells.y > 0.0 && cells.z > 0.0))
                return false;
            if(!(std::isfinite(cells.x) && std::isfinite(cells.y) && std::isfinite(cells.z)))
                return false;
            if(!(std::isfinite(R0.x) && std::isfinite(R0.y) && std::isfinite(R0.z)))
                return false;
            double const s[3] = {R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
            for(int a = 0; a < 3; ++a)
                if(!(s[a] > -1.5 && s[a] < 1.5))
                    return false;
            // Piece containing s*, the origin's interior margin and the piece
            // centre in source-index units.
            double const knots[4] = {-1.5, -0.5, 0.5, 1.5};
            double sc[3];
            for(int a = 0; a < 3; ++a)
            {
                int k = s[a] < -0.5 ? 0 : (s[a] < 0.5 ? 1 : 2);
                double const lo = knots[k];
                double const hi = knots[k + 1];
                double const m = std::min(s[a] - lo, hi - s[a]);
                if(!(m > 1.0e-6))
                    return false;
                sc[a] = static_cast<double>(k - 1);
            }
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = u / std::sqrt(uu2);
            auto const M = [&](Vec3 const& v) { return applyContract(v, n, 1.0 / gam); };
            Vec3 const dc = M(Vec3{R0.x - sc[0] * cells.x, R0.y - sc[1] * cells.y,
                                   R0.z - sc[2] * cells.z});
            Vec3 const e[3] = {
                M(Vec3{cells.x, 0.0, 0.0}),
                M(Vec3{0.0, cells.y, 0.0}),
                M(Vec3{0.0, 0.0, cells.z})};
            if(!(std::isfinite(dc.x) && std::isfinite(dc.y) && std::isfinite(dc.z)))
                return false;
            double emax = 0.0;
            for(int a = 0; a < 3; ++a)
            {
                double const l = norm3(e[a]);
                if(!(l > 0.0) || !std::isfinite(l))
                    return false;
                emax = std::max(emax, l);
            }
            Vec3 q[6][4];
            Vec3 nrm[6];
            double h[6];
            polyBoxFaces(dc, e, q, nrm, h);
            double hmin = 1e300;
            for(int f = 0; f < 6; ++f)
            {
                if(!std::isfinite(h[f]))
                    return false;
                hmin = std::min(hmin, std::abs(h[f]));
            }
            if(!(hmin > 1.0e-8 * emax))
                return false;
            return true;
        }


        /*! Production validity gate for the exact s-space evaluator.
         *
         * Extends polyBoxUsable to the two cases the singular d-space gate
         * rejects: the static observer (uu2 == 0) and an observer outside the
         * TSC support.  The s-space recurrence is a pure boundary/volume
         * identity that only needs the origin off every cell-face plane; the
         * knot margin below guarantees that.  In-support moving pairs still
         * go through the validated polyBoxUsable conditioning check.
         */
        inline bool sSpaceExactUsable(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            if(!(uu2 >= 0.0) || !std::isfinite(uu2))
                return false;
            if(!(cells.x > 0.0 && cells.y > 0.0 && cells.z > 0.0))
                return false;
            if(!(std::isfinite(cells.x) && std::isfinite(cells.y) && std::isfinite(cells.z)))
                return false;
            if(!(std::isfinite(R0.x) && std::isfinite(R0.y) && std::isfinite(R0.z)))
                return false;
            Vec3 const Rg = snapKnotOrigin(R0, cells, kKnotGateSnap);
            double const s[3] = {Rg.x / cells.x, Rg.y / cells.y, Rg.z / cells.z};
            bool inside = true;
            for(int a = 0; a < 3; ++a)
            {
                if(!std::isfinite(s[a]))
                    return false;
                inside = inside && std::abs(s[a]) < 1.5;
            }
            if(inside && uu2 > 0.0)
                return polyBoxUsable(Rg, cells, u);
            return true;
        }

        //! Source-relative TSC cell of an observer, clamped to the support.
        //! Returns the number of non-zero cell-offset axes: 0 is the cell
        //! containing the source, 1 a face-adjacent cell, 2/3 edge or corner.
        inline int supportCellClass(Vec3 const& R0, Vec3 const& cells)
        {
            int nonzero = 0;
            for(int a = 0; a < 3; ++a)
            {
                double const s = R0[a] / cells[a];
                // The shared knot plane belongs to the source-containing cell,
                // so the closest node on either side of the source is central.
                int m = (std::abs(s) <= 0.5 + kKnotDetect)
                    ? 0
                    : (s > 0.0 ? 1 : -1);
                if(m != 0)
                    ++nonzero;
            }
            return nonzero;
        }

        /*! Whether one (source, node) pair takes the exact s-space evaluator.
         *
         * In-support pairs follow ``Settings::exactCellReach``; out-of-support
         * pairs stay on the smooth remainder unless their d/gamma tube cannot
         * be resolved at the quadrature cap or gamma reaches the exact-only
         * threshold.  This predicate is shared by the host evaluator, the
         * device evaluator and the diagnostic pair counter so all three agree.
         */
        inline bool routesExact(Vec3 const& R0, Vec3 const& cells, Vec3 const& u, Settings const& settings)
        {
            // Reject the cheap invalid cases first.  In particular the knot
            // planes are common for a source on a grid node; they must not pay
            // for the conditioning gate (polyBoxUsable), which is the costly
            // part of the exact-path validity check.
            bool inside = true;
            for(int a = 0; a < 3; ++a)
            {
                double const s = R0[a] / cells[a];
                if(!std::isfinite(s))
                    return false;
                inside = inside && std::abs(s) < 1.5;
            }
            if(inside)
            {
                if(settings.exactCellReach < 0)
                    return false;
                int const cls = supportCellClass(R0, cells);
                bool const selected = settings.exactCellReach >= 2
                    || (settings.exactCellReach == 1 && cls <= 1)
                    || (settings.exactCellReach == 0 && cls == 0);
                if(!selected)
                    return false;
                return sSpaceExactUsable(R0, cells, u);
            }
            double const gamma = std::sqrt(1.0 + nbref::dot(u, u));
            int const need = pbfast::quadratureOrderFor(gamma, settings.tolerance, settings.order);
            if(!(pbfast::isExactGamma(gamma) || need > pbfast::kQuadratureOrderCap))
                return false;
            return sSpaceExactUsable(R0, cells, u);
        }


        /*! Exact box term for oblique motion.
         *
         * Ebox = int_B R / |M R|^3 dR over the support box B centred at R0.
         * With d = M R and P = M B this is M^{-1} int_P d/|d|^3 dd, the
         * Newtonian field at the origin of the Lorentz-contracted
         * parallelepiped P.  The polyhedron field is a pure boundary term
         * (Werner & Scheeres 1996),
         *
         *     int_P d/|d|^3 dd = -sum_faces n_f int_face dA/|d|,
         *
         * and every face integral is the closed form above.  This is exact
         * (up to FP rounding), O(1) per pair and has no sliver-tetrahedron or
         * M^{-1}-cancellation failure mode.
         */
        inline Vec3 exactObliqueBox(Vec3 const& R0, Vec3 const& cells, Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = u / std::sqrt(uu2);
            double const invg = 1.0 / gam;
            auto const M = [&](Vec3 const& v) { return applyContract(v, n, invg); };
            Vec3 const c0 = M(R0);
            Vec3 const e[3] = {
                M(Vec3{1.5 * cells.x, 0.0, 0.0}),
                M(Vec3{0.0, 1.5 * cells.y, 0.0}),
                M(Vec3{0.0, 0.0, 1.5 * cells.z})};
            Vec3 V[2][2][2];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        V[a][b][c] = c0 + (a ? e[0] : -1.0 * e[0]) + (b ? e[1] : -1.0 * e[1])
                            + (c ? e[2] : -1.0 * e[2]);
            Vec3 S{};
            auto addFace = [&](Vec3 const qin[4]) {
                Vec3 q[4] = {qin[0], qin[1], qin[2], qin[3]};
                Vec3 nn = nbref::cross(q[1] - q[0], q[2] - q[0]);
                nn = nn / norm3(nn);
                Vec3 const cen = (q[0] + q[1] + q[2] + q[3]) / 4.0;
                if(nbref::dot(cen - c0, nn) < 0.0)
                {
                    q[0] = qin[3];
                    q[1] = qin[2];
                    q[2] = qin[1];
                    q[3] = qin[0];
                    nn = nbref::cross(q[1] - q[0], q[2] - q[0]);
                    nn = nn / norm3(nn);
                }
                S = S + boxFaceIntegral(q, nn) * nn;
            };
            for(int a = 0; a < 2; ++a)
            {
                Vec3 const q[4] = {V[a][0][0], V[a][0][1], V[a][1][1], V[a][1][0]};
                addFace(q);
            }
            for(int b = 0; b < 2; ++b)
            {
                Vec3 const q[4] = {V[0][b][0], V[0][b][1], V[1][b][1], V[1][b][0]};
                addFace(q);
            }
            for(int c = 0; c < 2; ++c)
            {
                Vec3 const q[4] = {V[0][0][c], V[0][1][c], V[1][1][c], V[1][0][c]};
                addFace(q);
            }
            Vec3 const F = -1.0 * S;
            return applyContract(F, n, gam); // M^{-1}
        }

        struct BoxPlane
        {
            Vec3 n;
            double h;
        };

        //! Outward face planes of the Lorentz-contracted support parallelepiped.
        //! The origin is strictly inside when the observer is in support.
        inline void supportPlanes(
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u,
            BoxPlane planes[6])
        {
            double const uu2 = nbref::dot(u, u);
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = u / std::sqrt(uu2);
            auto const M = [&](Vec3 const& v) { return applyContract(v, n, 1.0 / gam); };
            Vec3 const c0 = M(R0);
            Vec3 const e[3] = {
                M(Vec3{1.5 * cells.x, 0.0, 0.0}),
                M(Vec3{0.0, 1.5 * cells.y, 0.0}),
                M(Vec3{0.0, 0.0, 1.5 * cells.z})};
            Vec3 V[2][2][2];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        V[a][b][c] = c0 + (a ? e[0] : -1.0 * e[0]) + (b ? e[1] : -1.0 * e[1])
                            + (c ? e[2] : -1.0 * e[2]);
            int const FI[6][4][3] = {
                {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
            for(int f = 0; f < 6; ++f)
            {
                Vec3 q[4];
                for(int v = 0; v < 4; ++v)
                    q[v] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]];
                Vec3 nn = nbref::cross(q[1] - q[0], q[2] - q[0]);
                nn = nn / norm3(nn);
                Vec3 const cen = (q[0] + q[1] + q[2] + q[3]) / 4.0;
                if(nbref::dot(cen - c0, nn) < 0.0)
                {
                    Vec3 const t0 = q[0], t1 = q[1], t2 = q[2], t3 = q[3];
                    q[0] = t3;
                    q[1] = t2;
                    q[2] = t1;
                    q[3] = t0;
                    nn = nbref::cross(q[1] - q[0], q[2] - q[0]);
                    nn = nn / norm3(nn);
                }
                planes[f].n = nn;
                planes[f].h = nbref::dot(q[0], nn);
            }
        }

        /*! Full in-support field of an oblique pair in d-space.
         *
         * With d = M R the integrand becomes a plain radial factor,
         * d/|d|^3 dV = Omega dr dOmega, so
         *
         *   int S(s) K ds^3
         *     = (1/V) M^{-1} int_{S^2} Omega [ int_0^rmax S(s(r Omega)) dr ] dOmega.
         *
         * The inner integral is a piecewise polynomial of degree 6 along the
         * ray; four Gauss nodes on every knot interval integrate it exactly.
         * The polar axis of the angular rule is aligned with beta, which puts
         * the beamed tube near mu = 1 where Gauss-Legendre clusters strongly.
         * This direct form avoids the large W*box/(S-W*) cancellation of the
         * constant-density split, whose finite-angle odd residual would show
         * up in the small transverse field components.
         */
        inline Vec3 obliqueRayField(
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u)
        {
            double const uu2 = nbref::dot(u, u);
            double const gam = std::sqrt(1.0 + uu2);
            Vec3 const n = u / std::sqrt(uu2);
            Vec3 const sstar{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
            BoxPlane planes[6];
            supportPlanes(R0, cells, u, planes);

            Vec3 const helper = std::abs(n.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
            Vec3 e1 = nbref::cross(n, helper);
            e1 = e1 / norm3(e1);
            Vec3 const e2 = nbref::cross(n, e1);

            Rule const& muR = rule(16);
            Rule const& phR = rule(32);
            Rule const& radR = rule(4);
            // pi*(x+1) for the phi rule is evaluated per node below.
            constexpr double pi = 3.141592653589793238462643383279502884;
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};

            Vec3 acc{};
            auto const Minv = [&](Vec3 const& v) { return applyContract(v, n, gam); };
            for(int im = 0; im < 16; ++im)
            {
                double const mu = muR.x[static_cast<std::size_t>(im)];
                double const st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                double const wmu = muR.w[static_cast<std::size_t>(im)];
                for(int ip = 0; ip < 32; ++ip)
                {
                    double const phi = pi * (phR.x[static_cast<std::size_t>(ip)] + 1.0);
                    double const wphi = pi * phR.w[static_cast<std::size_t>(ip)];
                    Vec3 const Om = mu * n + st * (std::cos(phi) * e1 + std::sin(phi) * e2);
                    double rmax = 1e300;
                    for(int f = 0; f < 6; ++f)
                    {
                        double const dn = nbref::dot(planes[f].n, Om);
                        if(dn > 1e-15)
                            rmax = std::min(rmax, planes[f].h / dn);
                    }
                    if(!(rmax > 1e-14))
                        continue;
                    Vec3 const BO = Minv(Om);
                    double const BOa[3] = {
                        BO.x / cells.x,
                        BO.y / cells.y,
                        BO.z / cells.z};
                    double const sst[3] = {sstar.x, sstar.y, sstar.z};
                    double bp[16];
                    int nb = 0;
                    bp[nb++] = 0.0;
                    bp[nb++] = rmax;
                    for(int a = 0; a < 3; ++a)
                    {
                        if(std::abs(BOa[a]) < 1e-300)
                            continue;
                        for(int k = 0; k < 4; ++k)
                        {
                            double const r = (sst[a] - knots[k]) / BOa[a];
                            if(r > 1e-14 && r < rmax - 1e-14)
                            {
                                bool dup = false;
                                for(int q = 0; q < nb; ++q)
                                    if(std::abs(bp[q] - r) < 1e-12 * rmax)
                                        dup = true;
                                if(!dup)
                                    bp[nb++] = r;
                            }
                        }
                    }
                    for(int i = 1; i < nb; ++i)
                    {
                        double const key = bp[i];
                        int j = i - 1;
                        while(j >= 0 && bp[j] > key)
                        {
                            bp[j + 1] = bp[j];
                            --j;
                        }
                        bp[j + 1] = key;
                    }
                    double const wa = wmu * wphi;
                    for(int i = 0; i + 1 < nb; ++i)
                    {
                        double const r0 = bp[i];
                        double const r1 = bp[i + 1];
                        if(r1 - r0 <= 1e-15)
                            continue;
                        double const mid = 0.5 * (r0 + r1);
                        double const half = 0.5 * (r1 - r0);
                        for(int ir = 0; ir < 4; ++ir)
                        {
                            double const r
                                = mid + half * radR.x[static_cast<std::size_t>(ir)];
                            double const wr
                                = half * radR.w[static_cast<std::size_t>(ir)];
                            Vec3 const s{
                                sstar.x - r * BOa[0],
                                sstar.y - r * BOa[1],
                                sstar.z - r * BOa[2]};
                            double const q = tsc(s.x) * tsc(s.y) * tsc(s.z);
                            acc = acc + (wa * wr * q) * Om;
                        }
                    }
                }
            }
            return Minv(acc) / (cells.x * cells.y * cells.z);
        }

        struct Panels
        {
            std::array<double, 4> lo{};
            std::array<double, 4> hi{};
            int count = 0;
        };

        //! TSC knot panels; split at s* when s* is strictly interior.
        inline Panels axisPanels(double sstar)
        {
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};
            Panels p;
            if(sstar > -1.5 && sstar < 1.5)
            {
                for(int k = 0; k < 3; ++k)
                {
                    if(sstar > knots[k] && sstar < knots[k + 1])
                    {
                        p.lo[static_cast<std::size_t>(p.count)] = knots[k];
                        p.hi[static_cast<std::size_t>(p.count)] = sstar;
                        ++p.count;
                        p.lo[static_cast<std::size_t>(p.count)] = sstar;
                        p.hi[static_cast<std::size_t>(p.count)] = knots[k + 1];
                        ++p.count;
                    }
                    else
                    {
                        p.lo[static_cast<std::size_t>(p.count)] = knots[k];
                        p.hi[static_cast<std::size_t>(p.count)] = knots[k + 1];
                        ++p.count;
                    }
                }
            }
            else
            {
                for(int k = 0; k < 3; ++k)
                {
                    p.lo[static_cast<std::size_t>(p.count)] = knots[k];
                    p.hi[static_cast<std::size_t>(p.count)] = knots[k + 1];
                    ++p.count;
                }
            }
            return p;
        }

        //! Singular box field for a uniformly moving charge.
        inline void boxTerm(
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u,
            int /*obliqueOrder*/,
            int /*obliquePower*/,
            Vec3& E,
            Vec3& B)
        {
            double const uu2 = nbref::dot(u, u);
            Vec3 Ebox{};
            Vec3 Bbox{};
            if(uu2 == 0.0)
            {
                double const p[3] = {R0.x, R0.y, R0.z};
                double const lo[3] = {-1.5 * cells.x, -1.5 * cells.y, -1.5 * cells.z};
                double const hi[3] = {1.5 * cells.x, 1.5 * cells.y, 1.5 * cells.z};
                for(int c = 0; c < 3; ++c)
                    Ebox[c] = boxField8T(p, lo, hi, c);
            }
            else
            {
                double const gam = std::sqrt(1.0 + uu2);
                Vec3 const beta = u / gam;
                Vec3 const betaH = u / std::sqrt(uu2);
                int stretchAxis = -1;
                for(int a = 0; a < 3; ++a)
                    if(std::abs(std::abs(betaH[a]) - 1.0) < 1e-12)
                        stretchAxis = a;
                if(stretchAxis >= 0)
                {
                    double const p[3] = {R0[0], R0[1], R0[2]};
                    double lo[3], hi[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        lo[a] = p[a] - 1.5 * cells[a];
                        hi[a] = p[a] + 1.5 * cells[a];
                    }
                    lo[stretchAxis] *= gam;
                    hi[stretchAxis] *= gam;
                    double const tlo[3] = {-hi[0], -hi[1], -hi[2]};
                    double const thi[3] = {-lo[0], -lo[1], -lo[2]};
                    double const zero[3] = {0.0, 0.0, 0.0};
                    for(int c = 0; c < 3; ++c)
                        Ebox[c] = boxField8T(zero, tlo, thi, c);
                    Ebox[stretchAxis] /= gam;
                    Bbox = nbref::cross(beta, Ebox);
                }
                else
                {
                    // Exact Lorentz-contracted support-box field.  The oblique
                    // order/power arguments are irrelevant on this path and
                    // are kept only for CLI/settings compatibility.
                    Ebox = exactObliqueBox(R0, cells, u);
                    Bbox = nbref::cross(beta, Ebox);
                }
            }
            double const invVolume = 1.0 / (cells.x * cells.y * cells.z);
            E = invVolume * Ebox;
            B = invVolume * Bbox;
        }

    } // namespace detail

    //! TSC-convolved per-unit-charge field of one uniformly moving source.
    inline void shape_field(
        Vec3 const& x_obs,
        Vec3 const& x_src,
        Vec3 const& u,
        double dx,
        double dy,
        double dz,
        Settings const& settings,
        Vec3& E,
        Vec3& B)
    {
        Vec3 const cells{dx, dy, dz};
        Vec3 const R0 = x_obs - x_src;
        Vec3 const sstar{R0.x / dx, R0.y / dy, R0.z / dz};
        double const Wstar = detail::tsc(sstar.x) * detail::tsc(sstar.y) * detail::tsc(sstar.z);
        bool const singular = Wstar > 0.0;

        double const uu2 = nbref::dot(u, u);
        Vec3 const beta = uu2 == 0.0 ? Vec3{} : u / std::sqrt(1.0 + uu2);
        double const invg2 = 1.0 / (1.0 + uu2);
        double const gamma = std::sqrt(1.0 + uu2);

        // Exact s-space evaluator for the selected support cell classes.
        // Out-of-support high-gamma pairs still route exact through the shared
        // predicate.  Everything else keeps the tolerance-driven quadrature.
        bool const useExact = detail::routesExact(R0, cells, u, settings);
        if(useExact)
        {
            E = detail::polyBoxFieldS(R0, cells, u);
            B = nbref::cross(beta, E);
            return;
        }

        // Legacy fallback for geometry the exact evaluator rejects.  Oblique
        // in-support pairs use the d-space ray integral; axial pairs use the
        // gamma-stretched box, both plus the subtracted remainder below.
        if(singular)
        {
            Vec3 const betaH = uu2 == 0.0 ? Vec3{} : u / std::sqrt(uu2);
            if(uu2 > 0.0 && !detail::isAxialBeta(u, betaH))
            {
                E = detail::obliqueRayField(R0, cells, u);
                B = nbref::cross(beta, E);
                return;
            }
        }

        Vec3 Ebox{}, Bbox{};
        if(singular)
            detail::boxTerm(R0, cells, u, settings.obliqueOrder, settings.obliquePower, Ebox, Bbox);

        // Remaining quadrature uses a tolerance- and gamma-derived order with
        // a hard convergence cap.  Pairs above the cap are routed to the
        // exact evaluator above, so this rule always exists.
        int const quadOrder = std::min(
            pbfast::kQuadratureOrderCap,
            std::max(2, pbfast::quadratureOrderFor(gamma, settings.tolerance, settings.order)));
        detail::Rule const& gl = detail::rule(quadOrder);
        int const n = quadOrder;
        detail::Panels const px = detail::axisPanels(sstar.x);
        detail::Panels const py = detail::axisPanels(sstar.y);
        detail::Panels const pz = detail::axisPanels(sstar.z);

        Vec3 Erem{};
        for(int ax = 0; ax < px.count; ++ax)
            for(int ay = 0; ay < py.count; ++ay)
                for(int az = 0; az < pz.count; ++az)
                {
                    double const mx = 0.5 * (px.hi[ax] - px.lo[ax]);
                    double const mxy = 0.5 * (py.hi[ay] - py.lo[ay]);
                    double const mxz = 0.5 * (pz.hi[az] - pz.lo[az]);
                    double const cx = 0.5 * (px.lo[ax] + px.hi[ax]);
                    double const cy = 0.5 * (py.lo[ay] + py.hi[ay]);
                    double const cz = 0.5 * (pz.lo[az] + pz.hi[az]);
                    for(int ix = 0; ix < n; ++ix)
                    {
                        double const sx = cx + mx * gl.x[ix];
                        double const wx = mx * gl.w[ix];
                        double const tscx = detail::tsc(sx);
                        for(int iy = 0; iy < n; ++iy)
                        {
                            double const sy = cy + mxy * gl.x[iy];
                            double const wy = wx * mxy * gl.w[iy];
                            double const tscxy = tscx * detail::tsc(sy);
                            for(int iz = 0; iz < n; ++iz)
                            {
                                double const sz = cz + mxz * gl.x[iz];
                                double const S = tscxy * detail::tsc(sz);
                                double const w = wy * mxz * gl.w[iz] * (S - Wstar);
                                double const Rx = R0.x - sx * dx;
                                double const Ry = R0.y - sy * dy;
                                double const Rz = R0.z - sz * dz;
                                if(uu2 == 0.0)
                                {
                                    double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                                    r2 = std::max(r2, 1e-300);
                                    double const f = 1.0 / (r2 * std::sqrt(r2));
                                    Erem = Erem + (w * f) * Vec3{Rx, Ry, Rz};
                                }
                                else
                                {
                                    double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                                    double const rb = Rx * beta.x + Ry * beta.y + Rz * beta.z;
                                    double D = r2 * invg2 + rb * rb;
                                    D = std::max(D, 1e-300);
                                    double const f = invg2 / (D * std::sqrt(D));
                                    Erem = Erem + (w * f) * Vec3{Rx, Ry, Rz};
                                }
                            }
                        }
                    }
                }

        E = Wstar * Ebox + Erem;
        B = Wstar * Bbox + nbref::cross(beta, Erem);


    }

} // namespace picBunch::pbfast
