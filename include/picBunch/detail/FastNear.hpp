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
        //! Force the legacy plain octant box (diagnostics / equivalence tests).
        bool legacyOblique = false;
    };

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

        // Oblique in-support: exact Lorentz-contracted support-box field plus
        // a stable d-space ray integral of the TSC remainder.  This replaces
        // the beamed-peak Duffy quadrature; the oblique order is irrelevant
        // here and is kept only for CLI/settings compatibility.
        if(singular)
        {
            double const uu2 = nbref::dot(u, u);
            Vec3 const betaH = uu2 == 0.0 ? Vec3{} : u / std::sqrt(uu2);
            if(uu2 > 0.0 && !detail::isAxialBeta(u, betaH))
            {
                E = detail::obliqueRayField(R0, cells, u);
                Vec3 const beta = u / std::sqrt(1.0 + uu2);
                B = nbref::cross(beta, E);
                return;
            }
        }

        Vec3 Ebox{}, Bbox{};
        if(singular)
            detail::boxTerm(R0, cells, u, settings.obliqueOrder, settings.obliquePower, Ebox, Bbox);

        double const uu2 = nbref::dot(u, u);
        Vec3 const beta = uu2 == 0.0 ? Vec3{} : u / std::sqrt(1.0 + uu2);
        double const invg2 = 1.0 / (1.0 + uu2);

        detail::Panels const px = detail::axisPanels(sstar.x);
        detail::Panels const py = detail::axisPanels(sstar.y);
        detail::Panels const pz = detail::axisPanels(sstar.z);
        detail::Rule const& gl = detail::rule(settings.order);
        int const n = settings.order;

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
