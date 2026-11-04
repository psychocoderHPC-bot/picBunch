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
// Moving sources: axial beta has the exact gamma-stretched box; oblique beta
// has no closed form and uses the rest-frame octant quadrature (8*order^3
// nodes).  The oblique remainder is not a controlled approximation in the
// beamed peak, hence the documented 7.7e-2 interior limit.

#pragma once

#include "picBunch/detail/NearReference.hpp"

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
            if(n == 4)
                return r4;
            if(n == 5)
                return r5;
            if(n == 6)
                return r6;
            if(n == 8)
                return r8;
            static std::mutex mutex;
            static std::map<int, Rule> cache;
            std::lock_guard<std::mutex> lock(mutex);
            return cache.emplace(n, makeRule(n)).first->second;
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
            int obliqueOrder,
            int obliquePower,
            Vec3& E,
            Vec3& B)
        {
            double const uu2 = nbref::dot(u, u);
            double Ebox[3] = {0.0, 0.0, 0.0};
            if(uu2 == 0.0)
            {
                double const p[3] = {R0.x, R0.y, R0.z};
                double const lo[3] = {-1.5 * cells.x, -1.5 * cells.y, -1.5 * cells.z};
                double const hi[3] = {1.5 * cells.x, 1.5 * cells.y, 1.5 * cells.z};
                for(int c = 0; c < 3; ++c)
                    Ebox[c] = boxField8T(p, lo, hi, c);
                B = Vec3{};
            }
            else
            {
                double const gam = std::sqrt(1.0 + uu2);
                Vec3 const beta = u / gam;
                // Axial beta: exact gamma-stretched box, no oblique quadrature.
                Vec3 const betaH = u / std::sqrt(uu2);
                int stretchAxis = -1;
                for(int a = 0; a < 3; ++a)
                    if(std::abs(std::abs(betaH[a]) - 1.0) < 1e-12)
                        stretchAxis = a;
                if(stretchAxis >= 0)
                {
                    // t-box in the stretched coordinate; E' = boxField(0, -hi, -lo).
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
                    B = nbref::cross(beta, Vec3{Ebox[0], Ebox[1], Ebox[2]});
                }
                else
                {
                    // Oblique: rest-frame octant quadrature over the lab box.
                    double const mm = gam - 1.0;
                    // Symmetric M = I + (gamma-1) n n^T, applied to d.
                    auto applyM = [&](Vec3 const& d) -> Vec3 {
                        double const dn = nbref::dot(d, betaH);
                        return d + mm * dn * betaH;
                    };
                    double const Lp[3] = {
                        1.5 * cells.x - R0.x,
                        1.5 * cells.y - R0.y,
                        1.5 * cells.z - R0.z};
                    double const Ln[3] = {
                        1.5 * cells.x + R0.x,
                        1.5 * cells.y + R0.y,
                        1.5 * cells.z + R0.z};
                    Rule const& gl = rule(obliqueOrder);
                    int const n = obliqueOrder;
                    double const power = obliquePower;
                    Vec3 acc{};
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
                                for(int ix = 0; ix < n; ++ix)
                                    for(int iy = 0; iy < n; ++iy)
                                        for(int iz = 0; iz < n; ++iz)
                                        {
                                            double const tx = 0.5 * gl.x[ix] + 0.5;
                                            double const ty = 0.5 * gl.x[iy] + 0.5;
                                            double const tz = 0.5 * gl.x[iz] + 0.5;
                                            double const px = L[0] * std::pow(tx, power);
                                            double const py = L[1] * std::pow(ty, power);
                                            double const pz = L[2] * std::pow(tz, power);
                                            double const jx = 0.5 * L[0] * power
                                                * std::pow(tx, power - 1.0) * gl.w[ix];
                                            double const jy = 0.5 * L[1] * power
                                                * std::pow(ty, power - 1.0) * gl.w[iy];
                                            double const jz = 0.5 * L[2] * power
                                                * std::pow(tz, power - 1.0) * gl.w[iz];
                                            Vec3 const d{sx * px, sy * py, sz * pz};
                                            Vec3 const md = applyM(d);
                                            double const rho2 = nbref::dot(md, md);
                                            double const rho = std::sqrt(rho2);
                                            double const f = -gam / (rho2 * rho) * jx * jy * jz;
                                            acc = acc + f * d;
                                        }
                            }
                    Ebox[0] = acc.x;
                    Ebox[1] = acc.y;
                    Ebox[2] = acc.z;
                    B = nbref::cross(beta, acc);
                }
            }
            // The box integrals above are over the physical displacement box
            // (units 1/m^2 * m^3).  The TSC shape integral is over the
            // dimensionless cell coordinate s, ds^3 = d^3R / (dx dy dz), so the
            // singular term carries one inverse cell volume.
            double const invVolume = 1.0 / (cells.x * cells.y * cells.z);
            E = invVolume * Vec3{Ebox[0], Ebox[1], Ebox[2]};
            B = invVolume * B;
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
