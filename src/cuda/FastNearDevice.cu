// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// CUDA instantiation of the fast near-field device backend.  The kernel is a
// direct port of include/picBunch/detail/FastNear.hpp and of the validated
// tools/closed-form/gpu_extract.cu extract kernel (picongpu branch
// experimental/closed-form, b3699dff).  All precision-critical work is double.

#include "picBunch/detail/FastNearDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace picBunch::pbfast
{
    namespace
    {
        // ---- device algebra, mirroring FastNear.hpp -------------------------

        ALPAKA_FN_ACC inline double dtsc(double x)
        {
            double const ax = x < 0.0 ? -x : x;
            if(ax < 0.5)
                return 0.75 - ax * ax;
            if(ax < 1.5)
            {
                double const t = 1.5 - ax;
                return 0.5 * t * t;
            }
            return 0.0;
        }

        ALPAKA_FN_ACC inline double dTfun(double A, double B, double u)
        {
            if(std::abs(u) < 1e-300)
                return A * std::asinh(B / A) + B * std::asinh(A / B);
            return A * std::asinh(B / std::sqrt(u * u + A * A))
                + B * std::asinh(A / std::sqrt(u * u + B * B))
                - u * std::atan(A * B / (u * std::sqrt(A * A + u * u + B * B)));
        }

        ALPAKA_FN_ACC inline double dboxField8T(double const p[3], double const lo[3], double const hi[3], int c)
        {
            double E = 0.0;
            if(c == 0)
            {
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ey[a], ez[b], hi[0] - p[0]) - dTfun(ey[a], ez[b], p[0] - lo[0]);
            }
            else if(c == 1)
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ex[a], ez[b], hi[1] - p[1]) - dTfun(ex[a], ez[b], p[1] - lo[1]);
            }
            else
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ex[a], ey[b], hi[2] - p[2]) - dTfun(ex[a], ey[b], p[2] - lo[2]);
            }
            return E;
        }

        ALPAKA_FN_ACC inline void dboxTerm(
            double const R0[3],
            double const cells[3],
            double const u[3],
            DeviceParams const& prm,
            double E[3],
            double B[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double Ebox[3] = {0.0, 0.0, 0.0};
            if(uu2 == 0.0)
            {
                double const p[3] = {R0[0], R0[1], R0[2]};
                double const lo[3] = {-1.5 * cells[0], -1.5 * cells[1], -1.5 * cells[2]};
                double const hi[3] = {1.5 * cells[0], 1.5 * cells[1], 1.5 * cells[2]};
                for(int c = 0; c < 3; ++c)
                    Ebox[c] = dboxField8T(p, lo, hi, c);
                B[0] = B[1] = B[2] = 0.0;
            }
            else
            {
                double const gam = std::sqrt(1.0 + uu2);
                double const beta[3] = {u[0] / gam, u[1] / gam, u[2] / gam};
                double const betaH[3] = {
                    u[0] / std::sqrt(uu2),
                    u[1] / std::sqrt(uu2),
                    u[2] / std::sqrt(uu2)};
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
                        Ebox[c] = dboxField8T(zero, tlo, thi, c);
                    Ebox[stretchAxis] /= gam;
                    B[0] = beta[1] * Ebox[2] - beta[2] * Ebox[1];
                    B[1] = beta[2] * Ebox[0] - beta[0] * Ebox[2];
                    B[2] = beta[0] * Ebox[1] - beta[1] * Ebox[0];
                }
                else
                {
                    // Oblique: Lorentz-contracted tetrahedral Duffy quadrature.
                    // In d-space the kernel is |M R|^{-3} R; with d' = M R the
                    // (1-beta^2) prefactor cancels the Jacobian and the
                    // integrand becomes the plain Coulomb field M^{-1} d'/|d'|^3.
                    // Each corner box is split into the six tetrahedra of a
                    // cube and mapped with the collapsed tetrahedral map, which
                    // turns the vertex singularity into a smooth integrand.
                    double const Lp[3] = {
                        1.5 * cells[0] - R0[0],
                        1.5 * cells[1] - R0[1],
                        1.5 * cells[2] - R0[2]};
                    double const Ln[3] = {
                        1.5 * cells[0] + R0[0],
                        1.5 * cells[1] + R0[1],
                        1.5 * cells[2] + R0[2]};
                    int const n = prm.obliqueOrder;
                    double acc[3] = {0.0, 0.0, 0.0};
                    auto applyMmin = [&](double const v[3], double out[3]) {
                        double const vn = v[0] * betaH[0] + v[1] * betaH[1] + v[2] * betaH[2];
                        double const f = 1.0 / gam - 1.0;
                        out[0] = v[0] + f * (v[0] - vn * betaH[0]);
                        out[1] = v[1] + f * (v[1] - vn * betaH[1]);
                        out[2] = v[2] + f * (v[2] - vn * betaH[2]);
                    };
                    auto applyMplus = [&](double const v[3], double out[3]) {
                        double const vn = v[0] * betaH[0] + v[1] * betaH[1] + v[2] * betaH[2];
                        double const f = gam - 1.0;
                        out[0] = v[0] + f * (v[0] - vn * betaH[0]);
                        out[1] = v[1] + f * (v[1] - vn * betaH[1]);
                        out[2] = v[2] + f * (v[2] - vn * betaH[2]);
                    };
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
                                double ev[3][3] = {{sx * L[0], 0.0, 0.0}, {0.0, sy * L[1], 0.0},
                                                   {0.0, 0.0, sz * L[2]}};
                                double e[3][3];
                                for(int a = 0; a < 3; ++a)
                                    applyMmin(ev[a], e[a]);
                                for(int pi = 0; pi < 6; ++pi)
                                {
                                    int const i0 = perm[pi][0];
                                    int const i1 = perm[pi][1];
                                    int const i2 = perm[pi][2];
                                    double const A[3] = {e[i0][0], e[i0][1], e[i0][2]};
                                    double const Bv[3]
                                        = {A[0] + e[i1][0], A[1] + e[i1][1], A[2] + e[i1][2]};
                                    double const C[3]
                                        = {Bv[0] + e[i2][0], Bv[1] + e[i2][1], Bv[2] + e[i2][2]};
                                    double const BA[3] = {Bv[0] - A[0], Bv[1] - A[1], Bv[2] - A[2]};
                                    for(int iu = 0; iu < n; ++iu)
                                    {
                                        double const tu = 0.5 * prm.oxi[iu] + 0.5;
                                        double const wu = 0.5 * prm.owi[iu];
                                        for(int iv = 0; iv < n; ++iv)
                                        {
                                            double const tv = 0.5 * prm.oxi[iv] + 0.5;
                                            double const wv = 0.5 * prm.owi[iv];
                                            for(int iw = 0; iw < n; ++iw)
                                            {
                                                double const tw = 0.5 * prm.oxi[iw] + 0.5;
                                                double const ww = 0.5 * prm.owi[iw];
                                                double const a0 = (1.0 - tv) * (1.0 - tw);
                                                double const a1 = tv * (1.0 - tw);
                                                double const Q[3] = {
                                                    a0 * A[0] + a1 * Bv[0] + tw * C[0],
                                                    a0 * A[1] + a1 * Bv[1] + tw * C[1],
                                                    a0 * A[2] + a1 * Bv[2] + tw * C[2]};
                                                double const Qv[3]
                                                    = {(1.0 - tw) * BA[0], (1.0 - tw) * BA[1],
                                                       (1.0 - tw) * BA[2]};
                                                double const Qw[3] = {
                                                    -(1.0 - tv) * A[0] - tv * Bv[0] + C[0],
                                                    -(1.0 - tv) * A[1] - tv * Bv[1] + C[1],
                                                    -(1.0 - tv) * A[2] - tv * Bv[2] + C[2]};
                                                double const det = Q[0] * (Qv[1] * Qw[2] - Qv[2] * Qw[1])
                                                    - Q[1] * (Qv[0] * Qw[2] - Qv[2] * Qw[0])
                                                    + Q[2] * (Qv[0] * Qw[1] - Qv[1] * Qw[0]);
                                                double const dp[3] = {tu * Q[0], tu * Q[1], tu * Q[2]};
                                                double const rho2
                                                    = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
                                                if(rho2 < 1e-300)
                                                    continue;
                                                double const rho = std::sqrt(rho2);
                                                double Rv[3];
                                                applyMplus(dp, Rv);
                                                double const jac = tu * tu * std::abs(det) * wu * wv * ww;
                                                double const f = -jac / (rho2 * rho);
                                                acc[0] += f * Rv[0];
                                                acc[1] += f * Rv[1];
                                                acc[2] += f * Rv[2];
                                            }
                                        }
                                    }
                                }
                            }
                    Ebox[0] = acc[0];
                    Ebox[1] = acc[1];
                    Ebox[2] = acc[2];
                    B[0] = beta[1] * acc[2] - beta[2] * acc[1];
                    B[1] = beta[2] * acc[0] - beta[0] * acc[2];
                    B[2] = beta[0] * acc[1] - beta[1] * acc[0];
                }
            }
            double const invVolume = 1.0 / (cells[0] * cells[1] * cells[2]);
            E[0] = invVolume * Ebox[0];
            E[1] = invVolume * Ebox[1];
            E[2] = invVolume * Ebox[2];
            B[0] = invVolume * B[0];
            B[1] = invVolume * B[1];
            B[2] = invVolume * B[2];
        }

        ALPAKA_FN_ACC inline void dshapeField(
            double const x_obs[3],
            double const x_src[3],
            double const u[3],
            double const cells[3],
            DeviceParams const& prm,
            double E[3],
            double B[3])
        {
            double const R0[3] = {x_obs[0] - x_src[0], x_obs[1] - x_src[1], x_obs[2] - x_src[2]};
            double const sstar[3] = {R0[0] / cells[0], R0[1] / cells[1], R0[2] / cells[2]};
            double const Wstar = dtsc(sstar[0]) * dtsc(sstar[1]) * dtsc(sstar[2]);
            bool const singular = Wstar > 0.0;

            double Ebox[3] = {0.0, 0.0, 0.0};
            double Bbox[3] = {0.0, 0.0, 0.0};
            if(singular)
                dboxTerm(R0, cells, u, prm, Ebox, Bbox);

            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const beta[3] = {u[0] / std::sqrt(1.0 + uu2), u[1] / std::sqrt(1.0 + uu2),
                                    u[2] / std::sqrt(1.0 + uu2)};
            double const invg2 = 1.0 / (1.0 + uu2);

            constexpr double kKnots[3][2] = {{-1.5, -0.5}, {-0.5, 0.5}, {0.5, 1.5}};
            double plo[3][4], phi[3][4];
            int npan[3];
            for(int a = 0; a < 3; ++a)
            {
                if(sstar[a] > -1.5 && sstar[a] < 1.5)
                {
                    int m = 0;
                    for(int k = 0; k < 3; ++k)
                    {
                        if(sstar[a] > kKnots[k][0] && sstar[a] < kKnots[k][1])
                        {
                            plo[a][m] = kKnots[k][0];
                            phi[a][m] = sstar[a];
                            ++m;
                            plo[a][m] = sstar[a];
                            phi[a][m] = kKnots[k][1];
                            ++m;
                        }
                        else
                        {
                            plo[a][m] = kKnots[k][0];
                            phi[a][m] = kKnots[k][1];
                            ++m;
                        }
                    }
                    npan[a] = m;
                }
                else
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        plo[a][k] = kKnots[k][0];
                        phi[a][k] = kKnots[k][1];
                    }
                    npan[a] = 3;
                }
            }

            double Erem[3] = {0.0, 0.0, 0.0};
            int const n = prm.order;
            for(int ax = 0; ax < npan[0]; ++ax)
                for(int ay = 0; ay < npan[1]; ++ay)
                    for(int az = 0; az < npan[2]; ++az)
                    {
                        double const mx = 0.5 * (phi[0][ax] - plo[0][ax]);
                        double const my = 0.5 * (phi[1][ay] - plo[1][ay]);
                        double const mz = 0.5 * (phi[2][az] - plo[2][az]);
                        double const cx = 0.5 * (plo[0][ax] + phi[0][ax]);
                        double const cy = 0.5 * (plo[1][ay] + phi[1][ay]);
                        double const cz = 0.5 * (plo[2][az] + phi[2][az]);
                        for(int ix = 0; ix < n; ++ix)
                        {
                            double const sx = cx + mx * prm.xi[ix];
                            double const wx = mx * prm.wi[ix];
                            double const tscx = dtsc(sx);
                            for(int iy = 0; iy < n; ++iy)
                            {
                                double const sy = cy + my * prm.xi[iy];
                                double const wy = wx * my * prm.wi[iy];
                                double const tscxy = tscx * dtsc(sy);
                                for(int iz = 0; iz < n; ++iz)
                                {
                                    double const sz = cz + mz * prm.xi[iz];
                                    double const S = tscxy * dtsc(sz);
                                    double const w = wy * mz * prm.wi[iz] * (S - Wstar);
                                    double const Rx = R0[0] - sx * cells[0];
                                    double const Ry = R0[1] - sy * cells[1];
                                    double const Rz = R0[2] - sz * cells[2];
                                    double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                                    if(uu2 == 0.0)
                                    {
                                        r2 = r2 < 1e-300 ? 1e-300 : r2;
                                        double const f = 1.0 / (r2 * std::sqrt(r2));
                                        Erem[0] += w * f * Rx;
                                        Erem[1] += w * f * Ry;
                                        Erem[2] += w * f * Rz;
                                    }
                                    else
                                    {
                                        double const rb = Rx * beta[0] + Ry * beta[1] + Rz * beta[2];
                                        double D = r2 * invg2 + rb * rb;
                                        D = D < 1e-300 ? 1e-300 : D;
                                        double const f = invg2 / (D * std::sqrt(D));
                                        Erem[0] += w * f * Rx;
                                        Erem[1] += w * f * Ry;
                                        Erem[2] += w * f * Rz;
                                    }
                                }
                            }
                        }
                    }

            E[0] = Wstar * Ebox[0] + Erem[0];
            E[1] = Wstar * Ebox[1] + Erem[1];
            E[2] = Wstar * Ebox[2] + Erem[2];
            B[0] = Wstar * Bbox[0] + (beta[1] * Erem[2] - beta[2] * Erem[1]);
            B[1] = Wstar * Bbox[1] + (beta[2] * Erem[0] - beta[0] * Erem[2]);
            B[2] = Wstar * Bbox[2] + (beta[0] * Erem[1] - beta[1] * Erem[0]);
        }

        //! One component of the normalized point field (Kernel.hpp).
        ALPAKA_FN_ACC inline double dpointComp(
            double Rx,
            double Ry,
            double Rz,
            double const u[3],
            int c)
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const invG2 = 1.0 / (1.0 + uu2);
            double Ex, Ey, Ez;
            if(uu2 == 0.0)
            {
                double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                double const r3 = r2 * std::sqrt(r2);
                double const f = r3 > 0.0 ? 1.0 / r3 : 0.0;
                Ex = f * Rx;
                Ey = f * Ry;
                Ez = f * Rz;
            }
            else
            {
                double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                double const gamma = std::sqrt(1.0 + uu2);
                double const rb = (Rx * u[0] + Ry * u[1] + Rz * u[2]) / gamma;
                double const D = r2 * invG2 + rb * rb;
                double const f = D > 0.0 ? invG2 / (D * std::sqrt(D)) : 0.0;
                Ex = f * Rx;
                Ey = f * Ry;
                Ez = f * Rz;
            }
            if(c == 0)
                return Ex;
            if(c == 1)
                return Ey;
            if(c == 2)
                return Ez;
            double const gamma = std::sqrt(1.0 + uu2);
            double const bx = u[0] / gamma;
            double const by = u[1] / gamma;
            double const bz = u[2] / gamma;
            if(c == 3)
                return by * Ez - bz * Ey;
            if(c == 4)
                return bz * Ex - bx * Ez;
            return bx * Ey - by * Ex;
        }

        //! 27-point TSC mesh-near stencil contribution (one component) of one
        //! source.  R = (yee - delta) * cells with the integer support offset
        //! delta, so no grid origin is needed.
        ALPAKA_FN_ACC inline double dmeshComp(
            double const xs[3],
            double const urep[3],
            double const cells[3],
            double const nodeBase[3],
            double const sp[3],
            double const yee[3],
            int c)
        {
            double tau[3];
            long lo[3];
            long hi[3];
            for(int a = 0; a < 3; ++a)
            {
                tau[a] = (xs[a] - nodeBase[a]) / sp[a];
                lo[a] = static_cast<long>(::ceil(tau[a] - 1.5));
                hi[a] = static_cast<long>(::floor(tau[a] + 1.5));
            }
            double sum = 0.0;
            for(long ia = lo[0]; ia <= hi[0]; ++ia)
            {
                double const sa = static_cast<double>(ia) - tau[0];
                double const wa = dtsc(sa);
                if(wa == 0.0)
                    continue;
                for(long ib = lo[1]; ib <= hi[1]; ++ib)
                {
                    double const sb = static_cast<double>(ib) - tau[1];
                    double const wb = wa * dtsc(sb);
                    if(wb == 0.0)
                        continue;
                    for(long ic = lo[2]; ic <= hi[2]; ++ic)
                    {
                        double const sc = static_cast<double>(ic) - tau[2];
                        double const wc = wb * dtsc(sc);
                        if(wc == 0.0)
                            continue;
                        double const Rx = (yee[0] - static_cast<double>(ia)) * cells[0];
                        double const Ry = (yee[1] - static_cast<double>(ib)) * cells[1];
                        double const Rz = (yee[2] - static_cast<double>(ic)) * cells[2];
                        sum += wc * dpointComp(Rx, Ry, Rz, urep, c);
                    }
                }
            }
            return sum;
        }

        // ---- kernel --------------------------------------------------------

        struct FastNearKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* out6,
                double const* px,
                double const* py,
                double const* pz,
                double const* pux,
                double const* puy,
                double const* puz,
                double const* pq,
                double const* prux,
                double const* pruy,
                double const* pruz,
                bool stencilMesh,
                std::uint32_t nsrc,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                double ox,
                double oy,
                double oz,
                double spx,
                double spy,
                double spz,
                double cx,
                double cy,
                double cz,
                std::uint32_t chunks,
                DeviceParams prm,
                std::uint32_t total) const
            {
                // One thread per (component, node, source chunk): the six
                // staggered components are independent observations, so this
                // keeps per-thread state to one accumulator and multiplies the
                // number of resident blocks six-fold (the per-node work was
                // block-starved on small tiles).
                std::uint32_t const perComp = static_cast<std::uint32_t>(nx * ny * nz) * chunks;
                for(auto [idx] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{total}}))
                {
                    int const c = static_cast<int>(idx / perComp);
                    std::uint32_t const rem = idx - static_cast<std::uint32_t>(c) * perComp;
                    std::uint32_t const node = rem / chunks;
                    std::uint32_t const chunk = rem - node * chunks;
                    std::uint32_t const ii = node % nx;
                    std::uint32_t const jj = (node / nx) % ny;
                    std::uint32_t const kk = node / (nx * ny);

                    double const nodeBase[3] = {
                        ox + static_cast<double>(ii) * spx,
                        oy + static_cast<double>(jj) * spy,
                        oz + static_cast<double>(kk) * spz};
                    double const sp[3] = {spx, spy, spz};
                    double const cells[3] = {cx, cy, cz};
                    double sum = 0.0;
                    for(std::uint32_t q = chunk; q < nsrc; q += chunks)
                    {
                        double const x_src[3] = {px[q], py[q], pz[q]};
                        double const u[3] = {pux[q], puy[q], puz[q]};
                        double const urep[3] = {prux[q], pruy[q], pruz[q]};
                        double const qw = pq[q];
                        double const x_obs[3] = {
                            nodeBase[0] + prm.delta[c][0] * cx,
                            nodeBase[1] + prm.delta[c][1] * cy,
                            nodeBase[2] + prm.delta[c][2] * cz};
                        double E[3], B[3];
                        dshapeField(x_obs, x_src, u, cells, prm, E, B);
                        double comp = c < 3 ? E[c] : B[c - 3];
                        if(stencilMesh)
                            comp -= dmeshComp(x_src, urep, cells, nodeBase, sp, prm.delta[c], c);
                        double const unit = c < 3 ? prm.unitE : prm.unitB;
                        sum += unit * qw * comp;
                    }
                    std::uint32_t const o = static_cast<std::uint32_t>(c) * nx * ny * nz + node;
                    alpaka::onAcc::atomicAdd(acc, &out6[o], sum);
                }
            }
        };

        // ---- device context -------------------------------------------------

        struct Context
        {
            using Selector = decltype(alpaka::onHost::makeDeviceSelector(
                alpaka::api::cuda,
                alpaka::deviceKind::nvidiaGpu));
            using Device = decltype(std::declval<Selector&>().makeDevice(0));
            using Queue = decltype(std::declval<Device&>().makeQueue());

            Device device;
            Queue queue;

            Context() : device(makeDevice()), queue(device.makeQueue()) {}

            static Device makeDevice()
            {
                auto selector = alpaka::onHost::makeDeviceSelector(
                    alpaka::api::cuda,
                    alpaka::deviceKind::nvidiaGpu);
                if(!selector.isAvailable())
                    throw std::runtime_error("fast near-shell: no CUDA device available");
                return selector.makeDevice(0);
            }
        };

        Context& context()
        {
            static Context ctx;
            return ctx;
        }

        template<typename T>
        auto alloc1(Context& ctx, std::uint32_t n)
        {
            return alpaka::onHost::allocUnified<T>(ctx.device, alpaka::Vec<std::uint32_t, 1u>{n});
        }
    } // namespace

    bool fastNearDeviceAvailable()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::api::cuda,
            alpaka::deviceKind::nvidiaGpu);
        return selector.isAvailable();
    }

    std::vector<double> evalFastNearDevice(
        Bunch const& near,
        Bunch const& mesh,
        bool stencilMesh,
        std::size_t nx,
        std::size_t ny,
        std::size_t nz,
        double ox,
        double oy,
        double oz,
        double spx,
        double spy,
        double spz,
        double cx,
        double cy,
        double cz,
        DeviceParams const& params)
    {
        std::size_t const nodes = nx * ny * nz;
        std::vector<double> out(6 * nodes, 0.0);
        std::size_t const nsrc = near.size();
        if(nodes == 0 || nsrc == 0)
            return out;

        Context& ctx = context();
        std::uint32_t const n = static_cast<std::uint32_t>(nsrc);
        auto px = alloc1<double>(ctx, n);
        auto py = alloc1<double>(ctx, n);
        auto pz = alloc1<double>(ctx, n);
        auto pux = alloc1<double>(ctx, n);
        auto puy = alloc1<double>(ctx, n);
        auto puz = alloc1<double>(ctx, n);
        auto pq = alloc1<double>(ctx, n);
        auto prux = alloc1<double>(ctx, n);
        auto pruy = alloc1<double>(ctx, n);
        auto pruz = alloc1<double>(ctx, n);
        auto dout = alloc1<double>(ctx, static_cast<std::uint32_t>(6 * nodes));
        for(std::uint32_t i = 0; i < n; ++i)
        {
            px.data()[i] = near.x[i];
            py.data()[i] = near.y[i];
            pz.data()[i] = near.z[i];
            pux.data()[i] = near.ux[i];
            puy.data()[i] = near.uy[i];
            puz.data()[i] = near.uz[i];
            pq.data()[i] = near.weight[i] * near.charge[i];
            prux.data()[i] = mesh.ux.empty() ? near.ux[i] : mesh.ux[i];
            pruy.data()[i] = mesh.uy.empty() ? near.uy[i] : mesh.uy[i];
            pruz.data()[i] = mesh.uz.empty() ? near.uz[i] : mesh.uz[i];
        }
        for(std::size_t i = 0; i < 6 * nodes; ++i)
            dout.data()[i] = 0.0;

        std::uint32_t const targetThreads = 131072u;
        std::uint32_t chunks = static_cast<std::uint32_t>(
            (targetThreads + nodes - 1u) / nodes);
        if(chunks < 1u)
            chunks = 1u;
        if(chunks > n)
            chunks = n;
        std::uint32_t const total
            = 6u * static_cast<std::uint32_t>(nodes) * chunks;

        std::uint32_t const frame = 256u;
        ctx.queue.enqueue(
            alpaka::onHost::FrameSpec{(total + frame - 1u) / frame, frame},
            FastNearKernel{},
            dout.data(),
            px.data(),
            py.data(),
            pz.data(),
            pux.data(),
            puy.data(),
            puz.data(),
            pq.data(),
            prux.data(),
            pruy.data(),
            pruz.data(),
            stencilMesh,
            n,
            static_cast<std::uint32_t>(nx),
            static_cast<std::uint32_t>(ny),
            static_cast<std::uint32_t>(nz),
            ox,
            oy,
            oz,
            spx,
            spy,
            spz,
            cx,
            cy,
            cz,
            chunks,
            params,
            total);
        alpaka::onHost::wait(ctx.queue);
        for(std::size_t i = 0; i < 6 * nodes; ++i)
            out[i] = dout.data()[i];
        return out;
    }
} // namespace picBunch::pbfast
