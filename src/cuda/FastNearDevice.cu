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
                    double const mm = gam - 1.0;
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
                                            double const tx = 0.5 * prm.oxi[ix] + 0.5;
                                            double const ty = 0.5 * prm.oxi[iy] + 0.5;
                                            double const tz = 0.5 * prm.oxi[iz] + 0.5;
                                            double const tx2 = tx * tx;
                                            double const ty2 = ty * ty;
                                            double const tz2 = tz * tz;
                                            double const px = L[0] * tx2 * tx2 * tx;
                                            double const py = L[1] * ty2 * ty2 * ty;
                                            double const pz = L[2] * tz2 * tz2 * tz;
                                            double const jx
                                                = 0.5 * L[0] * 5.0 * tx2 * tx2 * prm.owi[ix];
                                            double const jy
                                                = 0.5 * L[1] * 5.0 * ty2 * ty2 * prm.owi[iy];
                                            double const jz
                                                = 0.5 * L[2] * 5.0 * tz2 * tz2 * prm.owi[iz];
                                            double const d[3] = {sx * px, sy * py, sz * pz};
                                            double const dn
                                                = d[0] * betaH[0] + d[1] * betaH[1] + d[2] * betaH[2];
                                            double const md[3] = {
                                                d[0] + mm * dn * betaH[0],
                                                d[1] + mm * dn * betaH[1],
                                                d[2] + mm * dn * betaH[2]};
                                            double const rho2
                                                = md[0] * md[0] + md[1] * md[1] + md[2] * md[2];
                                            double const rho = std::sqrt(rho2);
                                            double const f = -gam / (rho2 * rho) * jx * jy * jz;
                                            acc[0] += f * d[0];
                                            acc[1] += f * d[1];
                                            acc[2] += f * d[2];
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
                for(auto [idx] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{total}}))
                {
                    std::uint32_t const node = idx / chunks;
                    std::uint32_t const chunk = idx - node * chunks;
                    std::uint32_t const ii = node % nx;
                    std::uint32_t const jj = (node / nx) % ny;
                    std::uint32_t const kk = node / (nx * ny);

                    double acc6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    for(std::uint32_t q = chunk; q < nsrc; q += chunks)
                    {
                        double const x_src[3] = {px[q], py[q], pz[q]};
                        double const u[3] = {pux[q], puy[q], puz[q]};
                        double const qw = pq[q];
                        for(int c = 0; c < 6; ++c)
                        {
                            double const x_obs[3] = {
                                ox + static_cast<double>(ii) * spx + prm.delta[c][0] * cx,
                                oy + static_cast<double>(jj) * spy + prm.delta[c][1] * cy,
                                oz + static_cast<double>(kk) * spz + prm.delta[c][2] * cz};
                            double const cells[3] = {cx, cy, cz};
                            double E[3], B[3];
                            dshapeField(x_obs, x_src, u, cells, prm, E, B);
                            double const comp = c < 3 ? E[c] : B[c - 3];
                            double const unit = c < 3 ? prm.unitE : prm.unitB;
                            acc6[c] += unit * qw * comp;
                        }
                    }
                    for(int c = 0; c < 6; ++c)
                    {
                        std::uint32_t const o = static_cast<std::uint32_t>(c) * nx * ny * nz + node;
                        alpaka::onAcc::atomicAdd(acc, &out6[o], acc6[c]);
                    }
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
            = static_cast<std::uint32_t>(nodes) * chunks;

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
