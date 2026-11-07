// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/Deposit.hpp"

#include "picBunch/Kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace picBunch
{
    void Bunch::resize(std::size_t n)
    {
        x.resize(n);
        y.resize(n);
        z.resize(n);
        ux.resize(n);
        uy.resize(n);
        uz.resize(n);
        weight.resize(n);
        charge.resize(n);
        restMass.resize(n);
    }

    std::array<std::array<double, 3>, 2> Bunch::bounds() const
    {
        if(size() == 0)
            throw std::runtime_error("Bunch::bounds: empty bunch");
        std::array<std::array<double, 3>, 2> b{};
        b[0].fill(std::numeric_limits<double>::max());
        b[1].fill(-std::numeric_limits<double>::max());
        for(std::size_t p = 0; p < size(); ++p)
        {
            double const v[3] = {x[p], y[p], z[p]};
            for(int a = 0; a < 3; ++a)
            {
                b[0][a] = std::min(b[0][a], v[a]);
                b[1][a] = std::max(b[1][a], v[a]);
            }
        }
        return b;
    }

    template<Shape S>
    std::vector<double> depositShape(Bunch const& bunch, Grid const& grid)
    {
        std::vector<double> rho(grid.size(), 0.0);
        std::size_t const nx = grid.n[0];
        std::size_t const ny = grid.n[1];
        std::size_t const nz = grid.n[2];
        double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
        double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};
        int const reach = static_cast<int>(std::ceil(shapeSupport<S>()));

        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            double const q = bunch.weight[p] * bunch.charge[p];
            int base[3];
            double frac[3];
            for(int a = 0; a < 3; ++a)
            {
                double const t = (pos[a] - org[a]) / sp[a];
                base[a] = static_cast<int>(std::floor(t));
                frac[a] = t - base[a];
            }
            for(int da = -reach; da <= reach; ++da)
            {
                long const ia = static_cast<long>(base[0]) + da;
                if(ia < 0 || ia >= static_cast<long>(nx))
                    continue;
                double const wa = shapeWeight<S>( da - frac[0]);
                if(wa == 0.0)
                    continue;
                for(int db = -reach; db <= reach; ++db)
                {
                    long const ib = static_cast<long>(base[1]) + db;
                    if(ib < 0 || ib >= static_cast<long>(ny))
                        continue;
                    double const wb = wa * shapeWeight<S>( db - frac[1]);
                    if(wb == 0.0)
                        continue;
                    for(int dc = -reach; dc <= reach; ++dc)
                    {
                        long const ic = static_cast<long>(base[2]) + dc;
                        if(ic < 0 || ic >= static_cast<long>(nz))
                            continue;
                        double const wc = wb * shapeWeight<S>( dc - frac[2]);
                        if(wc == 0.0)
                            continue;
                        rho[static_cast<std::size_t>(ia)
                            + nx * (static_cast<std::size_t>(ib) + ny * static_cast<std::size_t>(ic))]
                            += wc * q;
                    }
                }
            }
        }
        return rho;
    }

    template<Shape S>
    std::vector<double> depositPaddedShape(Bunch const& bunch, Grid const& grid, double refinement)
    {
        if(refinement <= 0.0)
            throw std::invalid_argument("depositTSCPaddedHost: refinement must be positive");
        if(grid.fftPadding < 2u)
            throw std::invalid_argument(
                "depositTSCPaddedHost: a source halo needs fftPadding >= 2");
        std::size_t const nx = grid.n[0];
        std::size_t const ny = grid.n[1];
        std::size_t const nz = grid.n[2];
        std::size_t const Mx = nx * grid.fftPadding;
        std::size_t const My = ny * grid.fftPadding;
        std::size_t const Mz = nz * grid.fftPadding;
        if(Mx < nx + 2u || My < ny + 2u || Mz < nz + 2u)
            throw std::invalid_argument(
                "depositTSCPaddedHost: the padded extent must leave room for the source halo");
        double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
        double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};
        // The TSC shape is sampled in node units: grid.spacing already carries the
        // refinement, so no extra refinement factor belongs in the argument.
        double const support = shapeSupport<S>();
        std::vector<double> rho(Mx * My * Mz, 0.0);
        auto const wrap = [](long i, std::size_t m) {
            long const mm = static_cast<long>(m);
            long r = i % mm;
            return r < 0 ? r + mm : r;
        };

        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            double const q = bunch.weight[p] * bunch.charge[p];
            double t[3];
            long lo[3];
            long hi[3];
            for(int a = 0; a < 3; ++a)
            {
                t[a] = (pos[a] - org[a]) / sp[a];
                lo[a] = static_cast<long>(std::ceil(t[a] - support));
                hi[a] = static_cast<long>(std::floor(t[a] + support));
            }
            for(long ia = lo[0]; ia <= hi[0]; ++ia)
            {
                double const wa = shapeWeight<S>( static_cast<double>(ia) - t[0]);
                if(wa == 0.0)
                    continue;
                long const ma = wrap(ia, Mx);
                for(long ib = lo[1]; ib <= hi[1]; ++ib)
                {
                    double const wb = wa * shapeWeight<S>( static_cast<double>(ib) - t[1]);
                    if(wb == 0.0)
                        continue;
                    long const mb = wrap(ib, My);
                    for(long ic = lo[2]; ic <= hi[2]; ++ic)
                    {
                        double const wc = wb * shapeWeight<S>( static_cast<double>(ic) - t[2]);
                        if(wc == 0.0)
                            continue;
                        long const mc = wrap(ic, Mz);
                        rho[static_cast<std::size_t>(ma)
                            + Mx
                                * (static_cast<std::size_t>(mb)
                                   + My * static_cast<std::size_t>(mc))]
                            += wc * q;
                    }
                }
            }
        }
        return rho;
    }

    std::vector<double> depositTSC(Bunch const& bunch, Grid const& grid)
    {
        switch(grid.shape)
        {
        case Shape::CIC:
            return depositShape<Shape::CIC>(bunch, grid);
        case Shape::PQS:
            return depositShape<Shape::PQS>(bunch, grid);
        case Shape::TSC:
        default:
            return depositShape<Shape::TSC>(bunch, grid);
        }
    }

    std::vector<double> depositTSCPaddedHost(Bunch const& bunch, Grid const& grid, double refinement)
    {
        switch(grid.shape)
        {
        case Shape::CIC:
            return depositPaddedShape<Shape::CIC>(bunch, grid, refinement);
        case Shape::PQS:
            return depositPaddedShape<Shape::PQS>(bunch, grid, refinement);
        case Shape::TSC:
        default:
            return depositPaddedShape<Shape::TSC>(bunch, grid, refinement);
        }
    }

    bool allParticlesInside(Bunch const& bunch, Grid const& grid)
    {
        double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
        double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};
        double const hi[3] = {
            org[0] + static_cast<double>(grid.n[0] - 1) * sp[0],
            org[1] + static_cast<double>(grid.n[1] - 1) * sp[1],
            org[2] + static_cast<double>(grid.n[2] - 1) * sp[2]};
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
            for(int a = 0; a < 3; ++a)
                if(pos[a] < org[a] || pos[a] > hi[a])
                    return false;
        }
        return true;
    }

} // namespace picBunch
