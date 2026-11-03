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

    std::vector<double> depositTSC(Bunch const& bunch, Grid const& grid)
    {
        std::vector<double> rho(grid.size(), 0.0);
        std::size_t const nx = grid.n[0];
        std::size_t const ny = grid.n[1];
        std::size_t const nz = grid.n[2];
        double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
        double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};

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
            for(int da = -1; da <= 2; ++da)
            {
                long const ia = static_cast<long>(base[0]) + da;
                if(ia < 0 || ia >= static_cast<long>(nx))
                    continue;
                double const wa = tscShape(da - frac[0]);
                if(wa == 0.0)
                    continue;
                for(int db = -1; db <= 2; ++db)
                {
                    long const ib = static_cast<long>(base[1]) + db;
                    if(ib < 0 || ib >= static_cast<long>(ny))
                        continue;
                    double const wb = wa * tscShape(db - frac[1]);
                    if(wb == 0.0)
                        continue;
                    for(int dc = -1; dc <= 2; ++dc)
                    {
                        long const ic = static_cast<long>(base[2]) + dc;
                        if(ic < 0 || ic >= static_cast<long>(nz))
                            continue;
                        double const wc = wb * tscShape(dc - frac[2]);
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
