// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 6: padding control.  A no-padding run (fftPadding = 1) must reproduce
// the wraparound artefact; a padded run must agree with a more-padded run.
// This proves that free-space handling is real and not assumed.
//
// Bunch: 27 charges on an asymmetric 3x3x3 lattice spanning about 40 cells
// (marginCells = 4), all at rest.  With fftPadding = 1 the FFT box equals the
// output box, so source-observer lags beyond half the box wrap; with
// fftPadding = 2 every lag fits and the result is the free-space TSC sum.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace
{
    picBunch::Bunch latticeBunch()
    {
        picBunch::Bunch bunch;
        std::array<double, 3> const xs{10.0, 31.0, 50.0};
        std::array<double, 3> const ys{12.0, 30.0, 51.0};
        std::array<double, 3> const zs{9.0, 29.0, 50.0};
        for(std::size_t ix = 0; ix < 3; ++ix)
            for(std::size_t iy = 0; iy < 3; ++iy)
                for(std::size_t iz = 0; iz < 3; ++iz)
                {
                    std::size_t const p = bunch.size();
                    bunch.resize(p + 1);
                    bunch.x[p] = xs[ix];
                    bunch.y[p] = ys[iy];
                    bunch.z[p] = zs[iz];
                    bunch.ux[p] = 0.0;
                    bunch.uy[p] = 0.0;
                    bunch.uz[p] = 0.0;
                    // Deliberately non-uniform weights so periodic images do
                    // not cancel by lattice symmetry.
                    bunch.weight[p] = 0.5 + 0.25 * static_cast<double>((ix + 2 * iy + 3 * iz) % 4);
                    bunch.charge[p] = -picBunch::kElementaryCharge;
                }
        return bunch;
    }

    double maxDifference(picBunch::Field const& a, picBunch::Field const& b)
    {
        double result = 0.0;
        for(int c = 0; c < 6; ++c)
            for(std::size_t l = 0; l < a.data[c].size(); ++l)
                result = std::max(result, std::abs(a.data[c][l] - b.data[c][l]));
        return result;
    }

    //! Largest difference away from the outermost ``skip`` nodes per axis.
    double maxInteriorDifference(picBunch::Field const& a, picBunch::Field const& b, std::size_t skip)
    {
        double result = 0.0;
        auto const& n = a.grid.n;
        for(int c = 0; c < 6; ++c)
            for(std::size_t k = skip; k + skip < n[2]; ++k)
                for(std::size_t j = skip; j + skip < n[1]; ++j)
                    for(std::size_t i = skip; i + skip < n[0]; ++i)
                    {
                        std::size_t const l = i + n[0] * (j + n[1] * k);
                        result = std::max(result, std::abs(a.data[c][l] - b.data[c][l]));
                    }
        return result;
    }

    double maxField(picBunch::Field const& field)
    {
        double result = 0.0;
        for(auto const& component : field.data)
            for(double value : component)
                result = std::max(result, std::abs(value));
        return result;
    }
} // namespace

TEST(Padding, NoPaddingReproducesWraparoundAndPaddingConverges)
{
    picBunch::Bunch const bunch = latticeBunch();
    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 4.0;

    options.fftPadding = 1;
    picBunch::Field const unpadded = picBunch::computeField(bunch, options);
    options.fftPadding = 2;
    picBunch::Field const padded = picBunch::computeField(bunch, options);
    options.fftPadding = 3;
    picBunch::Field const paddedMore = picBunch::computeField(bunch, options);

    ASSERT_EQ(unpadded.grid.n, padded.grid.n);
    ASSERT_EQ(unpadded.grid.fftPadding, 1u);
    ASSERT_EQ(padded.grid.fftPadding, 2u);

    double const scale = maxField(padded);
    ASSERT_GT(scale, 0.0);
    double const wrapError = maxDifference(unpadded, padded) / scale;
    double const wrapInterior = maxInteriorDifference(unpadded, padded, 2u) / scale;
    double const paddingConverged = maxDifference(padded, paddedMore) / scale;
    std::printf(
        "padding control: max|pad1-pad2|/max|field|=%.6e, interior=%.6e, "
        "max|pad2-pad3|/max|field|=%.6e\n",
        wrapError,
        wrapInterior,
        paddingConverged);

    // The wraparound artefact is the known ~8.9e-3 effect; it must be clearly
    // present without padding.
    EXPECT_GT(wrapError, 1.0e-3) << "padding=1 must show a wraparound artefact";
    EXPECT_LT(wrapError, 1.0e-1) << "wraparound artefact must be a small correction";
    EXPECT_GT(wrapInterior, 1.0e-3) << "the interior must also show the artefact";

    // Padding 2 is already converged: adding more padding changes nothing
    // beyond FFT roundoff.
    EXPECT_LT(paddingConverged, 1.0e-9) << "padding=2 must be converged";
}
