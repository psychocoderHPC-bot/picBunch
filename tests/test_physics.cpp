// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Group 5: one particle vs an independent high-precision reference.
//
// Configuration: one electron (q = -e, weight 1) at (5,5,5) m, u = (0,0,10),
// cellSize = (1,1,1) m, marginCells = 4, tolerance = 1e-3 (fftPadding 2).
// makeGrid then gives n = (10,10,10) with origin (1,1,1).
//
// Reference provenance (values hardcoded below):
//   host      rosi_v100
//   code      ~/pr24/pr24w53-ref/indfield.py (Gauss-Legendre, order 12,
//             partition refine 3, knot-split and a singularity-cancelling power
//             transform; independent formulation, no imported tables)
//   invocation
//     cd ~/pr24/pr24w53-ref && env -u LD_PRELOAD python3 -c '
//       import numpy as np, indfield as I
//       q=-1.602176634e-19; k=8.9875517873681764e9; c=299792458.0
//       src=np.array([5.,5.,5.]); u=np.array([0.,0.,10.]); origin=np.array([1.,1.,1.])
//       for idx in [(4,4,9),(9,4,5)]:
//         for comp in range(6):
//           xobs=origin+np.array(idx,float)+I.OFFSETS[comp]
//           e,b=I.integrate_point(xobs,src,u,1.,1.,1.,order=12,refine=3)
//           En=np.concatenate([e,b])[comp]
//           print(idx, comp, k*q*(En if comp<3 else En/c))'
//   Output (E in V/m, B in T), measured |discrete TSC - continuum reference|:
//     point P1 node (4,4,9), offset (0,0,5), worst 1.6e-7 of the point scale
//       E = (-6.06397185445433445e-14, -6.06397185445428901e-14,
//            -4.83305196711727292e-13)
//       B = ( 1.49564992178825176e-22, -1.49564992178824000e-22, 0)
//     point P2 node (9,4,5), offset (5,0,1), worst 1.4e-2 of the point scale
//       E = (-1.02519670526764164e-10, -1.09195348836882690e-11,
//            -6.75390531206454471e-12)
//       B = ( 1.14360184508520855e-20, -1.15885229998238335e-19, 0)
//
// The reference is free space.  With fftPadding = 2 every source-observer lag
// fits inside the FFT box, so the discrete result equals the free-space TSC
// sum (no periodic images).  The residual difference at P2 is the coarseness
// of the kRefinement = 1 TSC quadrature near the source, not an FFT error; P1
// is far enough that it is 1.6e-7.

#include "picBunch/Compute.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    struct Point
    {
        std::array<std::size_t, 3> node;
        std::array<double, 6> expected;
        double tolerance;
    };

    // Components: E_x,E_y,E_z,B_x,B_y,B_z.
    std::vector<Point> referencePoints()
    {
        return {
            {{4, 4, 9},
             {-6.06397185445433445e-14,
              -6.06397185445428901e-14,
              -4.83305196711727292e-13,
              1.49564992178825176e-22,
              -1.49564992178824000e-22,
              0.0},
             1.0e-3},
            {{9, 4, 5},
             {-1.02519670526764164e-10,
              -1.09195348836882690e-11,
              -6.75390531206454471e-12,
              1.14360184508520855e-20,
              -1.15885229998238335e-19,
              0.0},
             2.0e-2},
        };
    }
} // namespace

TEST(Physics, OneParticleAgainstIndependentReference)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 5.0;
    bunch.y[0] = 5.0;
    bunch.z[0] = 5.0;
    bunch.ux[0] = 0.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 10.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 4.0;

    picBunch::Field const field = picBunch::computeField(bunch, options);
    EXPECT_EQ(field.grid.n, (std::array<std::size_t, 3>{10, 10, 10}));
    EXPECT_EQ(field.grid.origin, (std::array<double, 3>{1.0, 1.0, 1.0}));
    EXPECT_EQ(field.grid.fftPadding, 2u);

    for(auto const& point : referencePoints())
    {
        double const scale = [&] {
            double s = 0.0;
            for(double value : point.expected)
                s = std::max(s, std::abs(value));
            return s;
        }();
        ASSERT_GT(scale, 0.0);
        std::size_t const flat = point.node[0]
            + field.grid.n[0] * (point.node[1] + field.grid.n[1] * point.node[2]);
        for(int c = 0; c < 6; ++c)
        {
            double const actual = field.data[c][flat];
            EXPECT_LE(std::abs(actual - point.expected[c]), point.tolerance * scale)
                << "node (" << point.node[0] << "," << point.node[1] << "," << point.node[2]
                << ") component " << c << " expected " << point.expected[c] << " got " << actual;
        }
    }
}
