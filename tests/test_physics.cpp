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
// sum (no periodic images).  P1 is far enough from the source that the
// discrete TSC result matches at 1.6e-7 and is asserted at 1e-6.  P2 is close
// to the source: the kRefinement = 1 TSC assignment quadrature is too coarse
// there, and the measured residual is 1.4e-2.  That is a known limitation of
// the v0.1 deposit kernel, not an FFT error; it is asserted at 2e-2.

#include "picBunch/Compute.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    struct Point
    {
        picBunch::Idx3 node;
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
             1.0e-6}, // far field: measured 1.6e-7
            {{9, 4, 5},
             {-1.02519670526764164e-10,
              -1.09195348836882690e-11,
              -6.75390531206454471e-12,
              1.14360184508520855e-20,
              -1.15885229998238335e-19,
              0.0},
             2.0e-2}, // near field: measured 1.4e-2, coarse TSC quadrature at kRefinement=1
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
    EXPECT_EQ(field.grid.n, (picBunch::Idx3{10, 10, 10}));
    EXPECT_EQ(field.grid.origin, (picBunch::Vec3{1.0, 1.0, 1.0}));
    EXPECT_EQ(field.grid.fftPadding, 2u);

    // Normalize against the reference point-field scale q/(4 pi eps0) / (1 m)^2,
    // the scale the expected-value provenance quotes (not the component maximum,
    // which would hide the far-field agreement behind a much smaller number).
    double const scale = picBunch::kCoulombConstant * picBunch::kElementaryCharge;
    ASSERT_GT(scale, 0.0);
    for(auto const& point : referencePoints())
    {
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

// Multi-particle reference.  The values in
// tests/data/physics_multiparticle.csv are a sum over three electrons that
// share u = (0,0,5):
//   (4.0,4.0,4.0), (4.7,5.1,4.3), (6.2,5.8,6.1) m, q = -e, weight = 1.
// They were produced by the independent quadrature in
// ~/pr24/pr24w53-ref/indfield.py on rosi_v100:
//   cd ~/pr24/pr24w53-ref && env -u LD_PRELOAD PYTHONPATH=. python3 /tmp/gen_one.py <i> <j> <k> <component>
// with integrate_point(x_obs, x_src, u, 1, 1, 1, order=12, refine=3) and the
// same k*q conversion as the single-particle reference.  The reference file is
// not a git checkout; its SHA-256 is
//   a135598041a3dbe134667585631ea53fc6bdb7cf7a84419b5a8d762202599519.
// The CSV holds numbers only: i,j,k, Ex,Ey,Ez,Bx,By,Bz, relative tolerance.
// makeGrid gives n = (12,12,12) and origin (0,0,0) for this bunch.
TEST(Physics, MultiParticleAgainstIndependentReference)
{
    picBunch::Bunch bunch;
    std::array<picBunch::Vec3, 3> const positions{{
        {4.0, 4.0, 4.0},
        {4.7, 5.1, 4.3},
        {6.2, 5.8, 6.1},
    }};
    for(auto const& p : positions)
    {
        std::size_t const i = bunch.size();
        bunch.resize(i + 1);
        bunch.x[i] = p[0];
        bunch.y[i] = p[1];
        bunch.z[i] = p[2];
        bunch.ux[i] = 0.0;
        bunch.uy[i] = 0.0;
        bunch.uz[i] = 5.0;
        bunch.weight[i] = 1.0;
        bunch.charge[i] = -picBunch::kElementaryCharge;
    }

    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.marginCells = 4.0;

    picBunch::Field const field = picBunch::computeField(bunch, options);
    EXPECT_EQ(field.grid.n, (picBunch::Idx3{12, 12, 12}));
    EXPECT_EQ(field.grid.origin, (picBunch::Vec3{0.0, 0.0, 0.0}));
    EXPECT_EQ(field.grid.fftPadding, 2u);

    std::ifstream input(std::string(PICBUNCH_TEST_DATA_DIR) + "/physics_multiparticle.csv");
    ASSERT_TRUE(input.good()) << "cannot open the multi-particle reference CSV";
    double const scale = picBunch::kCoulombConstant * picBunch::kElementaryCharge;
    std::string line;
    std::size_t rows = 0;
    while(std::getline(input, line))
    {
        if(line.empty())
            continue;
        std::istringstream stream(line);
        std::string token;
        std::vector<double> values;
        while(std::getline(stream, token, ','))
            values.push_back(std::stod(token));
        ASSERT_EQ(values.size(), 10u) << "bad CSV row: " << line;
        std::size_t const i = static_cast<std::size_t>(values[0]);
        std::size_t const j = static_cast<std::size_t>(values[1]);
        std::size_t const k = static_cast<std::size_t>(values[2]);
        std::size_t const flat = i + field.grid.n[0] * (j + field.grid.n[1] * k);
        double const relativeTolerance = values[9];
        for(int c = 0; c < 6; ++c)
            EXPECT_LE(std::abs(field.data[c][flat] - values[3 + c]), relativeTolerance * scale)
                << "node (" << i << "," << j << "," << k << ") component " << c
                << " expected " << values[3 + c] << " got " << field.data[c][flat];
        ++rows;
    }
    EXPECT_EQ(rows, 3u);
}

// ---------------------------------------------------------------------------
// Absolute-magnitude guards (task W3n).
//
// These compare the shipped field against the analytic Coulomb field in SI
// (V/m) at a Yee-staggered observer node.  The reference is absolute and
// convention-free: k*q*R/r^3.  No cell-volume convention is assumed, so a
// regression that (wrongly) multiplies the read-out scale by the physical node
// volume -- cellx*celly*cellz -- fails here by the full volume factor (e.g.
// 1e-21 for 1e-7 m cells, 4 for 1,1,4 m cells).  The deviation that does
// remain is TSC deposit discretisation, which shrinks with distance; that is
// why the anisotropic case is asserted at a far node.
namespace
{
    std::array<double, 6> coulombField(
        picBunch::Vec3 const& obs,
        picBunch::Vec3 const& src,
        double q)
    {
        std::array<double, 6> out{};
        double Rx = obs[0] - src[0];
        double Ry = obs[1] - src[1];
        double Rz = obs[2] - src[2];
        double const r3 = (Rx * Rx + Ry * Ry + Rz * Rz) * std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
        if(r3 <= 0.0)
            return out;
        double const f = picBunch::kCoulombConstant * q / r3;
        out[0] = f * Rx;
        out[1] = f * Ry;
        out[2] = f * Rz;
        return out;
    }

    //! One stationary electron on an explicit even grid; max |field-analytic|/|analytic|
    //! over the six components at the on-axis observer node (obsX, charge, charge).
    double maxCoulombRelError(
        picBunch::Vec3 const& cell,
        std::size_t N,
        std::size_t charge,
        std::size_t obsX)
    {
        picBunch::Bunch bunch;
        bunch.resize(1);
        bunch.x[0] = static_cast<double>(charge) * cell[0];
        bunch.y[0] = static_cast<double>(charge) * cell[1];
        bunch.z[0] = static_cast<double>(charge) * cell[2];
        bunch.ux[0] = 0.0;
        bunch.uy[0] = 0.0;
        bunch.uz[0] = 0.0;
        bunch.weight[0] = 1.0;
        bunch.charge[0] = -picBunch::kElementaryCharge;

        picBunch::ComputeOptions options;
        options.tolerance = 1e-12;
        options.cellSize = cell;
        options.gridSize = {N, N, N};
        options.gridOrigin = {0.0, 0.0, 0.0};

        picBunch::Field const field = picBunch::computeField(bunch, options);
        picBunch::Vec3 const src{bunch.x[0], bunch.y[0], bunch.z[0]};
        std::size_t const flat = obsX + field.grid.n[0] * (charge + field.grid.n[1] * charge);
        double maxRel = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            picBunch::Vec3 obs{};
            for(int a = 0; a < 3; ++a)
            {
                std::size_t const node = (a == 0) ? obsX : charge;
                obs[a] = (static_cast<double>(node) + picBunch::kYeeOffset[c][a]) * cell[a];
            }
            std::array<double, 6> const ref = coulombField(obs, src, bunch.charge[0]);
            double const actual = field.data[c][flat];
            if(std::abs(ref[c]) > 0.0)
                maxRel = std::max(maxRel, std::abs(actual - ref[c]) / std::abs(ref[c]));
        }
        return maxRel;
    }
} // namespace

// 1e-7 m isotropic cells: the shipped field must be the physical Coulomb field in
// V/m, not 1e-21 of it.  Measured 7.8e-5 at 12 cells on both the CUDA (V100) and
// CPU (FFTW) backends.
TEST(Physics, AbsoluteMagnitudeIsotropic100nmCells)
{
    double const rel = maxCoulombRelError({1e-7, 1e-7, 1e-7}, /*N*/ 48, /*charge*/ 16, /*obsX*/ 28);
    EXPECT_LT(rel, 1.0e-3) << "1e-7 m isotropic cells disagree with the absolute Coulomb field by "
                           << rel << "; a missing node-volume factor would give ~1";
}

// 1,1,4 m anisotropic cells at a far node: the remaining deviation is TSC
// deposit discretisation (measured 1.03e-2 at 40 cells), not a uniform 4x
// (1/0.25) volume factor, which would give ~3.
TEST(Physics, AbsoluteMagnitudeAnisotropic114Cells)
{
    double const rel = maxCoulombRelError({1.0, 1.0, 4.0}, /*N*/ 64, /*charge*/ 16, /*obsX*/ 56);
    EXPECT_LT(rel, 2.0e-2) << "1,1,4 m anisotropic cells disagree with the absolute Coulomb field by "
                           << rel << "; a node-volume factor of 4 would give ~3";
}
// ---------------------------------------------------------------------------
// R2C parity (task W3o).
//
// The R2C half-spectrum on the last (x) axis has Mx/2+1 planes when Mx is even
// (DC at 0 and the Nyquist plane at Mx/2) and (Mx+1)/2 planes with NO Nyquist
// plane when Mx is odd.  Before this change the pipeline required an even Mx
// (computeField threw) and the kernel centring assumed one, so an odd padded
// extent was rejected rather than handled.  The centred index in
// KernelBuildKernel now splits on ``i <= (M-1)/2`` (identical to ``i < M/2``
// for even M, correct for odd M) and the even-only guards are relaxed.
//
// This test runs the same single-particle free-space TSC convolution with an
// even padded extent (padding 2 -> Mx=62) and an odd one (padding 3 -> Mx=93)
// on the same 31^3 output grid.  Neither box wraps (the source-observer lag is
// far under half a box), so the two fields must agree to transform roundoff.
// Without the parity support the odd run throws, so this fails.
TEST(Physics, OddExtentR2cParity)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 15.0;
    bunch.y[0] = 15.0;
    bunch.z[0] = 15.0;
    bunch.ux[0] = 0.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    auto run = [&](unsigned padding)
    {
        picBunch::ComputeOptions o;
        o.tolerance = 1e-12;
        o.cellSize = {1.0, 1.0, 1.0};
        o.gridSize = {31, 31, 31};
        o.gridOrigin = {0.0, 0.0, 0.0};
        o.fftPadding = padding;
        return picBunch::computeField(bunch, o);
    };

    picBunch::Field const even = run(2u); // Mx = 31*2 = 62 (even)
    picBunch::Field const odd = run(3u);  // Mx = 31*3 = 93 (odd, no Nyquist plane)
    ASSERT_EQ(even.grid.n, odd.grid.n);
    EXPECT_EQ(even.grid.n[0] * even.grid.fftPadding % 2u, 0u);
    EXPECT_EQ(odd.grid.n[0] * odd.grid.fftPadding % 2u, 1u) << "the odd run must use an odd Mx";

    double maxRel = 0.0;
    for(int c = 0; c < 6; ++c)
    {
        double scale = 0.0;
        for(std::size_t f = 0; f < even.grid.size(); ++f)
            scale = std::max(scale, std::abs(even.data[c][f]));
        for(std::size_t f = 0; f < even.grid.size(); ++f)
        {
            double const d = std::abs(even.data[c][f] - odd.data[c][f]);
            if(scale > 0.0)
                maxRel = std::max(maxRel, d / scale);
        }
    }
    EXPECT_LT(maxRel, 1.0e-6)
        << "odd padded extent (Mx=93) disagrees with the even extent (Mx=62) by "
        << maxRel << " relative; the R2C path is not parity-correct for an odd last axis";
}

// Regression for P1-2: a source on the boundary of a small compute box must
// deliver the same field as the same physical source placed well inside a
// larger box.  Before the source-halo fix the small box clipped the TSC support
// at node 0, so the field at the boundary nodes was off by the missing halo
// weight (about 1/8 of the charge at the nearest nodes).
TEST(Physics, BoundarySourceMatchesLargerComputeBox)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x[0] = 0.0;
    bunch.y[0] = 0.0;
    bunch.z[0] = 0.0;
    bunch.ux[0] = 0.0;
    bunch.uy[0] = 0.0;
    bunch.uz[0] = 0.0;
    bunch.weight[0] = 1.0;
    bunch.charge[0] = -picBunch::kElementaryCharge;

    auto const fieldFor = [&](picBunch::Idx3 const& n,
                              picBunch::Vec3 const& origin,
                              bool deterministic) {
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3; // padding 2
        options.cellSize = {1.0, 1.0, 1.0};
        options.gridSize = n;
        options.gridOrigin = origin;
        options.deterministicDeposit = deterministic;
        return picBunch::computeField(bunch, options);
    };

    // Check both the accelerator deposit and the serial host deposit: the
    // source halo must be identical in the two paths.
    for(bool const deterministic : {false, true})
    {
        // The particle sits on node (0,0,0) of the small box.  In the large
        // box it sits on node (8,8,8), so every small node (i,j,k) maps to
        // (i+8,j+8,k+8).
        picBunch::Field const small = fieldFor({8, 8, 8}, {0.0, 0.0, 0.0}, deterministic);
        picBunch::Field const large
            = fieldFor({32, 32, 32}, {-8.0, -8.0, -8.0}, deterministic);
        ASSERT_EQ(small.grid.n, (picBunch::Idx3{8, 8, 8}));
        ASSERT_EQ(large.grid.n, (picBunch::Idx3{32, 32, 32}));

        double const scale = picBunch::kCoulombConstant * std::abs(bunch.charge[0]);
        std::array<picBunch::Idx3, 5> const nodes{{
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 1},
        }};
        for(auto const& node : nodes)
        {
            std::size_t const si = node[0]
                + small.grid.n[0] * (node[1] + small.grid.n[1] * node[2]);
            std::size_t const li = (node[0] + 8u)
                + large.grid.n[0] * ((node[1] + 8u) + large.grid.n[1] * (node[2] + 8u));
            for(int c = 0; c < 6; ++c)
                EXPECT_NEAR(small.data[c][si], large.data[c][li], 2.0e-2 * scale)
                    << "deterministic=" << deterministic << " node (" << node[0] << ","
                    << node[1] << "," << node[2] << ") component " << c << " small "
                    << small.data[c][si] << " large " << large.data[c][li];
        }
    }
}

TEST(Physics, PqsChangesTheMeshField)
{
    picBunch::Bunch bunch;
    bunch.resize(1);
    bunch.x = {5.0};
    bunch.y = {5.0};
    bunch.z = {5.0};
    bunch.ux = {0.0};
    bunch.uy = {0.0};
    bunch.uz = {10.0};
    bunch.weight = {1.0};
    bunch.charge = {-picBunch::kElementaryCharge};

    picBunch::ComputeOptions options;
    options.tolerance = 1e-3;
    options.cellSize = {1.0, 1.0, 1.0};
    options.gridSize = {10, 10, 10};
    options.gridOrigin = {1.0, 1.0, 1.0};
    options.shape = picBunch::Shape::TSC;
    auto const tsc = picBunch::computeField(bunch, options);
    options.shape = picBunch::Shape::PQS;
    auto const pqs = picBunch::computeField(bunch, options);
    double peak = 0.0;
    double diff = 0.0;
    for(int c = 0; c < 6; ++c)
    {
        ASSERT_EQ(tsc.data[c].size(), pqs.data[c].size());
        for(std::size_t l = 0; l < tsc.data[c].size(); ++l)
        {
            peak = std::max(peak, std::abs(tsc.data[c][l]));
            diff = std::max(diff, std::abs(tsc.data[c][l] - pqs.data[c][l]));
        }
    }
    ASSERT_GT(peak, 0.0);
    EXPECT_GT(diff, 1.0e-6 * peak) << "PQS must change the deposited field";
}
