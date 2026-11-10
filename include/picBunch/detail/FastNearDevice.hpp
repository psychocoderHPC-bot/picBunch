// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Device backend of the fast near-field evaluator (FastNear.hpp).
//
// The per-(source, node) algebra is identical to the host evaluator: the
// eight-term asinh singular box on the TSC support box, the plain
// Gauss-Legendre remainder on the subtracted integrand (S - W*)K over the
// knot/s* panels.  Oblique in-support pairs use the d-space ray integral
// (exact radial Gauss per knot interval, beta-aligned angular rule) in the
// device kernel.  One thread owns one output node and a strided chunk of the near
// sources, accumulates the six components, and atomically adds its partial
// sum.  The source is plain alpaka C++ and is compiled per backend (host,
// CUDA, HIP) from src/FastNearDevice.cpp; no native accelerator code is used.
#pragma once

#include "picBunch/Types.hpp"
#include "picBunch/detail/FastNearConstants.hpp"
#include "picBunch/detail/ObliqueOrder.hpp"

#include <cstddef>
#include <vector>

namespace picBunch::pbfast
{
    //! Device-side aliases of the shared knot constants.  The kernel uses these
    //! names and the consistency test compares them against the host aliases in
    //! pbfast::detail, so a one-sided edit of either implementation is caught.
    namespace dconst
    {
        inline constexpr double kKnotDetect = picBunch::pbfast::kKnotDetect;
        inline constexpr double kKnotEvalSnapLowGamma = picBunch::pbfast::kKnotEvalSnapLowGamma;
        inline constexpr double kKnotEvalSnapHighGamma = picBunch::pbfast::kKnotEvalSnapHighGamma;
        inline constexpr double kKnotHighGammaThreshold = picBunch::pbfast::kKnotHighGammaThreshold;
        inline constexpr double kKnotGateSnap = picBunch::pbfast::kKnotGateSnap;
    } // namespace dconst

    //! POD copy of the quadrature and geometry tables used by the device kernel.
    struct DeviceParams
    {
        static constexpr int kMaxOrder = 32;
        int order = 5;
        //! Selected order for this launch; must be one of kObliqueOrders.
        int obliqueOrder = 4;
        double xi[kMaxOrder]{};
        double wi[kMaxOrder]{};
        //! Rules for every allowed oblique order, filled once on the host.
        double oxi[kObliqueOrderCount][kMaxOrder]{};
        double owi[kObliqueOrderCount][kMaxOrder]{};
        double delta[6][3]{};
        double unitE = 0.0;
        double unitB = 0.0;
        //! Per-pair error budget reserved for the exact-path validity gate.
        double tolerance = 1e-3;
        //! Exact routing cell reach; mirrors Settings::exactCellReach.
        int exactCellReach = 0;
    };

    static_assert(
        kQuadratureOrderCap <= DeviceParams::kMaxOrder,
        "near-field command order exceeds the device xi/wi table capacity");

    //! True when a device for the compiled backend is present and the kernel can run.
    bool fastNearDeviceAvailable();

    /*! Near-shell correction for one output tile on the device.
     *
     * Returns six arrays of ``nx*ny*nz`` values (x fastest), each already
     * multiplied by ``weight*charge`` and the E/B unit.  The caller still
     * subtracts the tile-local mesh field.  ``ox/oy/oz`` is the tile's node
     * zero position, ``spacing`` the node spacing, ``cells`` the PIC cell size.
     */
    /*! @param mesh  per-source representative velocity (binned) used for the
     *               mesh-near stencil; ignored when ``stencilMesh`` is false.
     *  @param stencilMesh  subtract the 27-point TSC mesh-near term inside the
     *               kernel instead of the caller's per-tile local FFT.
     */
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
        DeviceParams const& params);
} // namespace picBunch::pbfast
