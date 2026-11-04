// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Device backend of the fast near-field evaluator (FastNear.hpp).
//
// The per-(source, node) algebra is identical to the host evaluator: the
// eight-term asinh singular box on the TSC support box, the plain
// Gauss-Legendre remainder on the subtracted integrand (S - W*)K over the
// knot/s* panels, and the rest-frame octant quadrature for an oblique
// velocity.  One thread owns one output node and a strided chunk of the near
// sources, accumulates the six components, and atomically adds its partial
// sum.  Only alpaka/CUDA builds compile the kernel; other backends keep the
// host evaluator.
#pragma once

#include "picBunch/Types.hpp"

#include <cstddef>
#include <vector>

#if defined(PICBUNCH_BACKEND_CUDA)
#    include <alpaka/alpaka.hpp>
#endif

namespace picBunch::pbfast
{
    //! POD copy of the quadrature and geometry tables used by the device kernel.
    struct DeviceParams
    {
        static constexpr int kMaxOrder = 32;
        int order = 5;
        int obliqueOrder = 8;
        double xi[kMaxOrder]{};
        double wi[kMaxOrder]{};
        double oxi[kMaxOrder]{};
        double owi[kMaxOrder]{};
        double delta[6][3]{};
        double unitE = 0.0;
        double unitB = 0.0;
    };

#if defined(PICBUNCH_BACKEND_CUDA)
    //! True when a CUDA device is present and the kernel can run.
    bool fastNearDeviceAvailable();

    /*! Near-shell correction for one output tile on the device.
     *
     * Returns six arrays of ``nx*ny*nz`` values (x fastest), each already
     * multiplied by ``weight*charge`` and the E/B unit.  The caller still
     * subtracts the tile-local mesh field.  ``ox/oy/oz`` is the tile's node
     * zero position, ``spacing`` the node spacing, ``cells`` the PIC cell size.
     */
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
        DeviceParams const& params);
#endif
} // namespace picBunch::pbfast
