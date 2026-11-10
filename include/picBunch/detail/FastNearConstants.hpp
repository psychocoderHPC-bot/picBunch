// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Shared knot-plane constants of the fast near-field evaluator.
//
// include/picBunch/detail/FastNear.hpp (host) and src/FastNearDevice.cpp
// (generic alpaka device backend) contain two independent implementations of
// the same exact s-space algorithm.  Both MUST use the constants below so a
// future threshold or snap change cannot silently land on one side only.
// The host aliases live in pbfast::detail and the device aliases in
// pbfast::dconst; tests/test_fastnear_consistency.cpp compares both.
#pragma once

namespace picBunch::pbfast
{
    //! Distance below which a source-index coordinate is treated as a knot.
    inline constexpr double kKnotDetect = 1.0e-6;
    //! One-sided offset used by the exact evaluator on a knot plane, gamma low.
    inline constexpr double kKnotEvalSnapLowGamma = 1.0e-9;
    //! One-sided offset used by the exact evaluator on a knot plane, gamma high.
    inline constexpr double kKnotEvalSnapHighGamma = 1.0e-7;
    //! Gamma at and above which the larger knot-evaluation snap is used.
    inline constexpr double kKnotHighGammaThreshold = 20.0;
    //! Snap used by the exact-path validity gate (keeps it off the knot).
    inline constexpr double kKnotGateSnap = 1.0e-5;
} // namespace picBunch::pbfast
