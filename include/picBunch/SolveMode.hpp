// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/DeviceAffinity.hpp"
#include "picBunch/Types.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace picBunch
{
    /*! Thrown when the automatic solve policy finds no runnable mode.
     *
     * It is a distinct type so the far-tail margin fallback in the command
     * line, which branches on the substring "infeasible", cannot catch and
     * reinterpret it as a padded-node-cap problem.
     */
    class SolveModeRefusal : public std::runtime_error
    {
    public:
        explicit SolveModeRefusal(std::string const& message)
            : std::runtime_error(message)
        {
        }
    };

    //! Accepted command-line spellings, in the order the help lists them.
    inline constexpr char const* kSolveModeSpellings = "auto|chunked|none|in-core";

    //! Parse a solve-mode spelling.  Returns nullopt for an unknown value.
    inline std::optional<SolveMode> parseSolveMode(std::string_view text)
    {
        if(text == "auto")
            return SolveMode::Auto;
        if(text == "chunked")
            return SolveMode::Chunked;
        if(text == "none" || text == "in-core")
            return SolveMode::InCore;
        return std::nullopt;
    }

    /*! Decode the solve-mode value carried by the collective broadcast.
     *
     * The wire carries one of the ``SolveMode`` enumerators as a double.  The
     * ``-1`` refusal sentinel is handled by the caller and is not a legal
     * mode, so it is rejected here together with every non-integral or
     * out-of-range value.  Returning ``nullopt`` lets the caller abort on a
     * corrupt or version-skewed broadcast instead of casting garbage.
     */
    inline std::optional<SolveMode> solveModeFromWire(double value)
    {
        if(value == static_cast<double>(SolveMode::Auto))
            return SolveMode::Auto;
        if(value == static_cast<double>(SolveMode::InCore))
            return SolveMode::InCore;
        if(value == static_cast<double>(SolveMode::Chunked))
            return SolveMode::Chunked;
        return std::nullopt;
    }

    //! Canonical spelling of a resolved mode (for the decision log).
    inline char const* solveModeName(SolveMode mode)
    {
        switch(mode)
        {
        case SolveMode::Auto:
            return "auto";
        case SolveMode::InCore:
            return "in-core";
        case SolveMode::Chunked:
            return "chunked";
        }
        return "unknown";
    }

    /*! Peak resident bytes of the staged chunked solve.
     *
     * Mirrors the formula in ``StagedDistributedFft.hpp``: one virtual rank's
     * ``DistributedFft3D`` workspace plus the one-velocity-bin particle
     * staging.  ``nz`` and ``nx`` are the largest contiguous split counts,
     * i.e. ``ceil(Mz/N)`` and ``ceil(Mx/N)`` as produced by ``slabRange``.
     * ``particleBytes`` is counted once because the bins are reused.
     */
    inline std::size_t estimateChunkedRequirementBytes(
        std::size_t Mx,
        std::size_t My,
        std::size_t Mz,
        std::size_t virtualRanks,
        std::size_t particleBytes)
    {
        std::size_t const ranks = std::max<std::size_t>(1u, virtualRanks);
        std::size_t const nz = (Mz + ranks - 1u) / ranks;
        std::size_t const nx = (Mx + ranks - 1u) / ranks;
        std::size_t const workspace
            = 16u
            * (Mx * My * nz + 2u * nx * My * Mz
               + 2u * std::max<std::size_t>(Mx * nz, Mz * nx));
        return workspace + particleBytes;
    }

    /*! Peak resident bytes of the staged chunked solve with its output slab.
     *
     * ``estimateChunkedRequirementBytes`` covers one virtual-rank solver
     * workspace and the particle staging.  While the solver is alive the
     * caller also retains the six-component output slab it just read out, so
     * the true peak adds ``6 * slabPlanes * planeElems`` doubles.  For a
     * single rank ``slabPlanes`` is the whole output volume.  Kept pure so the
     * gate can be unit-tested without a build- or device-dependent path.
     */
    inline std::size_t chunkedResidentBytes(
        std::size_t solverBytes,
        std::size_t slabPlanes,
        std::size_t planeElems)
    {
        constexpr std::size_t componentCount = 6u;
        constexpr std::size_t valueBytes = sizeof(double);
        return solverBytes
            + componentCount * slabPlanes * planeElems * valueBytes;
    }

    /*! Number of output z-planes owned by ``rank`` in a ``parSize``-way split.
     *
     * A single rank owns the whole output volume.  Otherwise the planes
     * partition into the half-open ranges
     * ``[rank * outNz / parSize, (rank + 1) * outNz / parSize)``.  The chunked
     * budget gate and the real output slab allocation share this helper so
     * the estimate cannot drift from what is allocated.
     */
    inline std::size_t slabPlanesForRank(
        std::size_t rank,
        std::size_t parSize,
        std::size_t outNz)
    {
        if(parSize <= 1u)
            return outNz;
        return ((rank + 1u) * outNz) / parSize - (rank * outNz) / parSize;
    }

    //! Outcome of the automatic or explicit solve-mode policy.
    struct SolveModeDecision
    {
        //! Resolved mode.  Meaningless when ``refused`` is true.
        SolveMode mode = SolveMode::InCore;
        //! True when ``Auto`` selected a mode other than its first choice.
        bool switched = false;
        //! True when no runnable mode exists; the caller must throw a refusal.
        bool refused = false;
        //! Stable key naming why the decision was taken.
        std::string reason;
        //! Bytes that the decision needs (in-core estimate, or chunked peak).
        std::size_t requiredBytes = 0;
        //! Per-rank budget that was compared against ``requiredBytes``.
        std::size_t budgetBytes = 0;
    };

    /*! Resolve the solve mode from the request and the capacity estimate.
     *
     * ``InCore`` and ``Chunked`` are hard overrides: they are returned
     * unchanged, and ``Chunked`` is not gated by the in-core fit.  ``Auto``
     * keeps the in-core path when it fits, switches to the chunked path when
     * the staged peak is available and within budget, and otherwise returns a
     * refusal.  A zero budget means the capacity is unknown; the historical
     * in-core behaviour is kept in that case.
     *
     * ``chunkedAvailable`` is the caller's statement that the staged solve can
     * run in this build and process layout.  ``mpiSize`` is accepted for the
     * caller's context and future rank-aware policies; it does not change an
     * explicit override.
     */
    inline SolveModeDecision resolveSolveMode(
        SolveMode requested,
        DeviceFit const& inCoreFit,
        std::size_t chunkedBytes,
        bool chunkedAvailable,
        int mpiSize)
    {
        (void)mpiSize;
        SolveModeDecision decision;
        decision.requiredBytes = inCoreFit.requiredBytes;
        decision.budgetBytes = inCoreFit.shareBytes;

        if(requested == SolveMode::InCore)
        {
            decision.mode = SolveMode::InCore;
            decision.reason = "explicit-in-core";
            return decision;
        }
        if(requested == SolveMode::Chunked)
        {
            // A hard override: never switch and never compare the in-core fit.
            decision.mode = SolveMode::Chunked;
            decision.reason = "explicit-chunked";
            decision.requiredBytes = chunkedBytes;
            return decision;
        }

        // Auto.  An unknown capacity keeps the historical in-core behaviour.
        if(inCoreFit.shareBytes == 0u)
        {
            decision.mode = SolveMode::InCore;
            decision.reason = "capacity-unknown";
            return decision;
        }
        if(!inCoreFit.refused)
        {
            decision.mode = SolveMode::InCore;
            decision.reason = "in-core-fits";
            return decision;
        }

        decision.requiredBytes = chunkedBytes;
        if(chunkedAvailable && chunkedBytes <= inCoreFit.shareBytes)
        {
            decision.mode = SolveMode::Chunked;
            decision.switched = true;
            decision.reason = "in-core-does-not-fit;chunked-fits";
            return decision;
        }

        decision.refused = true;
        decision.reason = chunkedAvailable ? "in-core-does-not-fit;chunked-exceeds-budget"
                                           : "in-core-does-not-fit;chunked-unavailable";
        return decision;
    }

    //! Byte count rendered the way the decision log reports it.
    inline std::string formatSolveModeBytes(std::size_t bytes)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3)
            << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB";
        return out.str();
    }

    /*! One-line decision text for stderr.
     *
     * The auto-switch form is stable and parseable:
     * ``picBunch: solve mode: auto -> chunked (required X, budget Y per rank,
     * reason key)``.  Explicit overrides print their final mode once.
     */
    inline std::string formatSolveModeDecision(SolveModeDecision const& decision)
    {
        std::ostringstream out;
        out << "picBunch: solve mode: ";
        if(decision.refused)
            out << "auto -> refused";
        else if(decision.switched)
            out << "auto -> " << solveModeName(decision.mode);
        else if(decision.reason == "in-core-fits")
            out << "auto -> in-core";
        else if(decision.reason == "capacity-unknown")
            out << "auto -> in-core";
        else
            out << solveModeName(decision.mode);

        if(decision.refused || decision.switched)
            out << " (required " << formatSolveModeBytes(decision.requiredBytes) << ", budget "
                << formatSolveModeBytes(decision.budgetBytes) << " per rank, reason "
                << decision.reason << ")";
        else if(decision.reason == "capacity-unknown")
            out << " (capacity unknown, reason capacity-unknown)";
        else if(decision.reason == "in-core-fits")
            out << " (required " << formatSolveModeBytes(decision.requiredBytes)
                << ", budget " << formatSolveModeBytes(decision.budgetBytes)
                << " per rank, reason in-core-fits)";
        else
            out << " (explicit override, reason " << decision.reason << ")";
        return out.str();
    }

    /*! Actionable refusal text for a refused decision.
     *
     * Names the requirement and budget, then the switches that lower either
     * side.  A GPU build gets the explicit host/CPU-only note for the staged
     * solve.
     */
    inline std::string solveModeRefusalMessage(SolveModeDecision const& decision)
    {
        std::ostringstream out;
        out << "solve mode: no runnable mode (reason " << decision.reason << "): ";
        if(decision.reason == "in-core-does-not-fit;chunked-unavailable")
            out << "the in-core field solve does not fit and the chunked solve is "
                   "unavailable in this build or process layout";
        else
            out << "the in-core field solve does not fit and the chunked solve exceeds "
                   "the per-rank budget";
        out << "; required " << formatSolveModeBytes(decision.requiredBytes) << " per rank, budget "
            << formatSolveModeBytes(decision.budgetBytes)
            << " per rank. Lower the requirement with --fft-padding 1 or fewer ranks, or use "
               "--fft-mode distributed";
#if defined(PICBUNCH_BACKEND_CUDA) || defined(PICBUNCH_BACKEND_HIP)
        out << ". The staged chunked solve is host/CPU-only in a CUDA/HIP build";
#endif
        out << ".";
        return out.str();
    }

    //! Throw ``SolveModeRefusal`` when the decision carries no runnable mode.
    inline void enforceSolveMode(SolveModeDecision const& decision)
    {
        if(decision.refused)
            throw SolveModeRefusal(solveModeRefusalMessage(decision));
    }
} // namespace picBunch
