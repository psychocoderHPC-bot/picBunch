// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression tests for the MPI compute-options transfer (review2 P0-1).
//
// The fingerprint test runs in every build.  The broadcast test only has a
// meaningful multi-rank expectation when the binary is started under
// ``mpirun -n 2``; a single-rank CTest invocation skips it.  It never calls
// ``MPI_Finalize`` from the test itself; a process-exit guard does it once.

#include "picBunch/Mpi.hpp"
#include "picBunch/Types.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#if defined(PICBUNCH_ENABLE_MPI)
#    include <mpi.h>

namespace
{
    void ensureMpiInitialised()
    {
        int initialised = 0;
        MPI_Initialized(&initialised);
        if(!initialised)
            MPI_Init(nullptr, nullptr);
    }

    // gtest_main returns from main without an MPI_Finalize call.  OpenMPI then
    // reports an improper exit even when every test passed.  Finalise once at
    // process exit; the guard is idempotent if another test already did.
    struct MpiFinalizer
    {
        ~MpiFinalizer()
        {
            int initialised = 0;
            int finalised = 0;
            MPI_Initialized(&initialised);
            MPI_Finalized(&finalised);
            if(initialised && !finalised)
                MPI_Finalize();
        }
    };
    MpiFinalizer g_mpiFinalizer;
} // namespace
#endif

TEST(MpiOptions, FingerprintIsStable)
{
    picBunch::ComputeOptions options;
    auto const base = picBunch::computeOptionsFingerprint(options);
    EXPECT_EQ(base, picBunch::computeOptionsFingerprint(options));
}

TEST(MpiOptions, FingerprintCoversNearShellFastOrder)
{
    picBunch::ComputeOptions options;
    auto const base = picBunch::computeOptionsFingerprint(options);

    options.nearShell.fastOrder = 8;
    auto const fast = picBunch::computeOptionsFingerprint(options);
    EXPECT_NE(fast, base);

    options.nearShell.fastObliqueOrder = 6;
    auto const oblique = picBunch::computeOptionsFingerprint(options);
    EXPECT_NE(oblique, fast);

    // Restoring the values restores the digest: the hash has no rank or
    // process state.
    options.nearShell.fastOrder = 5;
    options.nearShell.fastObliqueOrder = 0;
    EXPECT_EQ(picBunch::computeOptionsFingerprint(options), base);
}

TEST(MpiOptions, FingerprintCoversEveryVelocityBinningField)
{
    picBunch::ComputeOptions options;
    auto const base = picBunch::computeOptionsFingerprint(options);

    picBunch::ComputeOptions changed = options;
    changed.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
    EXPECT_NE(picBunch::computeOptionsFingerprint(changed), base);

    changed = options;
    changed.velocityBinning.explicitBins = 16;
    EXPECT_NE(picBunch::computeOptionsFingerprint(changed), base);

    changed = options;
    changed.velocityBinning.modelConstant = 3.0;
    EXPECT_NE(picBunch::computeOptionsFingerprint(changed), base);

    changed = options;
    changed.velocityBinning.maxBins = 64;
    EXPECT_NE(picBunch::computeOptionsFingerprint(changed), base);
}

#if defined(PICBUNCH_ENABLE_MPI)
TEST(MpiOptions, BroadcastDeliversRootPhysicsToEveryRank)
{
    ensureMpiInitialised();
    if(picBunch::mpiSize() < 2)
        GTEST_SKIP() << "run under 'mpirun -n 2' for the multi-rank check";

    picBunch::ComputeOptions options;
    if(picBunch::mpiRank() == 0)
    {
        options.tolerance = 1.0e-5;
        options.nearShell.mode = picBunch::NearShellOptions::Mode::Fast;
        options.nearShell.radiusCells = 3.5;
        options.nearShell.tileNodes = 8;
        options.nearShell.fastOrder = 8;
        options.nearShell.fastObliqueOrder = 6;
        options.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = 16;
        options.velocityBinning.modelConstant = 3.0;
        options.velocityBinning.maxBins = 64;
    }

    picBunch::mpiBroadcastComputeOptions(options, 0);

    // The broadcast verifies the digest internally and throws on a mismatch;
    // these assertions pin the individual fields the review found diverging.
    EXPECT_EQ(options.nearShell.fastOrder, 8);
    EXPECT_EQ(options.nearShell.fastObliqueOrder, 6);
    EXPECT_EQ(options.nearShell.tileNodes, 8u);
    EXPECT_EQ(options.nearShell.mode, picBunch::NearShellOptions::Mode::Fast);
    EXPECT_EQ(
        options.velocityBinning.mode,
        picBunch::VelocityBinningOptions::Mode::Explicit);
    EXPECT_EQ(options.velocityBinning.explicitBins, 16u);
    EXPECT_DOUBLE_EQ(options.velocityBinning.modelConstant, 3.0);
    EXPECT_EQ(options.velocityBinning.maxBins, 64u);

    // Independently confirm that every rank holds the same digest.
    std::uint64_t local = picBunch::computeOptionsFingerprint(options);
    std::uint64_t minimum = 0;
    std::uint64_t maximum = 0;
    MPI_Allreduce(&local, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum, maximum);
}
#endif
