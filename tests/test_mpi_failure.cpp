// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Regression tests for root-only setup failure propagation (review2 P0-2).
//
// These exercise the status/length/text collective that every rank calls after
// the root reader/config/geometry block.  They run serially (the non-MPI path
// still returns a failure status) and, when the binary is started under
// ``mpirun -n 2``, verify that root's text reaches the other rank unchanged.
//
// The end-to-end CLI check cannot be expressed as an in-process GTest because
// the application itself must be launched by mpirun:
//
//   timeout 30 mpirun -n 2 <build>/picBunchApp
//       --input <missing-file.h5> --cell-size 1e-6 --output /tmp/out.h5
//
// It must terminate nonzero on both ranks without waiting for a normal exit.
// Register it as a CTest entry with WILL_FAIL TRUE and TIMEOUT 30 (see the
// accompanying report for the exact lines).

#include "picBunch/Mpi.hpp"

#include <gtest/gtest.h>

#include <string>

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
} // namespace
#endif

TEST(MpiFailure, SetupStatusReportsRootErrorOnEveryRank)
{
#if defined(PICBUNCH_ENABLE_MPI)
    ensureMpiInitialised();
#endif
    std::string message;
    if(picBunch::mpiRank() == 0)
        message = "cannot read input file 'missing.h5'";

    int const status = picBunch::mpiBroadcastRootSetupStatus(message, 0);

    EXPECT_EQ(status, 1);
    EXPECT_EQ(message, "cannot read input file 'missing.h5'");
}

TEST(MpiFailure, SetupStatusReportsSuccess)
{
#if defined(PICBUNCH_ENABLE_MPI)
    ensureMpiInitialised();
#endif
    std::string message;
    int const status = picBunch::mpiBroadcastRootSetupStatus(message, 0);

    EXPECT_EQ(status, 0);
    EXPECT_TRUE(message.empty());
}

TEST(MpiFailure, AbortIsNoOpForSingleRank)
{
#if defined(PICBUNCH_ENABLE_MPI)
    ensureMpiInitialised();
#endif
    if(picBunch::mpiSize() > 1)
        GTEST_SKIP() << "MPI_Abort would terminate the multi-rank test job";

    // Serial builds and single-rank runs must keep the caller's return status.
    EXPECT_NO_THROW(picBunch::mpiAbort(1));
}
