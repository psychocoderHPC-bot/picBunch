# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera

add_executable(picBunchSmoke smoke.cpp)
picbunch_target_warnings(picBunchSmoke)
target_compile_features(picBunchSmoke PUBLIC cxx_std_20)
target_link_libraries(picBunchSmoke PRIVATE picBunch::picBunch picBunch::openPMD)
alpaka_finalize(picBunchSmoke)

add_test(NAME picBunch.smoke COMMAND picBunchSmoke)

if(PICBUNCH_BUILD_GTESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(
        picBunchUnitTests
        test_deposit.cpp
        test_domain.cpp
        test_kernel.cpp
        test_fft.cpp
        test_io.cpp
        test_physics.cpp
        test_padding.cpp
        test_restart.cpp
        test_restart_pml.cpp
        test_audit.cpp
        test_margin.cpp
        test_config.cpp
        test_placement.cpp
        test_velocity_bins.cpp
        test_velocity_moments.cpp
        test_alpaka_reductions.cpp
        test_mpi_options.cpp
        test_mpi_failure.cpp
        test_near_shell_accuracy.cpp
        test_near_shell_guard.cpp
        test_zero_field_guard.cpp
        test_near_shell_alpaka.cpp
        test_fastnear_poly.cpp
        test_fastnear_consistency.cpp
        test_readout.cpp
        test_mpi_readout.cpp
        test_alpaka_readout.cpp
        test_openpmd_wait.cpp
        test_checkpoint_field_units.cpp
        test_multirank_field_equality.cpp
        test_multirank_patches.cpp
    )
    picbunch_target_warnings(picBunchUnitTests)
    target_compile_features(picBunchUnitTests PUBLIC cxx_std_20)
    target_include_directories(picBunchUnitTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(
        picBunchUnitTests
        PRIVATE
            PICBUNCH_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
            PICBUNCH_APP_PATH="$<TARGET_FILE:picBunchApp>"
    )
    target_link_libraries(
        picBunchUnitTests
        PRIVATE picBunch::picBunch picBunch::openPMD GTest::gtest_main GTest::gtest
    )
    add_dependencies(picBunchUnitTests picBunchApp)
    alpaka_finalize(picBunchUnitTests)

    # One CTest entry per required test group.
    foreach(group Deposit Domain Kernel Fft Io Physics Padding Restart Audit Margin Config Placement VelocityBins VelocityMoments AlpakaReductions MpiOptions MpiFailure NearShellAccuracy NearShellGuard Readout NearShellAlpaka AlpakaReadout OpenPmdWait FastNearConsistency CheckpointFieldUnits MultiRankField)
        add_test(
            NAME picBunch.${group}
            COMMAND picBunchUnitTests --gtest_filter=${group}.*
        )
    endforeach()

    if(PICBUNCH_ENABLE_MPI)
        if(NOT MPIEXEC_EXECUTABLE)
            find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
        endif()
        if(MPIEXEC_EXECUTABLE)
            add_test(
                NAME picBunch.MpiOptionsBroadcast
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MpiOptions.BroadcastDeliversRootPhysicsToEveryRank
            )
            add_test(
                NAME picBunch.MpiFailureStatus
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MpiFailure.SetupStatusReportsRootErrorOnEveryRank
            )
            # The configured path must see the rank context before makeGrid so
            # the per-rank padded-node cap scales with the rank count.
            add_test(
                NAME picBunch.ConfigMultiRankCap
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=Config.MultiRankTomlCapRealContext
            )
            add_test(
                NAME picBunch.MpiInvalidInput
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchApp>
                    --input ${CMAKE_CURRENT_BINARY_DIR}/missing-input.h5
                    --cell-size 1e-6
            )
            set_tests_properties(picBunch.MpiInvalidInput PROPERTIES WILL_FAIL TRUE TIMEOUT 30)
            # The read-out gather/scale must stay rank-local.  Run the same
            # group under one and two ranks so both the single-rank and the
            # multi-rank launch path are exercised.
            add_test(
                NAME picBunch.MpiReadoutN1
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MpiReadout.*
            )
            add_test(
                NAME picBunch.MpiReadoutN2
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MpiReadout.*
            )
            # The distributed transform must deliver the same field as the
            # replicated one on every rank, for a rank count that does not
            # divide the grid.  A normalisation defect that scales with the
            # rank count shows up here immediately.
            add_test(
                NAME picBunch.MultiRankFieldN1
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MultiRankField.*
            )
            add_test(
                NAME picBunch.MultiRankFieldN2
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MultiRankField.*
            )
            add_test(
                NAME picBunch.MultiRankFieldN3
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 3 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MultiRankField.*
            )
            # The checkpoint's particlePatches table must be complete and
            # disjoint for any rank count.  The test writes collectively and
            # checks the records, components, counts and offset/extent tiling.
            add_test(
                NAME picBunch.MultiRankPatchesN2
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MultiRankPatches.*
            )
            add_test(
                NAME picBunch.MultiRankPatchesN4
                COMMAND
                    ${MPIEXEC_EXECUTABLE} -n 4 $<TARGET_FILE:picBunchUnitTests>
                    --gtest_filter=MultiRankPatches.*
            )
        endif()
    else()
        add_test(
            NAME picBunch.MpiReadout
            COMMAND picBunchUnitTests --gtest_filter=MpiReadout.*
        )
    endif()
endif()

# Independent openPMD/HDF5 consumer check: a small writer plus a Python reader
# that uses h5py only (no openPMD binding).  Registered when python3 + h5py are
# available; otherwise configure prints the manual command.
add_executable(picBunchWriteSample write_sample.cpp)
picbunch_target_warnings(picBunchWriteSample)
target_compile_features(picBunchWriteSample PUBLIC cxx_std_20)
target_link_libraries(
    picBunchWriteSample
    PRIVATE picBunch::picBunch picBunch::openPMD
)
alpaka_finalize(picBunchWriteSample)

add_executable(picBunchWriteRestartSample write_restart_sample.cpp)
picbunch_target_warnings(picBunchWriteRestartSample)
target_compile_features(picBunchWriteRestartSample PUBLIC cxx_std_20)
target_link_libraries(
    picBunchWriteRestartSample
    PRIVATE picBunch::picBunch picBunch::openPMD
)
alpaka_finalize(picBunchWriteRestartSample)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import h5py"
        RESULT_VARIABLE PICBUNCH_H5PY_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(PICBUNCH_H5PY_RESULT EQUAL 0)
        add_test(
            NAME picBunch.OpenPMDVerify
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/verify/check_openpmd.py
                --writer $<TARGET_FILE:picBunchWriteSample>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME picBunch.PicOnGPUVerify
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/verify/check_picongpu_restart.py
                --writer $<TARGET_FILE:picBunchWriteRestartSample>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}
        )
        # End-to-end species list: two species summed, a checkpoint-only empty
        # species, and disjoint generated ids through the real config path.
        add_test(
            NAME picBunch.SpeciesCli
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/species_cli.py
                --app $<TARGET_FILE:picBunchApp>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/species-cli
        )
        # End-to-end exercise of the shipped binary: --near-shell exact must be
        # accepted by the parser and must change the field inside the shell.
        add_test(
            NAME picBunch.NearShellCli
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/near_shell_cli.py
                --app $<TARGET_FILE:picBunchApp>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}
        )
        # h5py builds the openPMD 2.0 scalar-record layout; the C++ test then
        # reads it through picBunch::readBunch.
        set(PICBUNCH_SCALAR_RECORD_FILE ${CMAKE_CURRENT_BINARY_DIR}/scalar_record_sample.h5)
        add_test(
            NAME picBunch.ScalarRecordSample
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/make_scalar_record_sample.py
                ${PICBUNCH_SCALAR_RECORD_FILE}
        )
        set_tests_properties(picBunch.ScalarRecordSample PROPERTIES FIXTURES_SETUP ScalarRecordSample)
        add_test(
            NAME picBunch.IoScalarRecord
            COMMAND picBunchUnitTests --gtest_filter=Io.ReaderAcceptsOpenPMD2ScalarRecords
        )
        set_tests_properties(
            picBunch.IoScalarRecord
            PROPERTIES
                FIXTURES_REQUIRED ScalarRecordSample
                ENVIRONMENT "PICBUNCH_SCALAR_RECORD_FILE=${PICBUNCH_SCALAR_RECORD_FILE}"
        )
    else()
        message(
            STATUS
            "h5py not found: run tools/verify/check_openpmd.py --writer build/.../picBunchWriteSample --workdir ${CMAKE_CURRENT_BINARY_DIR}/openpmd-verify manually"
        )
    endif()
endif()

# Harden the suite against a wedged HIP call.  On the shared gfx1100 host a
# test can block forever inside libhsa (stream/queue creation).  CTest then
# waits indefinitely, the operator interrupts it, and the surviving process
# keeps the single device busy: later HIP runs hang at stream creation or abort
# in the runtime with allocator corruption.  A bounded timeout stops the first
# hang, and the resource lock keeps two device tests from running concurrently
# under `ctest -j`, which is the other way to leave the device in a bad state.
get_property(picBunchTests DIRECTORY PROPERTY TESTS)
if(picBunchTests)
    set_tests_properties(
        ${picBunchTests}
        PROPERTIES
            TIMEOUT 900
            RESOURCE_LOCK picBunchGpuDevice
    )
endif()

# ---------------------------------------------------------------------------
# Configuration contract
#
# The backend policy runs in script mode and needs only cmake.  The configure
# cases reuse the dependency sources this configure already fetched
# (FETCHCONTENT_SOURCE_DIR_* plus FETCHCONTENT_FULLY_DISCONNECTED), so they
# run offline and compile nothing.  They are registered after the device-lock
# block above on purpose: these tests touch no GPU.
# ---------------------------------------------------------------------------
add_test(
    NAME picBunch.BackendContract
    COMMAND
        ${CMAKE_COMMAND} -DPICBUNCH_CMAKE_DIR=${PROJECT_SOURCE_DIR}/cmake -P
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_backends.cmake
)
set_tests_properties(picBunch.BackendContract PROPERTIES TIMEOUT 120 RESOURCE_LOCK picBunchConfigureContract)

set(_picbunch_configure_cases documented-only cuda-conflict cuda-bad-arch hip-bad-arch)
find_program(PICBUNCH_TEST_NVCC NAMES nvcc)
if(PICBUNCH_ENABLE_CUDA OR PICBUNCH_TEST_NVCC)
    list(APPEND _picbunch_configure_cases cuda-fftw-off)
endif()
foreach(_picbunch_config_case IN LISTS _picbunch_configure_cases)
    add_test(
        NAME picBunch.Configure.${_picbunch_config_case}
        COMMAND
            ${CMAKE_COMMAND} -DCASE=${_picbunch_config_case} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DWORK_DIR=${CMAKE_BINARY_DIR}/configure-contract -DOPENPMD_DIR=${openPMD_DIR}
            -DGENERATOR=${CMAKE_GENERATOR}
            -DPB_ALPAKA3_SRC=${alpaka3_SOURCE_DIR} -DPB_ALPAKAVENDOR_SRC=${alpakaVendor_SOURCE_DIR}
            -DPB_TOMLPLUSPLUS_SRC=${tomlplusplus_SOURCE_DIR} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/configure_case.cmake
    )
    set_tests_properties(
        picBunch.Configure.${_picbunch_config_case}
        PROPERTIES TIMEOUT 900 RESOURCE_LOCK picBunchConfigureContract
    )
endforeach()
