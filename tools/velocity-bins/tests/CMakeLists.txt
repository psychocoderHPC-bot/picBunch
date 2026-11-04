# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera

add_executable(picBunchSmoke smoke.cpp)
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
        test_audit.cpp
        test_margin.cpp
        test_config.cpp
        test_placement.cpp
    )
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
    foreach(group Deposit Domain Kernel Fft Io Physics Padding Restart Audit Margin Config Placement)
        add_test(
            NAME picBunch.${group}
            COMMAND picBunchUnitTests --gtest_filter=${group}.*
        )
    endforeach()
endif()

# Independent openPMD/HDF5 consumer check: a small writer plus a Python reader
# that uses h5py only (no openPMD binding).  Registered when python3 + h5py are
# available; otherwise configure prints the manual command.
add_executable(picBunchWriteSample write_sample.cpp)
target_link_libraries(
    picBunchWriteSample
    PRIVATE picBunch::picBunch picBunch::openPMD
)
alpaka_finalize(picBunchWriteSample)

add_executable(picBunchWriteRestartSample write_restart_sample.cpp)
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
            "h5py not found: run tools/verify/check_openpmd.py --writer build/.../picBunchWriteSample --workdir /tmp manually"
        )
    endif()
endif()
