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
        test_kernel.cpp
        test_fft.cpp
        test_io.cpp
        test_physics.cpp
        test_padding.cpp
    )
    target_include_directories(picBunchUnitTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(
        picBunchUnitTests
        PRIVATE picBunch::picBunch picBunch::openPMD GTest::gtest_main GTest::gtest
    )
    alpaka_finalize(picBunchUnitTests)

    # One CTest entry per required test group.
    foreach(group Deposit Kernel Fft Io Physics Padding)
        add_test(
            NAME picBunch.${group}
            COMMAND picBunchUnitTests --gtest_filter=${group}.*
        )
    endforeach()
endif()
