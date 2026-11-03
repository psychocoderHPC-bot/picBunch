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
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
