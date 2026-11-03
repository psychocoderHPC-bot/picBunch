# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
cmake_minimum_required(VERSION 3.25)

project(
    picBunch
    VERSION 0.1.0
    DESCRIPTION "Standalone GPU bunch-field initializer: openPMD particles in, free-space hybrid fields out"
    HOMEPAGE_URL "https://github.com/psychocoderHPC-bot/picBunch"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(FetchContent)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PICBUNCH_BUILD_TESTS "Build the picBunch smoke test and CTest tests." ON)
option(PICBUNCH_ENABLE_CUDA "Compile the CUDA backend through alpaka (NVIDIA GPUs)." OFF)
option(PICBUNCH_ENABLE_HIP "Compile the HIP backend through alpaka (AMD GPUs)." OFF)
option(PICBUNCH_BUILD_GTESTS "Build GoogleTest-based unit tests (adds a dependency)." OFF)

set(
    PICBUNCH_ARCH
    ""
    CACHE STRING
    "Target GPU architecture: an sm_XX value for CUDA (e.g. 70 or sm_70) or a gfx value for HIP (e.g. gfx1100). \
Used as the default for CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES; an explicit value already set by the user wins."
)

set(
    openPMD_DIR
    ""
    CACHE PATH
    "Directory containing openPMDConfig.cmake (the openPMD-api install prefix or its lib/cmake/openPMD subdirectory)."
)

if(PICBUNCH_ENABLE_CUDA AND PICBUNCH_ENABLE_HIP)
    message(FATAL_ERROR "PICBUNCH_ENABLE_CUDA and PICBUNCH_ENABLE_HIP are mutually exclusive.")
endif()

# PIConGPU-style architecture inheritance: only fill CMAKE_<LANG>_ARCHITECTURES
# when the user/cache has not provided one, so -DCMAKE_CUDA_ARCHITECTURES always
# wins over PICBUNCH_ARCH.
if(PICBUNCH_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        if(PICBUNCH_ARCH STREQUAL "")
            set(PICBUNCH_ARCH "70" CACHE STRING "" FORCE)
        endif()
        string(REGEX REPLACE "^sm_" "" _picbunch_cuda_arch "${PICBUNCH_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES "${_picbunch_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    message(STATUS "picBunch: CUDA backend, CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
elseif(PICBUNCH_ENABLE_HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
        if(PICBUNCH_ARCH STREQUAL "")
            set(PICBUNCH_ARCH "gfx1100" CACHE STRING "" FORCE)
        endif()
        set(CMAKE_HIP_ARCHITECTURES "${PICBUNCH_ARCH}" CACHE STRING "HIP architectures" FORCE)
    endif()
    message(STATUS "picBunch: HIP backend, CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "picBunch: CPU backend (no CUDA/HIP dependency enabled)")
endif()

# ---------------------------------------------------------------------------
# Pinned dependencies
#
# alpaka           7b92361468d3280c260200198081ae7e98f0bc56  (reports 3.0.0)
# alpakaVendor     af24cc7b2bd125056e9889e270d3af7bdf972c6f  (FFT: cuFFT/rocFFT/FFTW)
# ---------------------------------------------------------------------------
set(ALPAKA_PIN "7b92361468d3280c260200198081ae7e98f0bc56")
set(ALPAKAVENDOR_PIN "af24cc7b2bd125056e9889e270d3af7bdf972c6f")

set(alpaka_TESTS OFF CACHE BOOL "alpaka tests" FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "alpaka examples" FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "alpaka benchmarks" FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "alpaka header checks" FORCE)
set(alpaka_CLANG_TIDY OFF CACHE BOOL "alpaka clang-tidy" FORCE)
set(alpaka_DEP_CUDA ${PICBUNCH_ENABLE_CUDA} CACHE BOOL "alpaka CUDA backend" FORCE)
set(alpaka_DEP_HIP ${PICBUNCH_ENABLE_HIP} CACHE BOOL "alpaka HIP backend" FORCE)

FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
    GIT_TAG ${ALPAKA_PIN}
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(alpaka)

set(alpakaV_TESTS OFF CACHE BOOL "alpakaVendor tests" FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "alpakaVendor docs" FORCE)
set(alpakaV_ALPAKA_PROVIDER "extern" CACHE STRING "reuse the pinned alpaka::alpaka target" FORCE)
set(alpakaV_DEP_FFTW ON CACHE BOOL "alpakaVendor FFTW host backend" FORCE)
set(alpakaV_DEP_OPENBLAS OFF CACHE BOOL "alpakaVendor OpenBLAS host backend" FORCE)
set(alpakaV_DEP_CUFFT ${PICBUNCH_ENABLE_CUDA} CACHE BOOL "alpakaVendor cuFFT backend" FORCE)
set(alpakaV_DEP_CUBLAS OFF CACHE BOOL "alpakaVendor cuBLAS backend" FORCE)
set(alpakaV_DEP_ROCFFT ${PICBUNCH_ENABLE_HIP} CACHE BOOL "alpakaVendor rocFFT backend" FORCE)
set(alpakaV_DEP_ROCBLAS OFF CACHE BOOL "alpakaVendor rocBLAS backend" FORCE)
set(alpakaV_DEP_ONEMKL OFF CACHE BOOL "alpakaVendor oneMKL backend" FORCE)

FetchContent_Declare(
    alpakaVendor
    GIT_REPOSITORY https://github.com/psychocoderHPC-bot/alpakaVendor.git
    GIT_TAG ${ALPAKAVENDOR_PIN}
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(alpakaVendor)

# ---------------------------------------------------------------------------
# openPMD-api (system install).  The reader/writer stages need the HDF5 backend.
# ---------------------------------------------------------------------------
find_package(openPMD CONFIG QUIET)
if(NOT openPMD_FOUND)
    message(
        FATAL_ERROR
        "openPMD-api was not found. Set -DopenPMD_DIR=<prefix> or "
        "-DopenPMD_DIR=<prefix>/lib/cmake/openPMD (or add the prefix to CMAKE_PREFIX_PATH). "
        "A build without the reader/writer stages is not supported."
    )
endif()
if(NOT openPMD_HAVE_HDF5)
    message(
        FATAL_ERROR
        "openPMD-api ${openPMD_VERSION} at ${openPMD_DIR} was built WITHOUT the HDF5 backend "
        "(openPMD_HAVE_HDF5=FALSE). picBunch reads and writes openPMD/HDF5 series; "
        "point openPMD_DIR at an HDF5-enabled install instead."
    )
endif()
message(STATUS "picBunch: openPMD-api ${openPMD_VERSION} (HDF5 backend: yes)")

# ---------------------------------------------------------------------------
# Library
#
# Host-only sources stay C++20.  Device sources live under src/cuda/*.cu or
# src/hip/*.hip and are added only when the matching backend is enabled.
# alpaka_finalize() propagates the backend flags and the language switch.
# ---------------------------------------------------------------------------
set(PICBUNCH_SOURCES
    src/Deposit.cpp
    src/Fft.cpp
)

if(PICBUNCH_ENABLE_CUDA)
    file(GLOB PICBUNCH_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
    list(APPEND PICBUNCH_SOURCES ${PICBUNCH_CUDA_SOURCES})
elseif(PICBUNCH_ENABLE_HIP)
    file(GLOB PICBUNCH_HIP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*.hip")
    list(APPEND PICBUNCH_SOURCES ${PICBUNCH_HIP_SOURCES})
endif()

add_library(picBunch ${PICBUNCH_SOURCES})
add_library(picBunch::picBunch ALIAS picBunch)
target_include_directories(
    picBunch
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(picBunch PUBLIC alpaka::alpaka alpakaVendor::alpakaVendor)
if(PICBUNCH_ENABLE_CUDA)
    target_compile_definitions(picBunch PUBLIC PICBUNCH_BACKEND_CUDA=1)
elseif(PICBUNCH_ENABLE_HIP)
    target_compile_definitions(picBunch PUBLIC PICBUNCH_BACKEND_HIP=1)
else()
    target_compile_definitions(picBunch PUBLIC PICBUNCH_BACKEND_CPU=1)
endif()
alpaka_finalize(picBunch)

# Host-only openPMD layer, kept out of the device compilation unit so nvcc/hipcc
# never has to parse the HDF5 headers.
add_library(picBunchOpenPMD src/OpenPMD.cpp)
add_library(picBunch::openPMD ALIAS picBunchOpenPMD)
target_include_directories(
    picBunchOpenPMD
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(picBunchOpenPMD PUBLIC openPMD::openPMD)

# ---------------------------------------------------------------------------
# Tests
#
# GoogleTest policy: system package first (find_package(GTest)); FetchContent
# fallback is opt-in via PICBUNCH_BUILD_GTESTS=ON.  The smoke test itself has
# no GoogleTest dependency, so CTest is not empty.
# ---------------------------------------------------------------------------
if(PICBUNCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS picBunch picBunchOpenPMD EXPORT picBunchTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT picBunchTargets
    NAMESPACE picBunch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/picBunch
)
