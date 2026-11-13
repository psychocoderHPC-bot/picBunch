# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
cmake_minimum_required(VERSION 3.25)

project(
    picBunch
    VERSION 0.9.2
    DESCRIPTION "Standalone GPU bunch-field initializer: openPMD particles in, free-space hybrid fields out"
    HOMEPAGE_URL "https://github.com/psychocoderHPC-bot/picBunch"
    LANGUAGES CXX
)

# A stale cache entry, a parent project, or an external build harness can
# otherwise compile individual sources below C++20 and turn missing language
# features into confusing template errors.  Fail at configure time instead.
# An empty value means "not set yet" and is fine; the set() below fills it in.
if(CMAKE_CXX_STANDARD AND CMAKE_CXX_STANDARD LESS 20)
    message(
        FATAL_ERROR
        "picBunch requires C++20; CMAKE_CXX_STANDARD is ${CMAKE_CXX_STANDARD}. "
        "Set -DCMAKE_CXX_STANDARD=20."
    )
endif()
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
option(PICBUNCH_BUILD_BENCH "Build the synthetic scaling benchmark (picBunchBench)." OFF)
option(PICBUNCH_ENABLE_CUDA "Compile the CUDA backend through alpaka (NVIDIA GPUs)." OFF)
option(PICBUNCH_ENABLE_HIP "Compile the HIP backend through alpaka (AMD GPUs)." OFF)
option(PICBUNCH_BUILD_GTESTS "Build the GoogleTest unit tests (system package or FetchContent)." ON)
option(
    PICBUNCH_ENABLE_MPI
    "Build MPI support: rank 0 reads and broadcasts; all ranks compute component or tile shares and write disjoint global chunks."
    OFF
)
option(
    PICBUNCH_ENABLE_WARNINGS
    "Compile picBunch's own targets with the fixed warning set (-Wall -Wextra, and the CUDA equivalent)."
    ON
)

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

set(
    FFTW3_ROOT
    "$ENV{FFTW3_ROOT}"
    CACHE PATH
    "Install prefix of an FFTW 3 build outside the default search path; adds <prefix>/lib/pkgconfig and <prefix>/lib64/pkgconfig to PKG_CONFIG_PATH."
)

set(
    PICBUNCH_SANITIZE
    "off"
    CACHE STRING
    "Instrument picBunch's own targets: 'thread' (ThreadSanitizer), 'address' (AddressSanitizer), or 'off'."
)
set_property(CACHE PICBUNCH_SANITIZE PROPERTY STRINGS off thread address)

# ---------------------------------------------------------------------------
# Backend policy
#
# All validation lives in cmake/PicBunchBackends.cmake so that the same rules
# are available to the script-mode tests.  Validate before any dependency is
# fetched: a wrong PICBUNCH_ARCH or two device backends at once is a user
# error that must not cost a network round trip.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PicBunchBackends.cmake)

picbunch_validate_backends("${PICBUNCH_ENABLE_CUDA}" "${PICBUNCH_ENABLE_HIP}")
if(PICBUNCH_ENABLE_CUDA)
    picbunch_validate_arch(CUDA "${PICBUNCH_ARCH}")
elseif(PICBUNCH_ENABLE_HIP)
    picbunch_validate_arch(HIP "${PICBUNCH_ARCH}")
elseif(NOT PICBUNCH_ARCH STREQUAL "")
    message(
        WARNING
        "picBunch: PICBUNCH_ARCH='${PICBUNCH_ARCH}' is ignored because neither PICBUNCH_ENABLE_CUDA nor "
        "PICBUNCH_ENABLE_HIP is ON."
    )
endif()

if(PICBUNCH_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    message(STATUS "picBunch: MPI enabled (MPI_CXX ${MPI_CXX_VERSION})")
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
# alpaka3          e91f6f4d8d1da321a6423a8ad5b4954562c57a9a  (alpaka-group/alpaka3)
# alpakaVendor     33a4a4f401de2b157de195bb68e8ec61d7460cee  (alpaka-group/alpakaVendor; FFT: cuFFT/rocFFT/FFTW)
# toml++           30172438cee64926dc41fdd9c11fb3ba5b2ba9de  (v3.4.0, header-only)
# ---------------------------------------------------------------------------
set(ALPAKA_PIN "e91f6f4d8d1da321a6423a8ad5b4954562c57a9a")
set(ALPAKAVENDOR_PIN "33a4a4f401de2b157de195bb68e8ec61d7460cee")
set(TOMLPLUSPLUS_PIN "30172438cee64926dc41fdd9c11fb3ba5b2ba9de")

set(alpaka_TESTS OFF CACHE BOOL "alpaka tests" FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "alpaka examples" FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "alpaka benchmarks" FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "alpaka header checks" FORCE)
set(alpaka_CLANG_TIDY OFF CACHE BOOL "alpaka clang-tidy" FORCE)
set(alpaka_DEP_CUDA ${PICBUNCH_ENABLE_CUDA} CACHE BOOL "alpaka CUDA backend" FORCE)
set(alpaka_DEP_HIP ${PICBUNCH_ENABLE_HIP} CACHE BOOL "alpaka HIP backend" FORCE)
# alpaka's HIP backend rejects OpenMP with a non-Clang CXX compiler
# (alpaka/cmake/alpakaHip.cmake: the CXX and HIP compilers must share a
# compatible OpenMP runtime).  The default CXX compiler on many hosts is GNU,
# so disable alpaka's OpenMP backend for HIP unless the user asked for it
# explicitly; with -Dalpaka_DEP_OMP=ON alpaka still reports its own error.
if(PICBUNCH_ENABLE_HIP AND NOT DEFINED alpaka_DEP_OMP)
    set(alpaka_DEP_OMP OFF CACHE BOOL "alpaka OpenMP backend (forced off for HIP; alpaka limitation)" FORCE)
endif()

FetchContent_Declare(
    alpaka3
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
    GIT_TAG ${ALPAKA_PIN}
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(alpaka3)

set(alpakaV_TESTS OFF CACHE BOOL "alpakaVendor tests" FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "alpakaVendor docs" FORCE)
# The host FFT default follows the selected device backend: the CPU backend has
# no vendor FFT, CUDA uses cuFFT and HIP uses rocFFT.  No FORCE: an explicit
# -DalpakaV_DEP_FFTW=OFF (or =ON) in the cache always wins over this default.
picbunch_default_host_fft("${PICBUNCH_ENABLE_CUDA}" "${PICBUNCH_ENABLE_HIP}" _picbunch_fftw_default)
set(
    alpakaV_DEP_FFTW
    ${_picbunch_fftw_default}
    CACHE BOOL "alpakaVendor FFTW host backend (default: ON for the CPU backend, OFF for CUDA/HIP)"
)
set(alpakaV_DEP_OPENBLAS OFF CACHE BOOL "alpakaVendor OpenBLAS host backend" FORCE)
# The device FFT backends follow PICBUNCH_ENABLE_CUDA/PICBUNCH_ENABLE_HIP and,
# like the host backend, honour an explicit user value.
set(
    alpakaV_DEP_CUFFT
    ${PICBUNCH_ENABLE_CUDA}
    CACHE BOOL "alpakaVendor cuFFT backend (default follows PICBUNCH_ENABLE_CUDA)"
)
set(alpakaV_DEP_CUBLAS OFF CACHE BOOL "alpakaVendor cuBLAS backend" FORCE)
set(
    alpakaV_DEP_ROCFFT
    ${PICBUNCH_ENABLE_HIP}
    CACHE BOOL "alpakaVendor rocFFT backend (default follows PICBUNCH_ENABLE_HIP)"
)
set(alpakaV_DEP_ROCBLAS OFF CACHE BOOL "alpakaVendor rocBLAS backend" FORCE)
set(alpakaV_DEP_ONEMKL OFF CACHE BOOL "alpakaVendor oneMKL backend" FORCE)
picbunch_check_fft_backends("${alpakaV_DEP_FFTW}" "${alpakaV_DEP_CUFFT}" "${alpakaV_DEP_ROCFFT}")
message(
    STATUS
    "picBunch: FFT backends, host FFTW=${alpakaV_DEP_FFTW} cuFFT=${alpakaV_DEP_CUFFT} rocFFT=${alpakaV_DEP_ROCFFT}"
)

# FFTW is a build requirement only when the host FFT path is used, i.e. when
# alpakaV_DEP_FFTW is ON.  alpakaVendor itself calls
# pkg_check_modules(FFTW3 REQUIRED fftw3 fftw3f); probing here first turns that
# generic failure into a message that names the package and the install command.
# fftw3 (double) additionally drives the tools/fft-size reference transform and
# stays optional.
if(FFTW3_ROOT)
    set(_picbunch_fftw_pc "${FFTW3_ROOT}/lib/pkgconfig:${FFTW3_ROOT}/lib64/pkgconfig")
    if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
        set(ENV{PKG_CONFIG_PATH} "${_picbunch_fftw_pc}:$ENV{PKG_CONFIG_PATH}")
    else()
        set(ENV{PKG_CONFIG_PATH} "${_picbunch_fftw_pc}")
    endif()
endif()
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    # Double precision FFTW only feeds the tools/fft-size reference transform.
    pkg_check_modules(PBFFTW3 QUIET fftw3)
endif()
if(alpakaV_DEP_FFTW)
    if(PkgConfig_FOUND)
        pkg_check_modules(PBFFTW3F QUIET fftw3f)
    endif()
    if(NOT PBFFTW3F_FOUND)
        message(
            FATAL_ERROR
            "picBunch: the host FFT backend is enabled (alpakaV_DEP_FFTW=ON) but FFTW 3 was not "
            "found (pkg-config module 'fftw3f'; ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt also "
            "needs pkg-config itself).\n"
            "Install it with one of:\n"
            "  Debian/Ubuntu:  sudo apt-get install libfftw3-dev\n"
            "  Spack:          spack install fftw\n"
            "For a non-standard prefix configure with -DFFTW3_ROOT=<prefix> (the directory that "
            "contains lib/pkgconfig or lib64/pkgconfig) or add it to PKG_CONFIG_PATH.\n"
            "A CUDA build does not need a host FFT: configure with -DPICBUNCH_ENABLE_CUDA=ON and "
            "alpakaV_DEP_FFTW=OFF (the default), which uses cuFFT.  HIP uses rocFFT the same way."
        )
    endif()
endif()

FetchContent_Declare(
    alpakaVendor
    GIT_REPOSITORY https://github.com/alpaka-group/alpakaVendor.git
    GIT_TAG ${ALPAKAVENDOR_PIN}
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(alpakaVendor)

# toml++ is a single header; FetchContent supplies the interface target without
# compiling a library.  The pin above is the v3.4.0 commit.
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG ${TOMLPLUSPLUS_PIN}
    GIT_SHALLOW FALSE
)
set(TOMLPLUSPLUS_BUILD_TESTS OFF CACHE BOOL "toml++ tests" FORCE)
set(TOMLPLUSPLUS_BUILD_EXAMPLES OFF CACHE BOOL "toml++ examples" FORCE)
FetchContent_MakeAvailable(tomlplusplus)

# ---------------------------------------------------------------------------
# openPMD-api (system install).  The reader/writer stages need the HDF5 backend.
# ---------------------------------------------------------------------------
# HDF5 C API for the PIConGPU checkpoint writer.  Prefer the parallel build so the
# process loads the same libhdf5 as openPMD's HDF5 backend (mixing two HDF5
# libraries would interpose their symbols).  Serial file access works without
# MPI_Init; the MPI target is needed only for the headers when HDF5 is parallel.
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
if(HDF5_IS_PARALLEL)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# openPMD's ADIOS2 MPI target (adios2::cxx11_mpi) links MPI::MPI_C.  The openPMD
# config reaches CMake's FindMPI through ADIOS2's find_dependency(MPI), and
# FindMPI only creates per-language targets for languages enabled by project().
# This project enables CXX only, so MPI::MPI_C would be missing and generation
# fails for an MPI/ADIOS2-enabled openPMD install.  Enable C before finding
# openPMD so the C MPI target exists; the C compiler is already a dependency of
# the HDF5 C API above.
enable_language(C)
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
if(NOT openPMD_HAVE_ADIOS2)
    message(
        WARNING
        "openPMD-api ${openPMD_VERSION} at ${openPMD_DIR} was built WITHOUT the ADIOS2 "
        "backend (openPMD_HAVE_ADIOS2=FALSE). HDF5 input stays fully supported, but a "
        "BP5/BP directory input (e.g. particles_000000.bp5) will be rejected with a clear "
        "error. Point openPMD_DIR at an ADIOS2-enabled install to read BP5."
    )
else()
    message(STATUS "picBunch: openPMD-api ${openPMD_VERSION} (ADIOS2 backend: yes)")
endif()

# ---------------------------------------------------------------------------
# Sanitizers
#
# Applied after the pinned dependencies have been added so that the fetched
# alpaka/alpakaVendor targets stay uninstrumented; only picBunch's own library,
# app and tests are compiled with the sanitizer.  Use the CPU backend for the
# thread sanitizer (device code is not instrumented by the host compiler).
# ---------------------------------------------------------------------------
if(NOT PICBUNCH_SANITIZE STREQUAL "off")
    if(NOT PICBUNCH_SANITIZE MATCHES "^(thread|address)$")
        message(FATAL_ERROR "PICBUNCH_SANITIZE must be one of: off, thread, address")
    endif()
    message(STATUS "picBunch: ${PICBUNCH_SANITIZE} sanitizer enabled for picBunch targets")
    add_compile_options(-fsanitize=${PICBUNCH_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${PICBUNCH_SANITIZE})
endif()

# ---------------------------------------------------------------------------
# Warnings
#
# A fixed set for picBunch's own targets, applied per target so no fetched
# dependency (alpaka, alpakaVendor, toml++, GoogleTest) inherits it.  The
# dependency headers are additionally marked SYSTEM, so diagnostics from them
# stay out of picBunch's translation units.  -Werror is deliberately not part of
# the set; a compiler update must not break the build.
# nvcc does not accept -Wall itself, so the CUDA case forwards the flags to the
# host compiler behind -Xcompiler.
# ---------------------------------------------------------------------------
function(picbunch_target_warnings target)
    if(NOT PICBUNCH_ENABLE_WARNINGS)
        return()
    endif()
    target_compile_options(
        ${target}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
            $<$<COMPILE_LANGUAGE:HIP>:-Wall> $<$<COMPILE_LANGUAGE:HIP>:-Wextra>
    )
endfunction()

if(PICBUNCH_ENABLE_WARNINGS)
    foreach(_picbunch_dependency alpaka alpakaVendor tomlplusplus)
        if(TARGET ${_picbunch_dependency})
            set_property(TARGET ${_picbunch_dependency} PROPERTY SYSTEM TRUE)
        endif()
    endforeach()
    message(STATUS "picBunch: warnings enabled for picBunch targets only (-Wall -Wextra)")
endif()

# ---------------------------------------------------------------------------
# Library
#
# Host sources stay C++20.  Device code is written as ALPAKA_FN_ACC functors
# and alpaka/C++ in these .cpp sources and compiled by the backend compiler
# (host, nvcc or hipcc) through alpaka; no .cu/.hip translation unit exists.
# src/FastNearDevice.cpp is part of every backend target, so the near-shell
# kernel is built from the same C++ source for CPU, CUDA and HIP.
# alpaka_finalize() propagates the backend flags and the language switch.
# ---------------------------------------------------------------------------
set(PICBUNCH_SOURCES
    src/Deposit.cpp
    src/Compute.cpp
    src/FarTail.cpp
    src/FastNearDevice.cpp
    src/NearShell.cpp
    src/VelocityBinning.cpp
    src/VelocityMoments.cpp
)

add_library(picBunch ${PICBUNCH_SOURCES})
picbunch_target_warnings(picBunch)
add_library(picBunch::picBunch ALIAS picBunch)
# Require the language level on the target itself, not only through the
# directory-level CMAKE_CXX_STANDARD, so consumers and external harnesses
# cannot compile picBunch sources below C++20.
target_compile_features(picBunch PUBLIC cxx_std_20)
target_include_directories(
    picBunch
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(picBunch PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(picBunch PUBLIC alpaka::alpaka alpakaVendor::alpakaVendor tomlplusplus::tomlplusplus)
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
add_library(
    picBunchOpenPMD
    src/Config.cpp
    src/OpenPMD.cpp
    src/OpenPMDWrite.cpp
    src/OpenPMDWriteOpenPMD.cpp
    src/Mpi.cpp
    src/Placement.cpp
)
add_library(picBunch::openPMD ALIAS picBunchOpenPMD)
picbunch_target_warnings(picBunchOpenPMD)
target_compile_features(picBunchOpenPMD PUBLIC cxx_std_20)
# This host-only layer never initializes an alpaka backend.  It does include
# alpaka::Vec through the public headers, so satisfy alpaka's finalize guard
# instead of finalizing the target (which would switch the openPMD/HDF5
# translation units to CUDA/HIP under those backends).
target_compile_definitions(
    picBunchOpenPMD
    PRIVATE
        ALPAKA_CMAKE_TARGET_ALPAKA_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HIP_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_ONEAPI_FINALIZE_CALLED
)
# The chunked writer needs the multipole far tail from the host library.
target_link_libraries(picBunchOpenPMD PUBLIC picBunch)
target_include_directories(
    picBunchOpenPMD
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(picBunchOpenPMD PUBLIC openPMD::openPMD HDF5::HDF5)
if(HDF5_IS_PARALLEL)
    target_compile_definitions(picBunchOpenPMD PUBLIC OMPI_SKIP_MPICXX=1)
    target_link_libraries(picBunchOpenPMD PUBLIC MPI::MPI_CXX)
endif()
if(PICBUNCH_ENABLE_MPI)
    target_compile_definitions(picBunchOpenPMD PUBLIC PICBUNCH_ENABLE_MPI=1)
    target_link_libraries(picBunchOpenPMD PUBLIC MPI::MPI_CXX)
endif()

# ---------------------------------------------------------------------------
# Command line tool
# ---------------------------------------------------------------------------
add_executable(picBunchApp src/main.cpp)
picbunch_target_warnings(picBunchApp)
target_compile_features(picBunchApp PUBLIC cxx_std_20)
set_target_properties(picBunchApp PROPERTIES OUTPUT_NAME picBunch)
target_link_libraries(picBunchApp PRIVATE picBunch::picBunch picBunch::openPMD)
if(PICBUNCH_ENABLE_MPI)
    target_link_libraries(picBunchApp PRIVATE MPI::MPI_CXX)
endif()
alpaka_finalize(picBunchApp)

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

# Installation stays limited to the libraries and headers; the pinned dependency
# targets are fetched into the build tree and are not exported in this skeleton.
install(TARGETS picBunch picBunchOpenPMD picBunchApp)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Measurement driver for the exact near shell (host-only, no openPMD).
add_executable(picBunchDfftCheck tools/dfft-check/check.cpp)
picbunch_target_warnings(picBunchDfftCheck)
target_compile_features(picBunchDfftCheck PUBLIC cxx_std_20)
target_link_libraries(picBunchDfftCheck PRIVATE picBunch::picBunch picBunch::openPMD)
alpaka_finalize(picBunchDfftCheck)

# FFT-extent characterisation driver: replicated pipeline vs an independent
# FFTW reference, per padded extent (tools/fft-size).
#
# The driver compares picBunch's own FFT against a plain double-precision FFTW
# transform, so it needs the fftw3 module (not fftw3f).  That is a development
# tool, not part of the library: when the module is absent the driver is
# skipped instead of failing the configure.  A host FFT build already required
# fftw3f above; Debian's libfftw3-dev ships both.
if(PBFFTW3_FOUND)
    add_library(PBFFTW3_pc INTERFACE)
    target_include_directories(PBFFTW3_pc INTERFACE ${PBFFTW3_INCLUDE_DIRS})
    if(PBFFTW3_LIBRARY_DIRS)
        target_link_directories(PBFFTW3_pc INTERFACE ${PBFFTW3_LIBRARY_DIRS})
    endif()
    target_link_libraries(PBFFTW3_pc INTERFACE ${PBFFTW3_LIBRARIES})
    add_library(PBFFTW3::PBFFTW3 ALIAS PBFFTW3_pc)

    # ABI probe: classic FFTW 3.3 plan constructors take a `sign` argument; the
    # newer FFTW-compatible builds (e.g. the MLIR-derived 3.6 on hal) omit it and
    # transform forward.  The driver compiles both forms under one macro.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${PBFFTW3_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LINK_DIRECTORIES ${PBFFTW3_LIBRARY_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${PBFFTW3_LIBRARIES})
    check_cxx_source_compiles(
        "
        #include <fftw3.h>
        int main()
        {
            int const n[3] = {4, 4, 4};
            fftw_plan const p = fftw_plan_many_dft_r2c(
                3, n, 1, nullptr, n, 1, 1, nullptr, n, 1, 1, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_destroy_plan(p);
            return 0;
        }"
        PBFFTW3_PLAN_TAKES_SIGN
    )
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LINK_DIRECTORIES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    message(STATUS "picBunch: FFTW plan constructor takes sign argument: ${PBFFTW3_PLAN_TAKES_SIGN}")
    add_executable(picBunchFftSizeCheck tools/fft-size/check.cpp)
    picbunch_target_warnings(picBunchFftSizeCheck)
    target_compile_features(picBunchFftSizeCheck PUBLIC cxx_std_20)
    if(PBFFTW3_PLAN_TAKES_SIGN)
        target_compile_definitions(picBunchFftSizeCheck PRIVATE PBFFTW3_PLAN_TAKES_SIGN=1)
    endif()
    target_link_libraries(picBunchFftSizeCheck PRIVATE picBunch::picBunch PBFFTW3::PBFFTW3)
    alpaka_finalize(picBunchFftSizeCheck)
else()
    message(STATUS "picBunch: picBunchFftSizeCheck skipped (pkg-config module fftw3 not found)")
endif()

add_executable(picBunchNearShellMeasure tools/near-shell/measure_near_shell.cpp)
picbunch_target_warnings(picBunchNearShellMeasure)
target_compile_features(picBunchNearShellMeasure PUBLIC cxx_std_20)
target_link_libraries(picBunchNearShellMeasure PRIVATE picBunch::picBunch)
alpaka_finalize(picBunchNearShellMeasure)

# Pointwise fast-near evaluator for the tab:fast 52-point validation set
# (tools/verify/pb_fast_near_52pt.cpp).  CPU-only and header-only: it calls
# picBunch::pbfast::shape_field directly, so it links no library symbols and
# needs no CUDA or FFTW.  The near-field headers include alpaka's macro header
# unconditionally, so the tool needs the header-only alpaka target.
add_executable(picBunchFastNear52pt tools/verify/pb_fast_near_52pt.cpp)
picbunch_target_warnings(picBunchFastNear52pt)
target_compile_features(picBunchFastNear52pt PUBLIC cxx_std_20)
target_include_directories(picBunchFastNear52pt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(picBunchFastNear52pt PRIVATE alpaka::alpaka)
alpaka_finalize(picBunchFastNear52pt)

# Standalone direct d-space ray-quadrature reference for the gamma-100 oblique
# in-support rows in tests/data/near_distance_golden.csv.  CPU only and
# dependency-free: it reimplements the defining integral from first principles,
# so it never links the evaluator under test.  Values are precomputed; the
# accuracy test only reads the CSV.
add_executable(picBunchNearShellRayReference tools/near-shell/ray_reference.cpp)
picbunch_target_warnings(picBunchNearShellRayReference)
target_compile_features(picBunchNearShellRayReference PUBLIC cxx_std_20)

# Measurement driver for velocity binning: synthetic mixed-velocity bunch,
# binned fields for a list of box counts, for the continuum reference
# comparison (tools/velocity-bins/compare_velocity_bins.py).
add_executable(picBunchVelocityBinsMeasure tools/velocity-bins/measure_velocity_bins.cpp)
picbunch_target_warnings(picBunchVelocityBinsMeasure)
target_compile_features(picBunchVelocityBinsMeasure PUBLIC cxx_std_20)
target_link_libraries(picBunchVelocityBinsMeasure PRIVATE picBunch::picBunch)
alpaka_finalize(picBunchVelocityBinsMeasure)

# ---------------------------------------------------------------------------
# Synthetic scaling benchmark (additive, opt-in)
#
# Times picBunch::computeField() on in-memory bunches; no openPMD input, no
# change to the library, the CLI, or any public API.
# ---------------------------------------------------------------------------
if(PICBUNCH_BUILD_BENCH)
    find_package(Threads REQUIRED)
    add_executable(picBunchBench tools/bench/bench.cpp)
    picbunch_target_warnings(picBunchBench)
    target_compile_features(picBunchBench PUBLIC cxx_std_20)
    target_link_libraries(picBunchBench PRIVATE picBunch::picBunch Threads::Threads)
    if(PICBUNCH_ENABLE_MPI)
        # The rank-parallel context lives in the host-only library; the MPI
        # benchmark path runs the same computeField call on every rank.
        target_link_libraries(picBunchBench PRIVATE picBunch::openPMD MPI::MPI_CXX)
    endif()
    if(PICBUNCH_ENABLE_CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            target_link_libraries(picBunchBench PRIVATE CUDA::cudart)
            target_compile_definitions(picBunchBench PRIVATE PICBUNCH_BENCH_HAVE_CUDA=1)
        endif()
    endif()
    alpaka_finalize(picBunchBench)
    install(TARGETS picBunchBench)
endif()

# ---------------------------------------------------------------------------
# Cache visibility
#
# Every alpaka_* and alpakaV_* cache variable belongs to a pinned dependency,
# including the entries the dependencies add themselves.  They are
# implementation details, not part of picBunch's interface: hide them from
# cmake-gui/ccmake so the documented PICBUNCH_* switches are the only visible
# knobs.  This runs last so the entries created by FetchContent are covered too.
# ---------------------------------------------------------------------------
get_cmake_property(_picbunch_cache_variables CACHE_VARIABLES)
foreach(_picbunch_cache_variable IN LISTS _picbunch_cache_variables)
    if(_picbunch_cache_variable MATCHES "^(alpaka|alpakaV)_")
        mark_as_advanced(${_picbunch_cache_variable})
    endif()
endforeach()
