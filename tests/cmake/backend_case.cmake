# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
cmake_minimum_required(VERSION 3.25)
#
# One backend-policy case per process.  The rules under test reject bad input
# with message(FATAL_ERROR), so running the function in its own `cmake -P`
# process turns "accepted" into exit status 0 and "rejected" into a non-zero
# exit status.  tests/cmake/test_backends.cmake drives the cases.

include("${PICBUNCH_CMAKE_DIR}/PicBunchBackends.cmake")

if(CASE STREQUAL "cuda-arch-numeric")
    picbunch_validate_arch(CUDA "70")
elseif(CASE STREQUAL "cuda-arch-sm-prefix")
    picbunch_validate_arch(CUDA "sm_80")
elseif(CASE STREQUAL "cuda-arch-list")
    picbunch_validate_arch(CUDA "sm_70;75")
elseif(CASE STREQUAL "cuda-arch-reject-gfx")
    picbunch_validate_arch(CUDA "gfx1100")
elseif(CASE STREQUAL "cuda-arch-reject-dot")
    picbunch_validate_arch(CUDA "8.0")
elseif(CASE STREQUAL "hip-arch-gfx")
    picbunch_validate_arch(HIP "gfx1100")
elseif(CASE STREQUAL "hip-arch-reject-numeric")
    picbunch_validate_arch(HIP "70")
elseif(CASE STREQUAL "backend-conflict")
    picbunch_validate_backends("ON" "ON")
elseif(CASE STREQUAL "backend-hip-only")
    picbunch_validate_backends("OFF" "ON")
elseif(CASE STREQUAL "fftw-default-cpu")
    picbunch_default_host_fft("OFF" "OFF" _default)
    message("RESULT=${_default}")
elseif(CASE STREQUAL "fftw-default-cuda")
    picbunch_default_host_fft("ON" "OFF" _default)
    message("RESULT=${_default}")
elseif(CASE STREQUAL "fftw-default-hip")
    picbunch_default_host_fft("OFF" "ON" _default)
    message("RESULT=${_default}")
elseif(CASE STREQUAL "fftw-explicit-off-wins")
    # The CPU default is ON; an explicit cache value must survive the default.
    set(alpakaV_DEP_FFTW OFF CACHE BOOL "" FORCE)
    picbunch_default_host_fft("OFF" "OFF" _default)
    set(alpakaV_DEP_FFTW ${_default} CACHE BOOL "")
    message("RESULT=${alpakaV_DEP_FFTW}")
elseif(CASE STREQUAL "fftw-explicit-on-wins")
    # The CUDA default is OFF; an explicit cache value must survive the default.
    set(alpakaV_DEP_FFTW ON CACHE BOOL "" FORCE)
    picbunch_default_host_fft("ON" "OFF" _default)
    set(alpakaV_DEP_FFTW ${_default} CACHE BOOL "")
    message("RESULT=${alpakaV_DEP_FFTW}")
elseif(CASE STREQUAL "fft-backends-none")
    picbunch_check_fft_backends("OFF" "OFF" "OFF")
elseif(CASE STREQUAL "fft-backends-cufft")
    picbunch_check_fft_backends("OFF" "ON" "OFF")
else()
    message(FATAL_ERROR "unknown backend test case '${CASE}'")
endif()
