# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
# Backend selection policy.
#
# These functions hold the rules that translate the documented PICBUNCH_*
# switches into alpaka/alpakaVendor cache variables.  They deliberately touch
# no project state, so tests/cmake/test_backends.cmake can include this file in
# script mode (`cmake -P`) and exercise every rule without a full configure.

# Reject mutually exclusive backends.
#
# cuda, hip: ON/OFF values of PICBUNCH_ENABLE_CUDA/PICBUNCH_ENABLE_HIP.
function(picbunch_validate_backends cuda hip)
    if(cuda AND hip)
        message(
            FATAL_ERROR
            "picBunch: PICBUNCH_ENABLE_CUDA=ON and PICBUNCH_ENABLE_HIP=ON are mutually exclusive. "
            "Select exactly one GPU backend, or neither for the host (CPU/OpenMP) backend."
        )
    endif()
endfunction()

# Validate PICBUNCH_ARCH against the selected backend.
#
# backend: "CUDA" or "HIP".
# arch:    may be empty (the backend default applies), a single value or a
#          semicolon-separated list.  CUDA accepts numeric compute capabilities
#          with an optional sm_ prefix (70, sm_80, 90a); HIP accepts gfx targets
#          (gfx1100, gfx90a).
function(picbunch_validate_arch backend arch)
    if(arch STREQUAL "")
        return()
    endif()
    if(backend STREQUAL "CUDA")
        foreach(entry IN LISTS arch)
            string(REGEX REPLACE "^sm_" "" numeric "${entry}")
            if(NOT numeric MATCHES "^[0-9]+a?$")
                message(
                    FATAL_ERROR
                    "picBunch: PICBUNCH_ARCH='${entry}' is not a CUDA architecture. "
                    "Use a numeric compute capability such as 70, 80 or sm_80; several values may be "
                    "separated by ';' (for example '70;75')."
                )
            endif()
        endforeach()
    elseif(backend STREQUAL "HIP")
        foreach(entry IN LISTS arch)
            if(NOT entry MATCHES "^gfx[0-9a-fA-F]+$")
                message(
                    FATAL_ERROR
                    "picBunch: PICBUNCH_ARCH='${entry}' is not a HIP architecture. "
                    "Use a gfx target such as gfx1100 or gfx90a."
                )
            endif()
        endforeach()
    endif()
endfunction()

# Default of the alpakaVendor FFTW host backend.
#
# The CPU backend needs a host FFT, so FFTW is on.  CUDA uses cuFFT and HIP
# uses rocFFT, so a host FFTW install is not required there.  The value is only
# a default: an explicit -DalpakaV_DEP_FFTW=... wins because the cache entry is
# created without FORCE.
function(picbunch_default_host_fft cuda hip out_var)
    if(cuda OR hip)
        set(${out_var} OFF PARENT_SCOPE)
    else()
        set(${out_var} ON PARENT_SCOPE)
    endif()
endfunction()

# Refuse a configuration without any FFT backend: the field pipeline is an FFT
# convolution, so silently continuing only fails later in the compiler.
function(picbunch_check_fft_backends fftw cufft rocfft)
    if(NOT fftw AND NOT cufft AND NOT rocfft)
        message(
            FATAL_ERROR
            "picBunch: no FFT backend is enabled (alpakaV_DEP_FFTW=OFF, "
            "alpakaV_DEP_CUFFT=OFF, alpakaV_DEP_ROCFFT=OFF). Enable exactly the backend that matches "
            "the selected device: -DalpakaV_DEP_FFTW=ON for the CPU backend, "
            "-DalpakaV_DEP_CUFFT=ON for CUDA or -DalpakaV_DEP_ROCFFT=ON for HIP."
        )
    endif()
endfunction()
