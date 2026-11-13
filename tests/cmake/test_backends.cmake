# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
cmake_minimum_required(VERSION 3.25)
#
# Fast contract test for the backend policy.  Each case runs in a child
# `cmake -P` process, so no project configuration and no network access is
# needed; the whole file runs in well under a second.
#
# Entry layout: <case>|<expected exit>|<expected output regex>

set(_cases
    "cuda-arch-numeric|0|"
    "cuda-arch-sm-prefix|0|"
    "cuda-arch-list|0|"
    "cuda-arch-reject-gfx|1|not a CUDA architecture"
    "cuda-arch-reject-dot|1|not a CUDA architecture"
    "hip-arch-gfx|0|"
    "hip-arch-reject-numeric|1|not a HIP architecture"
    "backend-conflict|1|mutually"
    "backend-hip-only|0|"
    "fftw-default-cpu|0|RESULT=ON"
    "fftw-default-cuda|0|RESULT=OFF"
    "fftw-default-hip|0|RESULT=OFF"
    "fftw-explicit-off-wins|0|RESULT=OFF"
    "fftw-explicit-on-wins|0|RESULT=ON"
    "fft-backends-none|1|no FFT backend is enabled"
    "fft-backends-cufft|0|"
)

set(_failures "")
foreach(_case IN LISTS _cases)
    string(REPLACE "|" ";" _fields "${_case}")
    list(GET _fields 0 _name)
    list(GET _fields 1 _expected_status)
    list(GET _fields 2 _expected_regex)

    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -DPICBUNCH_CMAKE_DIR=${PICBUNCH_CMAKE_DIR} -DCASE=${_name} -P
            ${CMAKE_CURRENT_LIST_DIR}/backend_case.cmake
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
    )

    if(_expected_status EQUAL 0)
        if(NOT _status EQUAL 0)
            string(APPEND _failures "\n  ${_name}: expected success, got exit ${_status}\n${_stderr}")
        elseif(NOT _expected_regex STREQUAL "" AND NOT "${_stdout}${_stderr}" MATCHES "${_expected_regex}")
            string(
                APPEND _failures
                "\n  ${_name}: output '${_stdout}${_stderr}' does not match '${_expected_regex}'"
            )
        endif()
    else()
        if(_status EQUAL 0)
            string(APPEND _failures "\n  ${_name}: expected rejection, the case succeeded")
        elseif(NOT "${_stdout}${_stderr}" MATCHES "${_expected_regex}")
            string(APPEND _failures "\n  ${_name}: rejection message does not match '${_expected_regex}':\n${_stderr}")
        endif()
    endif()
endforeach()

if(_failures)
    message(FATAL_ERROR "backend policy contract failed:${_failures}")
endif()
message(STATUS "backend policy contract: all cases passed")
