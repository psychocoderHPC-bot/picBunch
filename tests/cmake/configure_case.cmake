# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
cmake_minimum_required(VERSION 3.25)
#
# One configure-only case of the configuration contract.  Each case gets a
# fresh build directory under WORK_DIR; nothing is compiled.
#
# Success cases reuse the pinned dependency sources already fetched by the
# build tree that runs the test (FETCHCONTENT_SOURCE_DIR_* plus
# FETCHCONTENT_FULLY_DISCONNECTED), so the test needs no network access.
#
# Inputs: CASE SOURCE_DIR WORK_DIR OPENPMD_DIR GENERATOR
#         PB_ALPAKA3_SRC PB_ALPAKAVENDOR_SRC PB_TOMLPLUSPLUS_SRC

set(_build "${WORK_DIR}/${CASE}")
file(REMOVE_RECURSE "${_build}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(_cache_value build name out_var)
    file(STRINGS "${build}/CMakeCache.txt" _lines REGEX "^${name}:[A-Z]+=")
    if(NOT _lines)
        message(FATAL_ERROR "cache variable ${name} is missing from ${build}/CMakeCache.txt")
    endif()
    list(GET _lines 0 _line)
    string(REGEX REPLACE "^[^=]*=" "" _value "${_line}")
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

function(_expect_cache build name expected)
    _cache_value("${build}" "${name}" _actual)
    if(NOT _actual STREQUAL "${expected}")
        message(FATAL_ERROR "expected ${name}=${expected}, got '${_actual}'")
    endif()
endfunction()

# Documented user variables only: no alpaka_*/alpakaV_* entry is passed.
set(_args
    -S "${SOURCE_DIR}"
    -B "${_build}"
    -G "${GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DPICBUNCH_BUILD_TESTS=OFF
    -DPICBUNCH_BUILD_GTESTS=OFF
    -DopenPMD_DIR=${OPENPMD_DIR}
)

set(_expect_success TRUE)
if(CASE STREQUAL "documented-only")
    # default backend: CPU, host FFTW on, device FFTs off
elseif(CASE STREQUAL "cuda-fftw-off")
    # sm_70 was dropped from nvcc 13, which rejects compute_70 at configure
    # time; the contract this case pins is the backend switch, not a specific
    # architecture, so probe the toolchain and use its lowest still-supported
    # capability (70 before CUDA 13, 80 from it on).
    execute_process(
        COMMAND ${PICBUNCH_TEST_NVCC} --version
        OUTPUT_VARIABLE _nvcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE _nvcc_version_error ERROR_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nvcc_version_status
    )
    # The banner (stdout, or stderr on some toolchains) must be parseable, so
    # the arch probe never guesses; fail with both streams otherwise.
    if(NOT _nvcc_version MATCHES "release ([0-9]+)\."
       AND NOT _nvcc_version_error MATCHES "release ([0-9]+)\.")
        message(FATAL_ERROR "cannot parse the nvcc version (rc ${_nvcc_version_status}): out='${_nvcc_version}' err='${_nvcc_version_error}'")
    endif()
    set(_probe_arch 80)
    if(CMAKE_MATCH_1 LESS 13)
        set(_probe_arch 70)
    endif()
    list(APPEND _args -DPICBUNCH_ENABLE_CUDA=ON -DPICBUNCH_ARCH=${_probe_arch} -DalpakaV_DEP_FFTW=OFF)
elseif(CASE STREQUAL "cuda-conflict")
    set(_expect_success FALSE)
    list(APPEND _args -DPICBUNCH_ENABLE_CUDA=ON -DPICBUNCH_ENABLE_HIP=ON)
    set(_expect_regex "mutually")
elseif(CASE STREQUAL "cuda-bad-arch")
    set(_expect_success FALSE)
    list(APPEND _args -DPICBUNCH_ENABLE_CUDA=ON -DPICBUNCH_ARCH=gfx1100)
    set(_expect_regex "not a CUDA architecture")
elseif(CASE STREQUAL "hip-bad-arch")
    set(_expect_success FALSE)
    list(APPEND _args -DPICBUNCH_ENABLE_HIP=ON -DPICBUNCH_ARCH=70)
    set(_expect_regex "not a HIP architecture")
else()
    message(FATAL_ERROR "unknown configure test case '${CASE}'")
endif()

if(_expect_success)
    foreach(_pair
        "ALPAKA3|${PB_ALPAKA3_SRC}"
        "ALPAKAVENDOR|${PB_ALPAKAVENDOR_SRC}"
        "TOMLPLUSPLUS|${PB_TOMLPLUSPLUS_SRC}"
    )
        string(REPLACE "|" ";" _fields "${_pair}")
        list(GET _fields 0 _dep_name)
        list(GET _fields 1 _dep_dir)
        if(NOT IS_DIRECTORY "${_dep_dir}")
            message(FATAL_ERROR "dependency source for ${_dep_name} is missing: '${_dep_dir}'")
        endif()
        list(APPEND _args "-DFETCHCONTENT_SOURCE_DIR_${_dep_name}=${_dep_dir}")
    endforeach()
    list(APPEND _args -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} ${_args}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)
set(_output "${_stdout}\n${_stderr}")

if(_expect_success)
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "configure failed (exit ${_status}):\n${_output}")
    endif()
else()
    if(_status EQUAL 0)
        message(FATAL_ERROR "configure succeeded but the case must be rejected:\n${_output}")
    endif()
    if(NOT _output MATCHES "${_expect_regex}")
        message(FATAL_ERROR "rejection does not mention '${_expect_regex}':\n${_output}")
    endif()
    return()
endif()

# --- behaviour ------------------------------------------------------------
if(CASE STREQUAL "documented-only")
    _expect_cache("${_build}" alpakaV_DEP_FFTW ON)
    _expect_cache("${_build}" alpakaV_DEP_CUFFT OFF)
    _expect_cache("${_build}" alpakaV_DEP_ROCFFT OFF)
    _expect_cache("${_build}" PICBUNCH_ENABLE_WARNINGS ON)

    # Item 6: the warning set reaches picBunch's own translation units.
    file(READ "${_build}/compile_commands.json" _compile_commands)
    if(NOT _compile_commands MATCHES "src/Deposit.cpp")
        message(FATAL_ERROR "compile_commands.json has no entry for src/Deposit.cpp")
    endif()
    if(NOT _compile_commands MATCHES "-Wall")
        message(FATAL_ERROR "picBunch's own sources are not compiled with -Wall")
    endif()
    if(NOT _compile_commands MATCHES "-Wextra")
        message(FATAL_ERROR "picBunch's own sources are not compiled with -Wextra")
    endif()

    # Item 4: the dependency knobs are advanced, the documented switches are not.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -L -N "${_build}"
        RESULT_VARIABLE _list_status
        OUTPUT_VARIABLE _listing
        ERROR_VARIABLE _list_error
    )
    if(NOT _list_status EQUAL 0)
        message(FATAL_ERROR "cmake -L -N failed: ${_list_error}")
    endif()
    foreach(_hidden alpakaV_DEP_FFTW alpakaV_DEP_CUFFT alpaka_TESTS alpaka_DEP_CUDA)
        if(_listing MATCHES "${_hidden}")
            message(FATAL_ERROR "${_hidden} is expected to be an advanced (hidden) cache variable")
        endif()
    endforeach()
    foreach(_visible
        PICBUNCH_ENABLE_CUDA
        PICBUNCH_ENABLE_HIP
        PICBUNCH_ARCH
        PICBUNCH_BUILD_TESTS
        PICBUNCH_BUILD_GTESTS
        PICBUNCH_BUILD_BENCH
        PICBUNCH_SANITIZE
        openPMD_DIR
    )
        if(NOT _listing MATCHES "${_visible}")
            message(FATAL_ERROR "${_visible} must stay a visible, documented cache variable")
        endif()
    endforeach()
elseif(CASE STREQUAL "cuda-fftw-off")
    # The explicit OFF survives: no silent re-enable of the host backend.
    _expect_cache("${_build}" alpakaV_DEP_FFTW OFF)
    _expect_cache("${_build}" alpakaV_DEP_CUFFT ON)
    _expect_cache("${_build}" alpakaV_DEP_ROCFFT OFF)
    if(NOT _output MATCHES "FFT backends, host FFTW=OFF cuFFT=ON")
        message(FATAL_ERROR "the backend summary does not report FFTW=OFF cuFFT=ON:\n${_output}")
    endif()
endif()
