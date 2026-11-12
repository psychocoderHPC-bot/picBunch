#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
# Build the CPU-only independent reference driver used by the precision
# campaign.  No CUDA, openPMD or alpaka dependency: a C++20 compiler with
# OpenMP is enough.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/exact_field_driver}"
cxx="${CXX:-g++}"
"$cxx" -std=c++20 -O3 -march=native -fopenmp \
  -I "$here/refinc" "$here/exact_field_driver.cpp" -o "$out"
echo "built $out"
