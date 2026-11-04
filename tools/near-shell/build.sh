#!/bin/bash
# Build the standalone H100 in-support near-shell kernel.
#
# Notes for HPC nodes (e.g. JURECA jrc*): /tmp is often a tiny tmpfs and nvcc
# writes its .ii intermediates there, so the build dies with "No space left on
# device". Point TMPDIR at a writable scratch volume (or pass TMPDIR=...).
set -e
export PATH=/usr/local/cuda-13.4/bin:$PATH
ARCH=${ARCH:-sm_90}
HERE=$(cd "$(dirname "$0")" && pwd)
# Use a scratch TMPDIR if the current one is too small for nvcc's intermediates.
if [ -z "${TMPDIR:-}" ]; then
    avail_kb=$(df -Pk /tmp 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt 1048576 ]; then   # < 1 GiB free
        export TMPDIR="${TMPDIR_OVERRIDE:-$HOME/nearshell-build-tmp}"
        mkdir -p "$TMPDIR"
    fi
fi
nvcc -O3 -std=c++17 -arch="$ARCH" -Xcompiler=-fopenmp "$HERE/nearshell_kernel.cu" -o "$HERE/nearshell_kernel"
echo "built $HERE/nearshell_kernel ($ARCH)"
