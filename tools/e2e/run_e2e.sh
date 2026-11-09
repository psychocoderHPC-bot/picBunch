#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
#
# Thin entry point for the end-to-end wall-time harness.  All work happens in
# e2e.py; this wrapper only selects the interpreter.  Set E2E_PYTHON when the
# default python3 cannot import openpmd_api.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${E2E_PYTHON:-python3}"
exec "$python" "$here/e2e.py" "$@"
