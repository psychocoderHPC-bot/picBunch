#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: Rene Widera
"""Read an openPMD series and report its backend and available meshes.

Prints one JSON object on stdout.  Uses openPMD-api's Python bindings when the
interpreter given by ``--python`` can import them; the driver treats a missing
binding as an unverified row rather than a failed run.
"""

import json
import sys


def read_only_access(io):
    access = getattr(io, "Access", None)
    if access is not None and hasattr(access, "read_only"):
        return access.read_only
    return io.Access_Type.READ_ONLY


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"verify": "usage-error", "error": "expected one path"}))
        return 2
    path = sys.argv[1]
    try:
        import openpmd_api as io
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(json.dumps({"verify": "python-unavailable", "error": str(exc)}))
        return 0
    try:
        series = io.Series(path, read_only_access(io))
        iteration_keys = list(series.iterations)
        meshes = []
        if iteration_keys:
            iteration = series.iterations[iteration_keys[0]]
            meshes = sorted(list(iteration.meshes))
        print(
            json.dumps(
                {
                    "verify": "openpmd",
                    "backend": series.backend,
                    "iterations": len(iteration_keys),
                    "meshes": meshes,
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"verify": "error", "error": str(exc)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
