#!/usr/bin/env python3
"""Check that ``data/cases.json`` agrees with the provenance records.

The case registry is the human-facing summary; the record is the
machine-checked measurement.  They must not drift apart.  For every case that
has a record this script requires an exact match of

* ``measurement_kind``,
* ``host``, and
* the ``source_data`` path/SHA-256 set (order-insensitive).

When both sides carry ``gpu_model`` or ``io_backend`` those must match too.
A registry case without a record (``line-mpi``) is reported and skipped: it is
legacy in both the registry and the absence of a record, so there is nothing to
compare.

Usage:
  python3 docs/paper/scripts/check_cases_consistency.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
CASES = os.path.join(PAPER_DIR, "data", "cases.json")
RECORDS = os.path.join(PAPER_DIR, "provenance", "records")


def _digests(entries):
    return {(e["path"], e["sha256"]) for e in entries or []}


def main():
    with open(CASES, "r") as fh:
        registry = json.load(fh)["cases"]
    records = {}
    for path in sorted(glob.glob(os.path.join(RECORDS, "*.json"))):
        with open(path, "r") as fh:
            record = json.load(fh)
        records[record["case_id"]] = record

    errors = []
    checked = 0
    for case_id, case in sorted(registry.items()):
        record = records.get(case_id)
        if record is None:
            print("skip %s: no provenance record" % case_id)
            continue
        checked += 1
        if case.get("measurement_kind") != record.get("measurement_kind"):
            errors.append("%s: measurement_kind registry=%r record=%r"
                          % (case_id, case.get("measurement_kind"),
                             record.get("measurement_kind")))
        if case.get("host") != record.get("host"):
            errors.append("%s: host registry=%r record=%r"
                          % (case_id, case.get("host"), record.get("host")))
        reg_data = _digests(case.get("source_data"))
        rec_data = _digests(record.get("data_sha256"))
        if reg_data != rec_data:
            errors.append("%s: source_data differs\n  registry=%s\n  record=%s"
                          % (case_id, sorted(reg_data), sorted(rec_data)))
        for field in ("gpu_model", "io_backend"):
            reg_value = case.get(field)
            rec_value = record.get(field)
            if reg_value is not None and rec_value is not None and reg_value != rec_value:
                errors.append("%s: %s registry=%r record=%r"
                              % (case_id, field, reg_value, rec_value))

    if errors:
        for err in errors:
            print("FAIL: " + err, file=sys.stderr)
        print("cases/record consistency FAILED (%d problem(s))" % len(errors),
              file=sys.stderr)
        return 1
    print("cases/record consistency PASSED: %d case(s) checked" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
