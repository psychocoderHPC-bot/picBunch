#!/usr/bin/env python3
"""Prove that check_provenance.py is not vacuous.

Six deliberately broken inputs are handed to the real checker in a temporary
records directory; each must make it exit non-zero:

1. a record whose data_sha256 does not match the file on disk;
2. a record that violates the JSON Schema (missing required field);
3. an empty records directory, which leaves figures unclaimed;
4. a record whose case_id is not registered in docs/paper/data/cases.json;
5. two records that claim the same figure id;
6. a fresh-reference record without reference_only, which must not pass as a
   measured field.

The schema, figures, cases registry, not-reproduced list and non-data list stay
the real ones.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(os.path.dirname(HERE))
REPO_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
CHECKER = os.path.join(PAPER_DIR, "scripts", "check_provenance.py")
RECORDS = os.path.join(PAPER_DIR, "provenance", "records")


def run_checker(records_dir):
    cmd = [sys.executable, CHECKER, "--records-dir", records_dir]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def _valid_record():
    name = sorted(os.listdir(RECORDS))[0]
    with open(os.path.join(RECORDS, name), "r") as fh:
        return json.load(fh)


def main():
    if not os.path.isfile(CHECKER):
        print("FAIL: checker not found at %s" % CHECKER)
        return 1
    base = _valid_record()
    failures = []

    with tempfile.TemporaryDirectory(prefix="provenance-selftest-") as tmp:
        # 1. tampered digest
        bad = json.loads(json.dumps(base))
        bad["data_sha256"][0]["sha256"] = "0" * 64
        with open(os.path.join(tmp, "bad-digest.json"), "w") as fh:
            json.dump(bad, fh)
        result = run_checker(tmp)
        ok = result.returncode != 0 and "sha256 mismatch" in result.stderr
        print("self-test tampered-data-sha256: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("tampered digest did not fail as expected")

        # 2. schema violation
        os.remove(os.path.join(tmp, "bad-digest.json"))
        bad = json.loads(json.dumps(base))
        del bad["measurement_kind"]
        with open(os.path.join(tmp, "bad-schema.json"), "w") as fh:
            json.dump(bad, fh)
        result = run_checker(tmp)
        ok = result.returncode != 0 and "missing required property" in result.stderr
        print("self-test schema-violation: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("schema violation did not fail as expected")

        # 3. unclaimed figures
        os.remove(os.path.join(tmp, "bad-schema.json"))
        result = run_checker(tmp)
        ok = result.returncode != 0 and "neither claimed" in result.stderr
        print("self-test unclaimed-figures: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("unclaimed figures did not fail as expected")

        # 4. case_id not present in the cases registry
        bad = json.loads(json.dumps(base))
        bad["case_id"] = "not-a-registered-case"
        with open(os.path.join(tmp, "bad-case.json"), "w") as fh:
            json.dump(bad, fh)
        result = run_checker(tmp)
        ok = result.returncode != 0 and "not registered" in result.stderr
        print("self-test unknown-case-id: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("unknown case_id did not fail as expected")
        os.remove(os.path.join(tmp, "bad-case.json"))

        # 5. two records claiming the same figure
        first = json.loads(json.dumps(base))
        first["case_id"] = "line-mpi"
        second = json.loads(json.dumps(base))
        second["case_id"] = "line-w4b"
        with open(os.path.join(tmp, "duplicate-a.json"), "w") as fh:
            json.dump(first, fh)
        with open(os.path.join(tmp, "duplicate-b.json"), "w") as fh:
            json.dump(second, fh)
        result = run_checker(tmp)
        ok = (result.returncode != 0
              and "duplicate claim for figure" in result.stderr)
        print("self-test duplicate-figure-claim: %s" % ("PASS" if ok else "FAIL"))
        if not ok:
            failures.append("duplicate figure claim did not fail as expected")
        os.remove(os.path.join(tmp, "duplicate-a.json"))
        os.remove(os.path.join(tmp, "duplicate-b.json"))

        # 6. fresh-reference must carry reference_only=true
        reference = None
        for name in sorted(os.listdir(RECORDS)):
            with open(os.path.join(RECORDS, name), "r") as fh:
                candidate = json.load(fh)
            if candidate.get("measurement_kind") == "fresh-reference":
                reference = candidate
                break
        if reference is None:
            print("self-test fresh-reference-needs-flag: SKIP (no record)")
        else:
            bad = json.loads(json.dumps(reference))
            bad.pop("reference_only", None)
            with open(os.path.join(tmp, "bad-reference.json"), "w") as fh:
                json.dump(bad, fh)
            result = run_checker(tmp)
            ok = (result.returncode != 0
                  and "requires reference_only=true" in result.stderr)
            print("self-test fresh-reference-needs-flag: %s"
                  % ("PASS" if ok else "FAIL"))
            if not ok:
                failures.append("fresh-reference without reference_only "
                                "did not fail as expected")
            os.remove(os.path.join(tmp, "bad-reference.json"))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
