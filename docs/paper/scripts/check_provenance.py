#!/usr/bin/env python3
"""Check the provenance of every figure under ``docs/paper/figs``.

The checker enforces three facts:

1. every ``provenance/records/*.json`` file conforms to
   ``provenance/provenance.schema.json``;
2. every referenced SHA-256 digest matches the file on disk (data arrays, and
   any config/binary/script artifact that carries a path);
3. every figure in ``docs/paper/figs`` is claimed by a provenance record or is
   explicitly declared ``not-reproduced`` with a reason.

A record with ``measurement_kind == "not-reproduced"`` may have an empty
``data_sha256``.  That is the honest "renderer exists, campaign data pending"
state: the normal checker accepts it, the strict gate rejects it, and a record
for any other measurement_kind must still list at least one data file.

Two identity rules close bypasses that the JSON Schema cannot express:

* every ``case_id`` must be a registered case in
  ``docs/paper/data/cases.json``;
* exactly one record may claim a figure id.  A record may list several
  figures, but another record claiming any of them is an error.

``--strict`` is the final-gate mode: it additionally requires that every data
figure (every figure not listed in ``provenance/non_data_figures.json``) is
claimed by a record with ``measurement_kind == "fresh-h100"`` or by a
``measurement_kind == "fresh-reference"`` record that sets
``reference_only: true``.  ``fresh-reference`` documents a freshly recomputed
analytic/exact reference (for example a CPU order-12 reference) rather than a
GPU measurement.  The explicit flag keeps a measured field from being
mislabelled as a reference.  A not-reproduced or legacy figure still fails the
strict gate.

The script uses only the Python standard library, so it runs in a fresh clone.

Usage:
  python3 docs/paper/scripts/check_provenance.py
  python3 docs/paper/scripts/check_provenance.py --strict
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
PROV_DIR = os.path.join(PAPER_DIR, "provenance")
DEFAULT_SCHEMA = os.path.join(PROV_DIR, "provenance.schema.json")
DEFAULT_RECORDS = os.path.join(PROV_DIR, "records")
DEFAULT_FIGS = os.path.join(PAPER_DIR, "figs")
DEFAULT_NOT_REPRODUCED = os.path.join(PROV_DIR, "not_reproduced.json")
DEFAULT_NON_DATA = os.path.join(PROV_DIR, "non_data_figures.json")
DEFAULT_CASES = os.path.join(PAPER_DIR, "data", "cases.json")


# --------------------------------------------------------------------------
# Minimal JSON-Schema validator (the subset used by provenance.schema.json)
# --------------------------------------------------------------------------
def _resolve(ref, root):
    if not ref.startswith("#/"):
        raise ValueError("unsupported $ref %r" % ref)
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def _type_ok(instance, name):
    if name == "null":
        return instance is None
    if name == "object":
        return isinstance(instance, dict)
    if name == "array":
        return isinstance(instance, list)
    if name == "string":
        return isinstance(instance, str)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "boolean":
        return isinstance(instance, bool)
    raise ValueError("unknown schema type %r" % name)


def validate(instance, schema, root, path="$"):
    """Return a list of human-readable validation errors."""
    errors = []
    if "$ref" in schema:
        return validate(instance, _resolve(schema["$ref"], root), root, path)

    if "anyOf" in schema:
        sub_errors = [validate(instance, s, root, path) for s in schema["anyOf"]]
        if all(sub_errors):
            errors.append("%s: does not match any allowed schema" % path)
            for sub in sub_errors:
                errors.extend("  " + e for e in sub)
        return errors

    typ = schema.get("type")
    if typ is not None:
        names = typ if isinstance(typ, list) else [typ]
        if not any(_type_ok(instance, n) for n in names):
            errors.append("%s: expected type %s, got %s"
                          % (path, "/".join(names), type(instance).__name__))
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: %r is not one of %r"
                      % (path, instance, schema["enum"]))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("%s: shorter than %d characters" % (path, schema["minLength"]))
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append("%s: %r does not match %s"
                          % (path, instance, schema["pattern"]))

    if isinstance(instance, int) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: %r is below minimum %r"
                          % (path, instance, schema["minimum"]))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("%s: needs at least %d item(s)" % (path, schema["minItems"]))
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], root, "%s[%d]" % (path, i)))

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required property %r" % (path, key))
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], root, "%s.%s" % (path, key)))
            elif additional is False:
                errors.append("%s: unexpected property %r" % (path, key))
            elif isinstance(additional, dict):
                errors.extend(validate(value, additional, root, "%s.%s" % (path, key)))
    return errors


# --------------------------------------------------------------------------
# Provenance checks
# --------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _check_digest(digest, label, errors):
    path = _resolve_path(digest["path"])
    if not os.path.isfile(path):
        errors.append("%s: referenced file does not exist: %s" % (label, digest["path"]))
        return
    actual = sha256_file(path)
    if actual != digest["sha256"]:
        errors.append("%s: sha256 mismatch for %s (record %s, disk %s)"
                      % (label, digest["path"], digest["sha256"], actual))


def check_record(record, label, schema, errors):
    schema_errors = validate(record, schema, schema)
    errors.extend("%s: %s" % (label, e) for e in schema_errors)
    if not isinstance(record, dict) or schema_errors:
        # Schema shape is unusable or already reported; do not chase fields.
        return
    digests = record.get("data_sha256") or []
    if record.get("measurement_kind") != "not-reproduced" and not digests:
        errors.append("%s: data_sha256 must list at least one data file unless "
                      "measurement_kind is 'not-reproduced'" % label)
    for i, digest in enumerate(digests):
        if isinstance(digest, dict) and "path" in digest and "sha256" in digest:
            _check_digest(digest, "%s.data_sha256[%d]" % (label, i), errors)
    for field in ("config_sha256", "binary_sha256", "script_sha256"):
        digest = record.get(field)
        if isinstance(digest, dict) and "path" in digest:
            _check_digest(digest, "%s.%s" % (label, field), errors)
    generator = record.get("generator")
    if isinstance(generator, dict):
        for name, digest in sorted(generator.items()):
            if isinstance(digest, dict) and "path" in digest:
                _check_digest(digest, "%s.generator.%s" % (label, name), errors)
    if record.get("measurement_kind") == "fresh-reference":
        if record.get("reference_only") is not True:
            errors.append("%s: measurement_kind 'fresh-reference' requires "
                          "reference_only=true" % label)
        if not record.get("host"):
            errors.append("%s: measurement_kind 'fresh-reference' requires a "
                          "non-empty host" % label)
        if not isinstance(generator, dict) or not generator:
            errors.append("%s: measurement_kind 'fresh-reference' requires "
                          "generator digests" % label)


def _load_json(path, errors, what):
    if not os.path.isfile(path):
        errors.append("%s: missing file %s" % (what, path))
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        errors.append("%s: cannot read %s: %s" % (what, path, exc))
        return None


def run(records_dir, figs_dir, schema_path, not_reproduced_path,
        non_data_path, strict, cases_path=DEFAULT_CASES):
    errors = []

    schema = _load_json(schema_path, errors, "schema")
    if schema is None:
        return errors, {}

    registry = _load_json(cases_path, errors, "case registry")
    known_cases = None
    if isinstance(registry, dict) and isinstance(registry.get("cases"), dict):
        known_cases = set(registry["cases"])
    elif registry is not None:
        errors.append("case registry: %s needs a 'cases' object" % cases_path)

    if not os.path.isdir(records_dir):
        errors.append("records directory does not exist: %s" % records_dir)
        record_files = []
    else:
        record_files = sorted(
            os.path.join(records_dir, name)
            for name in os.listdir(records_dir) if name.endswith(".json"))

    if not record_files:
        errors.append("no provenance records found in %s" % records_dir)

    figures = sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(figs_dir) if name.endswith(".pdf")) \
        if os.path.isdir(figs_dir) else []
    if not figures:
        errors.append("no figures found in %s" % figs_dir)

    figure_claims = {}
    figure_claim_sources = {}
    records = []
    seen_cases = {}
    for path in record_files:
        label = os.path.relpath(path, REPO_ROOT)
        try:
            with open(path, "r") as fh:
                record = json.load(fh)
        except (OSError, ValueError) as exc:
            errors.append("%s: cannot parse JSON: %s" % (label, exc))
            continue
        records.append(record)
        check_record(record, label, schema, errors)
        case_id = record.get("case_id")
        if case_id in seen_cases:
            errors.append("%s: duplicate case_id %r (also in %s)"
                          % (label, case_id, seen_cases[case_id]))
        elif case_id is not None:
            seen_cases[case_id] = label
        if (isinstance(case_id, str) and known_cases is not None
                and case_id not in known_cases):
            errors.append("%s: case_id %r is not registered in %s"
                          % (label, case_id, os.path.relpath(cases_path, REPO_ROOT)))
        for fig in record.get("figure_ids", []) or []:
            if fig in figure_claims:
                errors.append("%s: duplicate claim for figure %r (already "
                              "claimed by %s)"
                              % (label, fig, figure_claim_sources[fig]))
            else:
                figure_claim_sources[fig] = label
            figure_claims.setdefault(fig, []).append(record)
            if fig not in figures:
                errors.append("%s: claims unknown figure %r (no %s.pdf in %s)"
                              % (label, fig, fig, figs_dir))

    not_reproduced = {}
    data = _load_json(not_reproduced_path, errors, "not-reproduced")
    if data is not None:
        declared = data.get("figures", {}) if isinstance(data, dict) else {}
        if not isinstance(declared, dict):
            errors.append("not-reproduced: 'figures' must be an object")
            declared = {}
        not_reproduced = declared
        for fig, entry in declared.items():
            if fig not in figures:
                errors.append("not-reproduced: unknown figure %r" % fig)
            reason = entry.get("reason") if isinstance(entry, dict) else None
            if not isinstance(reason, str) or not reason.strip():
                errors.append("not-reproduced: %r needs a non-empty reason" % fig)
            if fig in figure_claims:
                errors.append("not-reproduced: %r is also claimed by a record" % fig)

    non_data = {}
    if os.path.isfile(non_data_path):
        nd = _load_json(non_data_path, errors, "non-data")
        if isinstance(nd, dict):
            non_data = nd.get("figures", {}) or {}
            if not isinstance(non_data, dict):
                errors.append("non-data: 'figures' must be an object")
                non_data = {}
    else:
        errors.append("non-data: missing file %s" % non_data_path)
    for fig in non_data:
        if fig not in figures:
            errors.append("non-data: unknown figure %r" % fig)

    missing = [f for f in figures if f not in figure_claims and f not in not_reproduced]
    for fig in missing:
        errors.append("figure %r is neither claimed by a provenance record nor "
                      "declared not-reproduced" % fig)

    if strict:
        for fig in figures:
            if fig in non_data:
                continue
            claims = figure_claims.get(fig, [])
            fresh = [r for r in claims
                     if r.get("measurement_kind") == "fresh-h100"
                     or (r.get("measurement_kind") == "fresh-reference"
                         and r.get("reference_only") is True)]
            if not fresh:
                if fig in not_reproduced:
                    detail = "declared not-reproduced"
                elif claims:
                    kinds = sorted({str(r.get("measurement_kind")) for r in claims})
                    detail = "claimed as %s" % ", ".join(kinds)
                else:
                    detail = "unclaimed"
                errors.append("strict: data figure %r is not fresh-h100 or a "
                              "fresh-reference (reference_only) (%s)"
                              % (fig, detail))

    summary = {
        "records": len(records),
        "figures_total": len(figures),
        "figures_claimed": sum(1 for f in figures if f in figure_claims),
        "figures_not_reproduced": sum(1 for f in figures if f in not_reproduced),
        "figures_non_data": sum(1 for f in figures if f in non_data),
        "figures_missing": len(missing),
        "cases_registered": len(known_cases) if known_cases is not None else 0,
        "strict": strict,
    }
    return errors, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records-dir", default=DEFAULT_RECORDS)
    ap.add_argument("--figs-dir", default=DEFAULT_FIGS)
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--not-reproduced", default=DEFAULT_NOT_REPRODUCED)
    ap.add_argument("--non-data", default=DEFAULT_NON_DATA)
    ap.add_argument("--strict", action="store_true",
                    help="require fresh-h100 or reference-only fresh-reference "
                         "provenance for every data figure")
    ap.add_argument("--cases", default=DEFAULT_CASES,
                    help="case registry that every record case_id must name")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = ap.parse_args(argv)

    errors, summary = run(a.records_dir, a.figs_dir, a.schema,
                          a.not_reproduced, a.non_data, a.strict, a.cases)
    if a.json:
        print(json.dumps(summary, indent=1, sort_keys=True))
    if errors:
        for err in errors:
            print("FAIL: " + err, file=sys.stderr)
        print("provenance check FAILED (%d problem(s))" % len(errors), file=sys.stderr)
        return 1
    print("provenance check PASSED: %d record(s), %d/%d figure(s) covered, "
          "%d declared not-reproduced, %d non-data (strict=%s)"
          % (summary["records"], summary["figures_claimed"],
             summary["figures_total"], summary["figures_not_reproduced"],
             summary["figures_non_data"], summary["strict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
