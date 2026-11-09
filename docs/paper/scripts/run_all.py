#!/usr/bin/env python3
"""Render every paper figure from the in-repo reduced data.

Usage:
  python3 docs/paper/scripts/run_all.py --out-dir <staging-dir>

The output directory is a staging area.  The paper's ``docs/paper/figs`` is
written only by the final replot from fresh measurements.  Rendering is
deterministic: Agg backend, fixed PDF/PNG metadata, no RNG, no timestamps.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import axis  # noqa: E402
import check_provenance  # noqa: E402
import render_large  # noqa: E402
import render_pergamma  # noqa: E402
import render_scaling  # noqa: E402
import render_slices  # noqa: E402


def _write_assertions(path, results, self_test):
    lines = ["# Axis-convention assertions", "",
             "Each data set plots its recorded physical layout. The +y families",
             "(bdata2, var1b, hires) put the beam axis y horizontally. LARGE1 is",
             "axial-x at gamma=10 and stationary at gamma=1; its x-y cut puts the",
             "beam axis x horizontally and its z-y cut puts z horizontally.", ""]
    lines.append("## Mapping self-test")
    for r in self_test:
        lines.append(f"self-test {r['dataset']}/{r['plane']} "
                     f"stored={tuple(r['stored_axes'])} "
                     f"-> display={tuple(r['display_axes'])}: "
                     f"{'PASS' if r['pass'] else 'FAIL'}")
    lines.append("")
    lines.append("## Per-figure assertions")
    for r in results:
        peak = r.get("peak") or {}
        cen = r.get("centroid") or {}
        extra = ""
        if r.get("longitudinal_contraction"):
            lc = r["longitudinal_contraction"]
            extra = (f" contraction_ratio={lc['ratio']:.3f}")
        lines.append(
            f"{r['figure']} panel={r['panel']} dataset={r['dataset']} "
            f"plane={r['plane']} propagation={r['propagation_axis']} "
            f"stored={tuple(r['stored_axes'])} "
            f"display={tuple(r['display_axes'])} "
            f"horiz={r['horizontal_axis']} vert={r['vertical_axis']} "
            f"peak(h={peak.get('horizontal')},v={peak.get('vertical')}) "
            f"centroid(h={cen.get('horizontal')},v={cen.get('vertical')})"
            f"{extra} -> {'PASS' if r['pass'] else 'FAIL'}")
    lines.append("")
    lines.append(f"figures checked: {len(results)}; "
                 f"passed: {sum(1 for r in results if r['pass'])}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True,
                    help="staging directory for the rendered PDF/PNG files")
    ap.add_argument("--assertions", default=None,
                    help="assertion log (default: <out-dir>/axis-assertions.txt)")
    ap.add_argument("--check-provenance", action="store_true",
                    help="run check_provenance.py after rendering and fail on any gap")
    ap.add_argument("--strict-provenance", action="store_true",
                    help="with --check-provenance, require fresh-h100 for data figures")
    ap.add_argument("--strict", action="store_true",
                    help="fail if a rendered figure's reduced data is still pending")
    ap.add_argument("--include-unreferenced", action="store_true",
                    help="also render figures the manuscript no longer includes "
                         "(the V100 F_mpi_rank_scaling)")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)

    self_test = axis.self_test()
    results = []
    results += render_slices.run(out)
    results += render_large.run(out)
    results += render_pergamma.main(out)
    pending = render_scaling.run(out, include_unreferenced=a.include_unreferenced)

    log = a.assertions or os.path.join(out, "axis-assertions.txt")
    _write_assertions(log, results, self_test)
    for item in pending:
        print("PENDING figure %s: reduced table %s is absent for campaign case %s"
              % (item["figure"], item["path"], item["case_id"]), file=sys.stderr)
    summary = {"figures": len(results),
               "passed": sum(1 for r in results if r["pass"]),
               "assertions": results, "self_test": self_test,
               "pending": pending, "out_dir": out}
    with open(os.path.join(out, "render-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)

    if a.check_provenance:
        errors, prov = check_provenance.run(
            check_provenance.DEFAULT_RECORDS,
            check_provenance.DEFAULT_FIGS,
            check_provenance.DEFAULT_SCHEMA,
            check_provenance.DEFAULT_NOT_REPRODUCED,
            check_provenance.DEFAULT_NON_DATA,
            a.strict_provenance)
        for err in errors:
            print("provenance FAIL: " + err, file=sys.stderr)
        summary["provenance"] = prov
        summary["provenance_passed"] = not errors
        with open(os.path.join(out, "render-summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        if errors:
            print("provenance check FAILED (%d problem(s)); add a record or a "
                  "not-reproduced declaration" % len(errors), file=sys.stderr)
            return 1
        print("provenance check PASSED: %d figure(s) covered%s"
              % (prov["figures_claimed"],
                 " (strict fresh-h100)" if a.strict_provenance else ""))
    strict_data = a.strict or a.strict_provenance
    if pending and strict_data:
        print("strict: %d figure(s) pending reduced data; the pipeline did not "
              "silently skip them" % len(pending), file=sys.stderr)
        return 1
    return 0 if summary["passed"] == summary["figures"] else 1


if __name__ == "__main__":
    sys.exit(main())
