#!/usr/bin/env python3
"""Fail if a paper renderer sets plotting style locally.

Every style choice lives in ``figstyle.py``.  The renderers pass data and
geometry, never fonts, canvas sizes, colormaps, raster interpolation, mask
colours or colour-bar styling.  This checker scans the renderer sources for a
local override and exits nonzero with ``file:line`` evidence.

Usage::

    python3 docs/paper/scripts/check_figstyle_overrides.py
    python3 docs/paper/scripts/check_figstyle_overrides.py --verbose

The check is deliberately token-based: an override must be deleted, not
renamed.  ``figstyle.py`` itself is the only file allowed to carry these
tokens.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "figstyle.py"

# (rule name, compiled pattern).  A renderer must not match any of these.
FORBIDDEN = (
    ("font-size override", re.compile(r"\bfontsize\s*=")),
    ("tick-label-size override", re.compile(r"\blabelsize\s*=")),
    ("dpi override", re.compile(r"\bdpi\s*=")),
    ("colormap override", re.compile(r"\bcmap\s*=")),
    ("canvas-size override", re.compile(r"\bfigsize\s*=")),
    ("raster interpolation override", re.compile(r"\binterpolation\s*=")),
    ("raster shading override", re.compile(r"\bshading\s*=")),
    ("mask grey literal", re.compile(r"#(?:[cC]{6}|[dD]{6})\b")),
    ("raw colour literal", re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")),
    ("local mask constant", re.compile(r"\bMASKED_[A-Z_]+\s*=")),
    ("local colormap constant", re.compile(r"\bCMAP_[A-Z_]+\s*=")),
    ("local norm-floor constant",
     re.compile(r"\b(?:ERROR_FLOOR|ERR_FLOOR|SQ_FLOOR_RATIO)\s*=")),
    ("raw rcParams write",
     re.compile(r"\brcParams\s*(?:\[[^\]]*\]\s*=|\.update\s*\()")),
    ("raw face colour", re.compile(r"\bset_facecolor\s*\(")),
    ("direct raster call", re.compile(r"\.(?:imshow|pcolormesh)\s*\(")),
    ("legacy style entry point",
     re.compile(r"\bapply_style\s*\(|\bapply_field_style\s*\(")),
)


def renderer_files():
    """Every source that draws a paper figure."""
    files = sorted(HERE.glob("render_*.py"))
    files.append(HERE / "run_all.py")
    return [p for p in files if p.name != "check_figstyle_overrides.py"]


def scan(path):
    hits = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for rule, pattern in FORBIDDEN:
            if pattern.search(line):
                hits.append((path, lineno, rule, line.strip()))
    return hits


def check_canonical():
    """The canonical module must expose the single entry point and no alias."""
    if not CANONICAL.is_file():
        return [f"missing canonical style module: {CANONICAL}"]
    text = CANONICAL.read_text(encoding="utf-8")
    problems = []
    if "def apply_paper_style(" not in text:
        problems.append("figstyle.py has no apply_paper_style() entry point")
    for legacy in ("def apply_style(", "def apply_field_style("):
        if legacy in text:
            problems.append(f"figstyle.py still exposes legacy {legacy}")
    return problems


def check_audits(outdir):
    """Every rendered figure must record the same non-kind style tokens."""
    import glob
    paths = sorted(glob.glob(str(Path(outdir) / "audit" / "*.json")))
    if not paths:
        return [f"no audit JSON under {outdir}/audit"], set()
    problems = []
    kinds = set()
    reference = None
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            tokens = json.load(fh).get("style_tokens")
        if not tokens:
            problems.append(f"{path}: missing style_tokens")
            continue
        kinds.add(tokens.get("kind"))
        common = {k: v for k, v in tokens.items() if k != "kind"}
        if reference is None:
            reference = common
        elif common != reference:
            differing = sorted(k for k in set(common) | set(reference)
                               if common.get(k) != reference.get(k))
            problems.append(
                f"{path}: style tokens differ from the first figure: "
                f"{', '.join(differing)}")
    return problems, kinds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true",
                    help="print every scanned file")
    ap.add_argument("--audit-dir", default=None,
                    help="also require identical style_tokens in "
                         "<audit-dir>/audit/*.json")
    a = ap.parse_args(argv)

    files = renderer_files()
    if not files:
        print("figstyle override check: FAIL (no renderer sources found)",
              file=sys.stderr)
        return 2

    hits = []
    for path in files:
        hits.extend(scan(path))
        if a.verbose:
            print(f"scanned {path.relative_to(HERE)}")

    problems = check_canonical()
    kinds = set()
    if a.audit_dir is not None:
        audit_problems, kinds = check_audits(a.audit_dir)
        problems.extend(audit_problems)
    for path, lineno, rule, line in hits:
        problems.append(f"{path.relative_to(HERE)}:{lineno}: {rule}: {line}")
    if problems:
        print("figstyle override check: FAIL", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"{len(problems)} problem(s); move every style value into "
              f"{CANONICAL.name}", file=sys.stderr)
        return 1

    extra = ""
    if a.audit_dir is not None:
        extra = f"; audits agree ({', '.join(sorted(str(k) for k in kinds))})"
    print(f"figstyle override check: PASS ({len(files)} renderer sources, "
          f"all style in {CANONICAL.name})" + extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
