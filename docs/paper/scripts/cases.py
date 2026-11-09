"""Repo-relative paths and the machine-readable case registry.

Every path is resolved from this file's location, so the pipeline runs from a
fresh clone with no environment variables and no absolute run directories.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)            # docs/paper
REPO_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
DATA_DIR = os.path.join(PAPER_DIR, "data")


def data(*parts):
    return os.path.join(DATA_DIR, *parts)


def load_registry():
    path = os.path.join(DATA_DIR, "cases.json")
    with open(path, "r") as fh:
        return json.load(fh)


def case(case_id):
    reg = load_registry()
    try:
        return reg["cases"][case_id]
    except KeyError:
        raise KeyError(f"case {case_id!r} is not in {DATA_DIR}/cases.json")
