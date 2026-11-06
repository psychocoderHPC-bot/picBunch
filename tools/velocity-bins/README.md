# Velocity-binning reference tools

This directory is a **reference tool with no standalone build**, not a second copy
of picBunch. It contains only the independent arbiter and the calibration helpers:

- `indfield.py` — adaptive Gauss-Legendre evaluation of the free-space,
  TSC-convolved moving point-charge field. It is the independent truth used by
  the physics tests and the near-shell review.
- `compare_velocity_bins.py` — compares binned fields written by the main
  tree's measurement driver against `indfield.py` and reports the per-box-count
  measured/model sum ratio (`--fit`).
- `measure_velocity_bins.cpp` — the synthetic mixed-velocity measurement driver
  consumed by `compare_velocity_bins.py`. The root `CMakeLists.txt` builds it as
  the `picBunchVelocityBinsMeasure` target against the main `include/` and
  `src/` tree; this directory has no CMake project of its own.
- `calibrate_velocity_bins.py` — reproduces
  `docs/measurements/velocity_bin_calibration_fit.txt` from the committed
  accuracy points in `docs/measurements/velocity_bin_calibration.csv`.

Everything else (headers, implementation, tests, docs, examples, paper) lives
in the top-level tree. Use the top-level [README](../../README.md),
[docs/algorithm.md](../../docs/algorithm.md) and
[docs/configuration.md](../../docs/configuration.md) as the source of truth.
