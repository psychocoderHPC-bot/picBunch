"""Single source of truth for the beam/slice axis convention.

The physical propagation axis is a property of the data set, not a global
choice.  The evidence is the magnetostatic structure of a uniformly moving
bunch, ``B = (v/c^2) x E``: the B component *along* the propagation axis is
identically zero and the longitudinal E component is suppressed by ~1/gamma^2.
Measured reference peaks in the reduced data:

* LARGE1 gamma=10: ``Bx == 0`` on both slices, By and Bz nonzero, Ex
  suppressed (Ex/Ey = 0.034).  The bunch moves along **+x**.  LARGE1 gamma=1
  has all B components zero and mean gamma 1: it is **stationary**, with no
  beam axis.
* BDATA2 forward and HIRES: ``By == 0`` and Ey suppressed; the bunch moves
  along **+y**.

Consequently the display layout differs by data set.  ``ARRAY_AXES`` records
the physical axes of ``(array.shape[0], array.shape[1])`` as stored;
``DISPLAY_AXES`` records the correct ``(vertical, horizontal)`` axes for the
figure.  The horizontal axis is the beam axis whenever the beam lies in the
plotted plane; when the beam is out of plane (including a stationary bunch,
which has no beam axis), the horizontal axis is y.  For the +y families the
beam axis y is horizontal.  For LARGE1 the x-y cut puts the axial-x beam axis
x horizontally, while the z-y cut has the beam axis x out of plane and so puts
y horizontally and z vertically, as in the +y families' z-y cuts.

A dataset/plane combination with no verified entry raises
``BlockedAxisError`` instead of guessing.
"""
import numpy as np

# Physical propagation axis of each data set.  gamma=1 LARGE1 is stationary.
PROPAGATION_AXIS = {
    "bdata2": "y",
    "var1b": "y",
    "hires": "y",
    "large1": "x",
    # The multi-velocity samples carry the same +y drift as the val_* bunch;
    # the field cube records beam_axis="+y" (By==0 for the energy-only case).
    "multivel": "y",
}

# Physical axes of (array.shape[0], array.shape[1]) in the stored files.
ARRAY_AXES = {
    ("bdata2", "xy"): ("y", "x"),
    ("bdata2", "zy"): ("y", "z"),
    ("var1b", "xy"): ("y", "x"),
    ("large1", "xy"): ("y", "x"),
    ("large1", "zy"): ("z", "y"),
    ("hires", "xy"): ("y", "x"),
    # Transverse x-z view used only by the bunch-form row (not a stored slice).
    ("bdata2", "xz"): ("z", "x"),
    ("var1b", "zy"): ("y", "z"),
    ("var1b", "xz"): ("z", "x"),
    ("large1", "xz"): ("z", "x"),
    ("multivel", "xy"): ("y", "x"),
    ("multivel", "zy"): ("z", "y"),
    ("multivel", "xz"): ("z", "x"),
}

# Correct figure layout as (vertical, horizontal) per data set and plane.
DISPLAY_AXES = {
    ("bdata2", "xy"): ("x", "y"),   # +y beam horizontal
    ("bdata2", "zy"): ("z", "y"),
    ("var1b", "xy"): ("x", "y"),
    ("hires", "xy"): ("x", "y"),
    ("large1", "xy"): ("y", "x"),   # axial-x beam horizontal
    ("large1", "zy"): ("z", "y"),   # axial-x beam out of plane, y horizontal
    # Transverse x-z view: beam axis (y or x) is out of plane in every family.
    ("bdata2", "xz"): ("z", "x"),
    ("var1b", "zy"): ("z", "y"),
    ("var1b", "xz"): ("z", "x"),
    ("large1", "xz"): ("z", "x"),
    ("multivel", "xy"): ("x", "y"),
    ("multivel", "zy"): ("z", "y"),
    ("multivel", "xz"): ("z", "x"),
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class AxisConventionError(RuntimeError):
    """A slice would be plotted contrary to the recorded convention."""


class BlockedAxisError(AxisConventionError):
    """The stored or display layout is not recorded for this data set."""


def propagation_axis(dataset, gamma=None):
    """Return ``'x'``, ``'y'`` or ``'stationary'`` for this data set/case."""
    if dataset == "large1" and gamma is not None and float(gamma) == 1.0:
        return "stationary"
    if dataset not in PROPAGATION_AXIS:
        raise BlockedAxisError(
            f"no recorded propagation axis for dataset={dataset!r}")
    return PROPAGATION_AXIS[dataset]


def _key(dataset, plane, source=""):
    key = (dataset, plane)
    if key not in ARRAY_AXES or key not in DISPLAY_AXES:
        raise BlockedAxisError(
            f"no verified axis layout for dataset={dataset!r} plane={plane!r}; "
            f"source={source or '<unknown>'}. Refusing to guess the layout.")
    return key


def stored_axes(dataset, plane, source=""):
    return ARRAY_AXES[_key(dataset, plane, source)]


def display_axes(dataset, plane, source=""):
    return DISPLAY_AXES[_key(dataset, plane, source)]


def permutation(dataset, plane, source=""):
    """Return ``np.transpose`` axes mapping stored -> display."""
    stored = stored_axes(dataset, plane, source)
    display = display_axes(dataset, plane, source)
    if set(stored) != set(display):
        raise AxisConventionError(
            f"stored axes {stored} and display axes {display} differ for "
            f"{dataset}/{plane}")
    return tuple(stored.index(ax) for ax in display)


def to_display(array, dataset, plane, source=""):
    """Map a stored slice array to the recorded ``(vertical, horizontal)`` layout."""
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise AxisConventionError(
            f"{dataset}/{plane} slice must be 2-D, got shape {arr.shape}")
    return np.transpose(arr, permutation(dataset, plane, source))


def expected_stored_shape(dataset, plane, n, source=""):
    n = [int(v) for v in n]
    return tuple(n[_AXIS_INDEX[ax]] for ax in stored_axes(dataset, plane, source))


def check_stored_shape(dataset, plane, array, n, source=""):
    """Validate the recorded layout against the shape metadata."""
    expected = expected_stored_shape(dataset, plane, n, source)
    if tuple(np.asarray(array).shape) != expected:
        raise AxisConventionError(
            f"{dataset}/{plane}: stored shape {tuple(np.asarray(array).shape)} "
            f"does not match array_axes={ARRAY_AXES[(dataset, plane)]} and "
            f"n={list(n)} (expected {expected}); source={source or '<unknown>'}")
    return expected


def coords(dataset, plane, n, cell, origin, centres=False, source=""):
    """Return ``(horizontal, vertical)`` coordinate vectors in cell units."""
    _key(dataset, plane, source)
    n = [int(v) for v in n]
    cell = [float(v) for v in cell]
    origin = [float(v) for v in origin]
    vert, horiz = display_axes(dataset, plane, source)

    def axis_coord(name):
        i = _AXIS_INDEX[name]
        offset = 0.5 if centres else 0.0
        count = n[i] if centres else n[i] + 1
        return origin[i] + (np.arange(count) + offset) * cell[i]

    return axis_coord(horiz), axis_coord(vert)


def axis_labels(dataset, plane, source=""):
    """Matplotlib labels ``(horizontal, vertical)`` for a data set and plane."""
    vert, horiz = display_axes(dataset, plane, source)
    return (rf"${horiz}$ [$\mu$m]", rf"${vert}$ [$\mu$m]")


def magnitude_peak(display_array, horiz, vert):
    arr = np.abs(np.asarray(display_array, dtype=float))
    if not np.isfinite(arr).any() or np.nanmax(arr) <= 0:
        return None
    i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
    return float(horiz[j]), float(vert[i])


def centroid(display_array, horiz, vert):
    w = np.abs(np.asarray(display_array, dtype=float)) ** 2
    w = np.nan_to_num(w)
    if w.sum() <= 0:
        return None
    gy = w.sum(axis=0)
    gv = w.sum(axis=1)
    return (float((gy * horiz).sum() / w.sum()),
            float((gv * vert).sum() / w.sum()))


def assert_beam_axis(dataset, plane, display_array, horiz, vert, n, cell,
                     origin, gamma=None, centres=True, source="", figure="",
                     label="", beam_position=None,
                     require_longitudinal_contraction=False, signature=None):
    """Assert the figure uses the recorded physical axis layout.

    Checks:
      1. display shape is ``(len(vert), len(horiz))``;
      2. ``horiz``/``vert`` match the recorded physical axes from metadata;
      3. for a moving data set whose propagation axis lies in the plane, that
         axis is the horizontal one;
      4. for a stationary data set only the recorded labels must match;
      5. the maximum and centroid resolve to finite coordinates in range;
      6. optional: single-particle Lorentz contraction along the beam axis.

    Returns a result dict for the assertion log.
    """
    key = _key(dataset, plane, source)
    vert_name, horiz_name = display_axes(dataset, plane, source)
    prop = propagation_axis(dataset, gamma)
    arr = np.asarray(display_array)
    if arr.shape != (len(vert), len(horiz)):
        raise AxisConventionError(
            f"{figure} {label}: display shape {arr.shape} != "
            f"(len(vert)={len(vert)}, len(horiz)={len(horiz)})")

    h_ref, v_ref = coords(dataset, plane, n, cell, origin, centres=centres,
                          source=source)
    scale = 1.0
    if len(h_ref) == len(horiz) and np.max(np.abs(h_ref)) > 0:
        scale = float(np.max(np.abs(horiz)) / np.max(np.abs(h_ref)))
    if not (np.allclose(horiz, h_ref * scale, rtol=1e-9, atol=1e-9)
            and np.allclose(vert, v_ref * scale, rtol=1e-9, atol=1e-9)):
        raise AxisConventionError(
            f"{figure} {label}: coordinate vectors do not match the recorded "
            f"axes horizontal={horiz_name} vertical={vert_name}")

    peak = magnitude_peak(arr, horiz, vert)
    cen = centroid(arr, horiz, vert)
    in_plane = prop in (horiz_name, vert_name)
    result = {
        "figure": figure, "panel": label, "dataset": dataset, "plane": plane,
        "propagation_axis": prop,
        "stored_axes": list(stored_axes(dataset, plane, source)),
        "display_axes": [vert_name, horiz_name],
        "horizontal_axis": horiz_name, "vertical_axis": vert_name,
        "propagation_axis_in_plane": bool(in_plane),
        "horizontal_range_um": [float(horiz[0]), float(horiz[-1])],
        "vertical_range_um": [float(vert[0]), float(vert[-1])],
        "peak": None, "centroid": None, "beam_position": None,
        "longitudinal_contraction": None, "signature": signature,
        "pass": False,
    }
    ok = True
    if prop != "stationary" and in_plane:
        ok &= (prop == horiz_name)
    if peak is not None:
        result["peak"] = {"horizontal": peak[0], "vertical": peak[1]}
        ok &= bool(horiz[0] <= peak[0] <= horiz[-1])
        ok &= bool(vert[0] <= peak[1] <= vert[-1])
    if cen is not None:
        result["centroid"] = {"horizontal": cen[0], "vertical": cen[1]}
        ok &= bool(horiz[0] <= cen[0] <= horiz[-1])
        ok &= bool(vert[0] <= cen[1] <= vert[-1])
    if beam_position is not None:
        bh, bv = float(beam_position[horiz_name]), float(beam_position[vert_name])
        result["beam_position"] = {"horizontal": bh, "vertical": bv}
        ok &= bool(horiz[0] <= bh <= horiz[-1])
        ok &= bool(vert[0] <= bv <= vert[-1])
        if peak is not None and prop != "stationary" and in_plane:
            tol = max(3.0 * float(horiz[1] - horiz[0]),
                      abs(horiz[-1] - horiz[0]) * 0.25)
            ok &= bool(abs(peak[0] - bh) <= tol)
    if require_longitudinal_contraction and len(horiz) > 2 and len(vert) > 2:
        w = np.nan_to_num(np.abs(arr.astype(float)) ** 2)
        if w.sum() > 0:
            def rms(coord, marginal):
                c = (marginal * coord).sum() / marginal.sum()
                return float(np.sqrt((marginal * (coord - c) ** 2).sum()
                                     / marginal.sum()))
            sh = rms(horiz, w.sum(axis=0))
            sv = rms(vert, w.sum(axis=1))
            result["longitudinal_contraction"] = {
                "horizontal_rms_um": sh, "vertical_rms_um": sv,
                "ratio": sh / sv if sv > 0 else None}
            ok &= bool(sh < sv)
    result["pass"] = bool(ok)
    if not ok:
        raise AxisConventionError(
            f"{figure} {label}: axis assertion failed: {result}")
    return result


def self_test():
    """Check that each recorded display mapping puts the named axes correctly.

    Builds a stored array with a bright band along the *horizontal* physical
    axis and verifies the band is horizontal after ``to_display``.
    """
    results = []
    for dataset, plane in sorted(ARRAY_AXES):
        vert_name, horiz_name = DISPLAY_AXES[(dataset, plane)]
        stored = ARRAY_AXES[(dataset, plane)]
        shape = (5, 7)
        hslot = stored.index(horiz_name)
        a = np.zeros(shape)
        if hslot == 1:
            a[:, 3] = 1.0
        else:
            a[3, :] = 1.0
        disp = to_display(a, dataset, plane)
        col = disp.sum(axis=0)
        row = disp.sum(axis=1)
        ok = (int(np.argmax(col)) == 3 and np.allclose(row, row[0])
              and disp.shape == (shape[stored.index(vert_name)],
                                 shape[hslot]))
        results.append({"dataset": dataset, "plane": plane,
                        "stored_axes": list(stored),
                        "display_axes": [vert_name, horiz_name],
                        "pass": bool(ok)})
        if not ok:
            raise AxisConventionError(
                f"self_test: {dataset}/{plane} did not place {horiz_name} "
                "horizontally")
    return results
