#!/usr/bin/env python
"""Capture the numeric regression goldens for pyAMARES.

Goldens are frozen on one deliberate dependency stack (see ``CLAUDE.md``) and are
*not* regenerated casually — a diff here means the fitted numbers moved, which is
exactly what ``tests/test_regression.py`` exists to catch.

Usage
-----
::

    python tests/capture_goldens.py --write tests/goldens/
    python tests/capture_goldens.py --write tests/goldens/ --force   # overwrite
    python tests/capture_goldens.py --write /tmp/check/ --only example_readme

Existing files are never overwritten without ``--force``.

``tests/goldens/`` itself holds the canonical set, captured on darwin-arm64. A
``tests/goldens/<sys.platform>-<machine>/`` subdirectory (see
``regression_cases.platform_goldens_key``) overrides the canonical files per file on
that platform; ``.github/workflows/capture-goldens.yml`` produces such a set for
linux-x86_64 as a reviewable artifact.

The baseline stack the committed goldens were captured on::

    uv run --no-project --python 3.12 --with 'numpy==1.26.4' --with 'pandas==2.1.4' \
      --with 'nmrglue==0.11' --with . python tests/capture_goldens.py --write tests/goldens/
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys

import numpy as np

try:  # imported as part of the ``tests`` package (pytest, ``python -m``)
    from tests import regression_cases as rc
except ImportError:  # run directly: ``python tests/capture_goldens.py``
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import regression_cases as rc  # type: ignore[no-redef]

#: Written into every golden. Loosening a tolerance later is a data-only edit inside
#: the JSON (with a comment saying why) — no test code changes.
#:
#: These are **not** bit-for-bit, and cannot be. ``pyAMARES.kernel.fid.equation6``
#: multiplies freshly allocated complex128 arrays, and numpy's SIMD loop for that
#: multiply peels its leading elements according to the array's memory alignment —
#: which varies from call to call. Identical parameters therefore produce residuals
#: that differ in the last one or two ulp, and ``leastsq``/``least_squares`` amplify
#: that into the fitted values. It happens twice in a row inside one process, so it
#: is not an ASLR or hash-seed artefact, and it survives pinning every BLAS thread
#: count to 1 (the multiply is a numpy ufunc, not a BLAS call). On Apple Silicon the
#: responsible SIMD feature is part of numpy's non-disableable baseline (NEON), so
#: there is no knob to turn it off either.
#:
#: The values below come from 8 independent captures on the baseline stack. Worst
#: observed run-to-run relative spread was 1.6e-5 (``CRLB(cs%) `` / ``chem
#: shift(ppm)`` for a peak sitting at ~0 ppm) and 8.5e-6 everywhere else, so the
#: default leaves a ~10x margin — tight enough that a dependency bump moving a
#: fitted amplitude by 0.05% still fails the suite.
DEFAULT_TOLERANCES = {
    "default_rtol": 1e-4,
    "default_atol": 0.0,
    "per_column": {
        # PCr sits at ~0 ppm and the synthetic Alpha at ~-6e-6 ppm, so a relative
        # tolerance is meaningless there. The absolute floor is a nano-ppm — 150x
        # the observed absolute spread and still physically nothing.
        "chem shift(ppm)": {"atol": 1e-6},
        # CRLB as a percentage *of* that near-zero chemical shift, so it inherits
        # the amplification the atol above absorbs for the shift itself.
        "CRLB(cs%) ": {"rtol": 1e-3},
    },
}


#: Tolerances for the vendored-HSVD-backend golden. Sized from measurement, not
#: guessed — :data:`HSVD_COMMENT`, written into the golden itself, records the
#: configurations that were compared and the drift each one showed.
#:
#: Much tighter than :data:`DEFAULT_TOLERANCES` because this case is a single
#: linear-algebra pass — one SVD, one least-squares solve, one eigendecomposition,
#: one ``zgelss`` — with no iterative optimizer to amplify a last-ulp difference
#: into a visible one. The fit goldens go through ``leastsq``/``least_squares``,
#: which is exactly why they need 1e-4 (and 5e-3 on the real-data case) where this
#: one holds 1e-9 across every stack and platform measured.
HSVD_TOLERANCES = {
    "default_rtol": 1e-9,
    "default_atol": 0.0,
    "per_field": {
        # One component sits at -0.0135 Hz, so a relative tolerance says nothing
        # about it; the floor is what actually guards that cell.
        "frequency_hz": {"atol": 1e-9},
        # Same story for that component's phase, which is 0.0989 degrees.
        "phase_deg": {"atol": 1e-8},
    },
}

#: Written into the vendored-HSVD golden as its ``comment``.
HSVD_COMMENT = (
    "Tolerances are measured, not guessed. The runner was compared across six "
    "configurations on 2026-08-18: darwin-arm64 py3.12/numpy 1.26.4/scipy 1.17.1 "
    "(the capture stack, reference); darwin-arm64 py3.13/numpy 2.5.2/scipy 1.18.0; "
    "linux-aarch64 and linux-x86_64 py3.12/numpy 1.26.4/scipy 1.17.1 in containers; "
    "linux-x86_64 py3.13/numpy 2.5.2/scipy 1.18.0 in a container; and a real GitHub "
    "ubuntu-latest x86_64 runner on py3.12/numpy 1.26.4/scipy 1.17.1 via "
    "capture-goldens.yml. nsv_found was 8 in every one. Worst relative deviation "
    "over every frozen value and every configuration: 1.1e-10 — and that maximum "
    "belongs entirely to the two near-zero cells (the component at -0.0135 Hz and "
    "its 0.0989 deg phase). The worst on any cell where a relative comparison "
    "means something is 5.4e-13 (damping), 3.1e-13 (amplitude), 2.6e-13 (phase "
    "away from zero), 3.7e-15 (frequency away from zero) and 1.9e-15 (singular "
    "values). default_rtol 1e-9 "
    "is therefore ~10x the worst measured drift overall and ~1800x the worst "
    "meaningful one. The two atol floors absorb the near-zero cells: frequency_hz "
    "1e-9 Hz is ~440x the worst measured absolute frequency drift (2.3e-12 Hz) and "
    "still 1e-17 ppm at 120 MHz; phase_deg 1e-8 deg is ~870x the worst measured "
    "absolute phase drift (1.1e-11 deg). damping needs no floor — no component "
    "comes near zero (|damping| runs 5.8e-3 to 2.0e-2, worst drift 1800x inside "
    "the default). No phase comes near +-180 deg either (max |phase| 149.4 deg), "
    "so the comparison is plain rather than angular; an input that put a component "
    "on the wrap point would need an angular comparison this golden does not "
    "implement. nmrglue is recorded in meta but is not on this case's code path at "
    "all — the vendored backend is pure scipy — so its version cannot matter here."
)


def capture_meta() -> dict:
    """Record the stack the goldens were captured on."""
    import lmfit
    import nmrglue
    import pandas
    import scipy

    import pyAMARES

    return {
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "pyamares": pyAMARES.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "lmfit": lmfit.__version__,
        "nmrglue": nmrglue.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def golden_payload(case_name: str, result_multiplets) -> dict:
    """Turn one ``result_multiplets`` table into the golden JSON payload."""
    df = result_multiplets
    missing = [c for c in rc.GOLDEN_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{case_name}: result_multiplets lost columns {missing!r}")
    values = {
        col: {str(name): float(df.at[name, col]) for name in df.index}
        for col in rc.GOLDEN_COLUMNS
    }
    meta = capture_meta()
    meta["case"] = case_name
    return {
        "meta": meta,
        "index": [str(name) for name in df.index],
        "columns_exact_order": [str(c) for c in df.columns],
        "golden_columns": list(rc.GOLDEN_COLUMNS),
        "structural_columns": list(rc.STRUCTURAL_COLUMNS),
        "values": values,
        "tolerances": json.loads(json.dumps(DEFAULT_TOLERANCES)),
    }


def hsvd_golden_payload(case_name: str, result: dict) -> dict:
    """Turn :func:`regression_cases.run_hsvd_vendored_backend_case` into a payload.

    A different shape from :func:`golden_payload` on purpose: this case freezes a
    decomposition (rows of components plus the singular-value head), not a
    ``result_multiplets`` table, so it carries ``components`` /
    ``top_singular_values`` rather than ``values`` / ``index`` / column lists.
    """
    components = result["components"]
    missing = [f for f in rc.HSVD_COMPONENT_FIELDS if f not in components.columns]
    if missing:
        raise RuntimeError(f"{case_name}: HSVD components lost fields {missing!r}")
    meta = capture_meta()
    meta["case"] = case_name
    return {
        "meta": meta,
        "nsv_found": int(result["nsv_found"]),
        "component_fields": list(rc.HSVD_COMPONENT_FIELDS),
        "components": [
            {field: float(components.at[row, field]) for field in components.columns}
            for row in components.index
        ],
        "top_singular_values": [float(x) for x in result["top_singular_values"]],
        "tolerances": json.loads(json.dumps(HSVD_TOLERANCES)),
        "comment": HSVD_COMMENT,
    }


#: Every case name in a deterministic capture order.
CASE_ORDER = list(rc.GOLDEN_CASES) + [rc.HSVD_VENDORED_CASE]


def capture_case(case_name: str) -> dict:
    """Run one case and return its golden payload, whatever its payload shape."""
    if case_name == rc.HSVD_VENDORED_CASE:
        return hsvd_golden_payload(case_name, rc.run_hsvd_vendored_backend_case())
    return golden_payload(case_name, rc.GOLDEN_CASES[case_name]().result_multiplets)


def payload_size(payload: dict) -> str:
    """A one-line 'what got written' description for the capture log."""
    if "components" in payload:
        return f"{len(payload['components'])} components"
    return f"{len(payload['index'])} rows"


def write_golden(payload: dict, path: str) -> None:
    """Write one golden JSON at full float precision.

    ``json`` emits Python's ``repr`` for floats, which round-trips exactly, and
    ``NaN`` for not-a-number (non-standard JSON, but ``json.load`` reads it back
    unchanged and the loader is only ever this repo's test suite).
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        required=True,
        metavar="DIR",
        help="Directory to write the golden JSON files into (created if absent).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite goldens that already exist. Off by default, on purpose.",
    )
    parser.add_argument(
        "--only",
        metavar="CASE",
        action="append",
        help="Capture only this case (repeatable). Default: every case.",
    )
    args = parser.parse_args(argv)

    unknown = sorted(set(args.only or []) - rc.ALL_GOLDEN_NAMES)
    if unknown:
        parser.error(
            f"unknown case(s) {unknown!r}; known cases: {sorted(rc.ALL_GOLDEN_NAMES)}"
        )
    case_names = [n for n in CASE_ORDER if not args.only or n in args.only]

    outdir = os.path.abspath(args.write)
    os.makedirs(outdir, exist_ok=True)

    targets = {name: os.path.join(outdir, f"{name}.json") for name in case_names}
    existing = [p for p in targets.values() if os.path.exists(p)]
    if existing and not args.force:
        print(
            "Refusing to overwrite existing goldens (pass --force if you really "
            "mean to re-freeze them):",
            file=sys.stderr,
        )
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 1

    rc.quiet()
    for name in case_names:
        payload = capture_case(name)
        write_golden(payload, targets[name])
        print(f"wrote {targets[name]}  ({payload_size(payload)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
