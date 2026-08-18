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

    unknown = sorted(set(args.only or []) - set(rc.GOLDEN_CASES))
    if unknown:
        parser.error(
            f"unknown case(s) {unknown!r}; known cases: {sorted(rc.GOLDEN_CASES)}"
        )
    case_names = [n for n in rc.GOLDEN_CASES if not args.only or n in args.only]

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
        fidobj = rc.GOLDEN_CASES[name]()
        payload = golden_payload(name, fidobj.result_multiplets)
        write_golden(payload, targets[name])
        print(f"wrote {targets[name]}  ({len(payload['index'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
