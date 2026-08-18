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
from typing import Callable, NamedTuple

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


#: Per-case tolerance blocks and comments live on the registry entry in
#: ``regression_cases`` (``GoldenCase.tolerances`` / ``.comment``), not here: they
#: describe one dataset's measured drift, so baking them into a payload builder
#: would stamp them onto every future case of the same kind.


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


def _tolerances(case, default: dict) -> dict:
    """The case's own tolerance block if it declares one, else the kind's default.

    ``None`` is the "use the default" signal, per :class:`GoldenCase`'s contract —
    tested with ``is not None`` rather than for truthiness, so a case that
    deliberately declares an empty block gets the empty block it asked for instead
    of silently inheriting the default.

    Deep-copied through JSON either way, so a payload can never alias — and later
    mutate — the live constant it was built from.
    """
    block = case.tolerances if case.tolerances is not None else default
    return json.loads(json.dumps(block))


def golden_payload(case_name: str, result_multiplets, case) -> dict:
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
    payload = {
        "meta": meta,
        "index": [str(name) for name in df.index],
        "columns_exact_order": [str(c) for c in df.columns],
        "golden_columns": list(rc.GOLDEN_COLUMNS),
        "structural_columns": list(rc.STRUCTURAL_COLUMNS),
        "values": values,
        "tolerances": _tolerances(case, DEFAULT_TOLERANCES),
    }
    # A fit case may carry a comment too — the field is on GoldenCase, not on one
    # kind's builder, so honouring it here is what makes it mean the same thing
    # everywhere. Written only when there is one, so the three canonical fit
    # goldens (which declare none) keep exactly the keys they were frozen with.
    if case.comment:
        payload["comment"] = case.comment
    return payload


def hsvd_golden_payload(case_name: str, result: dict, case) -> dict:
    """Turn an HSVD-decomposition runner's result into a payload.

    A different shape from :func:`golden_payload` on purpose: this kind freezes a
    decomposition (rows of components plus the singular-value head), not a
    ``result_multiplets`` table, so it carries ``components`` /
    ``top_singular_values`` rather than ``values`` / ``index`` / column lists.

    Every number that describes *this dataset* — the tolerance floors, the
    narrative — comes off the registry entry, so the builder stays generic over
    the kind. Both are read straight off the case with no fallback: there is no
    default tolerance block that would mean anything for a decomposition, and no
    default narrative either. ``PayloadHandler.requires`` is what enforces their
    presence, before the case is ever run.
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
        # Built from the declared field list, not from the DataFrame's columns: a
        # column the runner grows later would otherwise be frozen here and never
        # compared, since the test iterates HSVD_COMPONENT_FIELDS.
        "components": [
            {
                field: float(components.at[row, field])
                for field in rc.HSVD_COMPONENT_FIELDS
            }
            for row in components.index
        ],
        "top_singular_values": [float(x) for x in result["top_singular_values"]],
        # Deep-copied so the payload cannot alias the live registry constant; no
        # default to fall back to, hence no _tolerances() call.
        "tolerances": json.loads(json.dumps(case.tolerances)),
        "comment": case.comment,
    }


def _result_multiplets_payload(case_name: str, fidobj, case) -> dict:
    """Adapter: a fit runner returns the FID object, not the table."""
    return golden_payload(case_name, fidobj.result_multiplets, case)


class PayloadHandler(NamedTuple):
    """Everything ``capture_goldens.py`` knows about one payload kind."""

    #: ``(case_name, runner_result, GoldenCase) -> payload dict``.
    build: Callable
    #: ``payload -> "8 components"``; how the capture log describes what it wrote.
    #: Declared per kind rather than sniffed off the payload's keys — reading the
    #: shape back out of the thing you just built is a guess, and it silently
    #: mislabels the first kind that happens to share a key name.
    describe: Callable
    #: :class:`regression_cases.GoldenCase` fields this kind cannot be captured
    #: without. Checked in :func:`capture_case` *before* the case is run, so a
    #: registry entry that is missing one costs a clear error rather than a long
    #: fit followed by a file the test suite then rejects.
    requires: tuple = ()


#: Payload kind -> handler, the one thing ``capture_goldens.py`` adds to the case
#: registry in ``regression_cases``. Looked up by name so a case declaring a kind
#: nobody can serialise fails loudly at capture time instead of silently.
PAYLOAD_HANDLERS = {
    rc.KIND_RESULT_MULTIPLETS: PayloadHandler(
        _result_multiplets_payload, lambda payload: f"{len(payload['index'])} rows"
    ),
    # Both are hard requirements at *test* time — `_hsvd_golden_problems` rejects a
    # golden of this kind that carries no comment, and the comparer needs a
    # tolerance block — so they are hard requirements at capture time too. They
    # were not, and capture could happily produce a file the suite would fail on.
    rc.KIND_HSVD_COMPONENTS: PayloadHandler(
        hsvd_golden_payload,
        lambda payload: f"{len(payload['components'])} components",
        requires=("tolerances", "comment"),
    ),
}


def capture_case(case_name: str):
    """Run one case; return ``(payload, one-line description)``."""
    case = rc.GOLDEN_CASE_REGISTRY[case_name]
    try:
        handler = PAYLOAD_HANDLERS[case.kind]
    except KeyError:
        raise RuntimeError(
            f"{case_name}: payload kind {case.kind!r} has no handler in "
            f"capture_goldens.PAYLOAD_HANDLERS (known: {sorted(PAYLOAD_HANDLERS)})"
        ) from None
    absent = [field for field in handler.requires if not getattr(case, field)]
    if absent:
        raise RuntimeError(
            f"{case_name}: a {case.kind!r} case must declare {absent} on its "
            "registry entry in regression_cases.GOLDEN_CASE_REGISTRY — there is no "
            "default for it that would mean anything for this payload kind."
        )
    payload = handler.build(case_name, case.runner(), case)
    return payload, handler.describe(payload)


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
    case_names = [n for n in rc.CASE_ORDER if not args.only or n in args.only]
    if not case_names:
        # Unreachable while --only is validated against the same registry the
        # order comes from. Kept as the belt to that braces: a selection that
        # writes nothing must never look like a successful capture.
        parser.error(
            f"no cases selected; --only {args.only!r} matched none of {rc.CASE_ORDER}"
        )

    outdir = os.path.abspath(args.write)

    # A platform golden set is only meaningful if the numbers in it came from
    # that platform. Refuse to fill someone else's directory — an arm64 laptop
    # writing into linux-x86_64/ would freeze the wrong BLAS's output under a
    # name the test suite trusts on ubuntu. Writing the canonical set (a
    # directory whose name is not platform-shaped, i.e. tests/goldens/ itself, or
    # any scratch path) stays allowed: that set is deliberately one platform's
    # numbers serving every platform.
    basename = os.path.basename(outdir)
    if rc.looks_like_platform_dir(basename) and basename != rc.platform_goldens_key():
        parser.error(
            f"--write {args.write!r} targets the platform golden set {basename!r}, "
            f"but this host is {rc.platform_goldens_key()!r}. Capture a platform "
            "set on its own platform (see .github/workflows/capture-goldens.yml), "
            "or write to a directory that is not named after a platform. A name "
            "counts as a platform set when its first component is one of "
            f"regression_cases.PLATFORM_PREFIXES "
            f"({', '.join(sorted(rc.PLATFORM_PREFIXES))}) — if a real platform is "
            "missing from that set, add it there rather than working around this."
        )

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
        payload, description = capture_case(name)
        write_golden(payload, targets[name])
        print(f"wrote {targets[name]}  ({description})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
