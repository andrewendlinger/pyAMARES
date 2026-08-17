"""Structural tests for the surface xmris consumes.

Run them with::

    pytest tests/test_regression.py tests/test_api_surface.py -o addopts=""

Nothing numeric is asserted here — only shape: column labels, the row index,
``FIDobj`` attribute names, call signatures, import paths, and the copy/pickle
contract the joblib (loky) workers depend on. Renaming anything pinned here breaks
xmris at import or attribute-access time, so it needs a coordinated release.
"""

from __future__ import annotations

import copy
import inspect
import pickle

import pytest

try:  # imported as part of the ``tests`` package
    from tests import regression_cases as rc
except ImportError:  # pragma: no cover - direct invocation from inside tests/
    import regression_cases as rc  # type: ignore[no-redef]


#: The exact column order of ``result_multiplets``, captured from the baseline run.
#: Sixteen labels, not thirteen: xmris reads the thirteen in
#: :data:`XMRIS_CONSUMED_COLUMNS`, and ``g``/``g_sd``/``g (%)`` ride along.
#: ``"CRLB(cs%) "`` carries a **trailing space** — it is emitted that way by
#: ``report_amares`` and xmris matches on the literal string, so the space is
#: load-bearing API, not a typo to tidy up.
RESULT_MULTIPLETS_COLUMNS = [
    "amplitude",
    "sd",
    "CRLB(%)",
    "chem shift(ppm)",
    "sd(ppm)",
    "CRLB(cs%) ",
    "LW(Hz)",
    "sd(Hz)",
    "CRLB(LW%)",
    "phase(deg)",
    "sd(deg)",
    "CRLB(phase%)",
    "g",
    "g_sd",
    "g (%)",
    "SNR",
]

#: The thirteen labels xmris itself reads out of ``result_multiplets``.
XMRIS_CONSUMED_COLUMNS = [
    "amplitude",
    "chem shift(ppm)",
    "LW(Hz)",
    "phase(deg)",
    "SNR",
    "sd",
    "sd(ppm)",
    "sd(Hz)",
    "sd(deg)",
    "CRLB(%)",
    "CRLB(cs%) ",
    "CRLB(LW%)",
    "CRLB(phase%)",
]

#: Namespace attributes xmris touches on a fitted ``FIDobj``. The last four are the
#: ones it ``delattr``s before shipping the object to a loky worker, so they have to
#: exist to be deleted.
FIDOBJ_ATTRIBUTES = [
    "initialParams",
    "peaklist",
    "fittedParams",
    "result_multiplets",
    "styled_df",
    "simple_df",
    "out_obj",
    "fitted_fid",
]

#: Stripped by xmris before pickling to a worker (``styled_df`` holds a pandas
#: Styler whose closures are unpicklable — see the pickle tests below).
XMRIS_STRIPPED_ATTRIBUTES = ["styled_df", "simple_df", "out_obj", "fitted_fid"]


@pytest.fixture(scope="module")
def fitted_example():
    """Case A, fitted once for the whole module."""
    return rc.golden_case_result("example_readme")


@pytest.fixture(scope="module")
def fitted_example_xmris_shape():
    """Case A refitted through xmris' own call shape (``inplace=True``)."""
    return rc.golden_case_result("example_xmris_shape")


# --------------------------------------------------------------------------------
# result_multiplets
# --------------------------------------------------------------------------------


def test_result_multiplets_column_labels_and_order(fitted_example):
    assert list(fitted_example.result_multiplets.columns) == RESULT_MULTIPLETS_COLUMNS


def test_xmris_consumed_columns_are_present(fitted_example):
    """Each of the thirteen labels xmris reads exists, byte for byte."""
    columns = list(fitted_example.result_multiplets.columns)
    missing = [c for c in XMRIS_CONSUMED_COLUMNS if c not in columns]
    assert not missing, (
        f"labels xmris reads are gone from result_multiplets: {missing!r}. "
        f"Present: {columns!r}"
    )


def test_crlb_chemical_shift_label_keeps_its_trailing_space(fitted_example):
    """Guard the one label that looks like a typo and is not."""
    columns = list(fitted_example.result_multiplets.columns)
    assert "CRLB(cs%) " in columns
    assert "CRLB(cs%)" not in columns


def test_result_multiplets_index_is_metabolite_names(fitted_example):
    index = list(fitted_example.result_multiplets.index)
    assert index, "result_multiplets has no rows"
    assert all(isinstance(name, str) for name in index), (
        f"non-string metabolite names in the index: {index!r}"
    )
    assert all(name.strip() for name in index), f"empty metabolite name in {index!r}"
    assert len(set(index)) == len(index), f"duplicate metabolite names in {index!r}"
    assert index == fitted_example.peaklist


def test_xmris_shape_fit_produces_the_same_table_shape(fitted_example_xmris_shape):
    """The in-place call shape xmris uses yields the same columns and index."""
    result = fitted_example_xmris_shape.result_multiplets
    assert list(result.columns) == RESULT_MULTIPLETS_COLUMNS
    assert all(isinstance(name, str) for name in result.index)


# --------------------------------------------------------------------------------
# FIDobj attributes
# --------------------------------------------------------------------------------


def test_fidobj_has_the_attributes_xmris_touches(fitted_example):
    missing = [a for a in FIDOBJ_ATTRIBUTES if not hasattr(fitted_example, a)]
    assert not missing, f"FIDobj lost attribute(s) {missing!r}"


def test_inplace_fit_omits_fittedparams_and_out_obj(fitted_example_xmris_shape):
    """Pin the one asymmetry between the two call shapes.

    ``fitAMARES`` only assigns ``fid_parameters.out_obj`` and
    ``fid_parameters.fittedParams`` on the ``inplace=False`` branch — with
    ``inplace=True`` it returns the ``MinimizerResult`` instead. xmris fits in place
    and guards its cleanup with ``hasattr``, so the absence is fine; it is pinned
    here so a future change that starts (or stops) setting them is visible.
    """
    present = [a for a in FIDOBJ_ATTRIBUTES if hasattr(fitted_example_xmris_shape, a)]
    assert sorted(present) == sorted(
        [a for a in FIDOBJ_ATTRIBUTES if a not in ("fittedParams", "out_obj")]
    )


# --------------------------------------------------------------------------------
# Signatures and import paths
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "priorknowledgefile",
        "MHz",
        "sw",
        "deadtime",
        "ppm_offset",
        "g_global",
        "normalize_fid",
        "preview",
    ],
)
def test_initialize_fid_accepts(name):
    import pyAMARES

    assert name in inspect.signature(pyAMARES.initialize_FID).parameters


@pytest.mark.parametrize(
    "name",
    [
        "fid_parameters",
        "fitting_parameters",
        "method",
        "initialize_with_lm",
        "ifplot",
        "inplace",
    ],
)
def test_fitamares_accepts(name):
    from pyAMARES.kernel.lmfit import fitAMARES

    assert name in inspect.signature(fitAMARES).parameters


def test_toplevel_imports():
    from pyAMARES import (  # noqa: F401
        initialize_FID,
        multieq6,
        result_pd_to_params,
        uninterleave,
    )


def test_submodule_imports():
    import pyAMARES.libs.logger
    from pyAMARES.kernel.lmfit import fitAMARES  # noqa: F401
    from pyAMARES.libs.logger import set_log_level  # noqa: F401

    assert isinstance(pyAMARES.libs.logger.DEFAULT_LOG_LEVEL, str)


# --------------------------------------------------------------------------------
# The loky-worker contract: deepcopy in, pickle across
# --------------------------------------------------------------------------------


def test_fitted_object_deepcopies(fitted_example):
    """xmris deepcopies the shared FIDobj once per voxel."""
    clone = copy.deepcopy(fitted_example)
    assert list(clone.result_multiplets.columns) == RESULT_MULTIPLETS_COLUMNS
    assert clone.result_multiplets.equals(fitted_example.result_multiplets)


def test_stripped_fitted_object_pickles(fitted_example):
    """The exact shape xmris ships to a loky worker round-trips through pickle.

    The *full* fitted object does not pickle — ``styled_df`` is a pandas Styler
    holding a local lambda from ``Styler.apply``. That is precisely why xmris
    ``delattr``s the four heavy attributes first; this test mirrors that and pins
    the round trip it depends on.
    """
    stripped = copy.deepcopy(fitted_example)
    for attribute in XMRIS_STRIPPED_ATTRIBUTES:
        if hasattr(stripped, attribute):
            delattr(stripped, attribute)

    revived = pickle.loads(pickle.dumps(stripped))
    assert revived.result_multiplets.equals(fitted_example.result_multiplets)
    assert revived.peaklist == fitted_example.peaklist
    assert list(revived.initialParams) == list(fitted_example.initialParams)


def test_unstripped_fitted_object_does_not_pickle(fitted_example):
    """Document the reason the strip exists, so nobody removes it as dead code."""
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        pickle.dumps(copy.deepcopy(fitted_example))
