"""Structural tests for the surface xmris consumes.

Run them with::

    pytest tests/test_regression.py tests/test_api_surface.py -o addopts=""

Nothing numeric is asserted here — only shape: column labels, the row index,
``FIDobj`` attribute names, call signatures, import paths, the copy/pickle
contract the joblib (loky) workers depend on, and the import graph of a bare
``import pyAMARES``. Renaming anything pinned here breaks xmris at import or
attribute-access time, so it needs a coordinated release.
"""

from __future__ import annotations

import ast
import copy
import inspect
import os
import pickle
import subprocess
import sys

import pytest

try:  # imported as part of the ``tests`` package
    from tests import regression_cases as rc
except ImportError:  # pragma: no cover - direct invocation from inside tests/
    import regression_cases as rc  # type: ignore[no-redef]


#: Single source of truth in ``regression_cases`` (the goldens' ``columns_exact_order``
#: is checked against the same constant): sixteen labels, of which xmris reads the
#: thirteen in :data:`XMRIS_CONSUMED_COLUMNS`.
RESULT_MULTIPLETS_COLUMNS = rc.RESULT_MULTIPLETS_COLUMNS

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
    # The name check above cannot stand on its own. Since 0.4.0 the order is
    # enforced by pyAMARES/util/report.py binding ``result = result.reindex(
    # fid_parameters.peaklist)`` when the two label sets agree -- and reindex
    # produces exactly those names whether or not any fitted value survives it.
    # So assert on the values too: a reindex against a divergent label set fills
    # the table with NaN while leaving this index perfect.
    values = fitted_example.result_multiplets
    for column in ("amplitude", "chem shift(ppm)", "LW(Hz)", "phase(deg)", "SNR"):
        assert values[column].notna().all(), (
            f"{column!r} has NaN rows, so result_multiplets was reindexed against "
            f"a peak set the fit does not cover:\n{values[column]}"
        )


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


# --------------------------------------------------------------------------------
# The import graph: bare `import pyAMARES` stays lazy (D17)
# --------------------------------------------------------------------------------

#: Modules that must NOT load on bare `import pyAMARES`. Matched as the exact
#: name or any submodule of it.
#: Deliberate absences:
#: - `matplotlib` (bare): lmfit.model does a module-scope `try: import
#:   matplotlib` (for _HAS_MATPLOTLIB), so the bare package always loads; the
#:   expensive half is matplotlib.pyplot, and that is what is pinned.
#: - `hlsvdpro`: under numpy<2 with an importable hlsvdpro, util/hsvd.py
#:   legitimately binds it at import time (rule pinned by
#:   test_hsvd_backend_selection).
#: - `jinja2`: util/report.py probes it at import time by design (core dep).
FORBIDDEN_ON_BARE_IMPORT = [
    "matplotlib.pyplot",
    "nmrglue",
    "mat73",
    "sympy",
    "IPython",
    "tqdm",
    "requests",
    "xlrd",
    "openpyxl",
    "nibabel",
]


def test_bare_import_keeps_heavy_modules_unloaded():
    """Import pyAMARES in a child process and see what came along with it.

    The child is handed this process' ``sys.path`` verbatim, so it resolves the
    same pyAMARES pytest resolved rather than whatever the working directory
    happens to offer. It prints one offending module name per line, so a failure
    shows the names — and any stray import-time chatter — as evidence rather than
    as a parse error.
    """
    code = (
        "import sys\n"
        "sys.path = " + repr(list(sys.path)) + "\n"
        "import pyAMARES\n"
        "names = " + repr(FORBIDDEN_ON_BARE_IMPORT) + "\n"
        "for n in names:\n"
        "    if n in sys.modules or any(m.startswith(n + '.') for m in sys.modules):\n"
        "        print(n)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, "bare import failed:\n" + proc.stderr
    assert not proc.stdout.strip(), (
        "bare `import pyAMARES` loaded modules it must not (see DIVERGENCE.md "
        "D17) — a module-level heavy import crept back in. Child stdout:\n"
        + proc.stdout
    )


#: Third-party distributions ``pyAMARES/**`` may import at module scope. Anything
#: else belongs inside the function that uses it (D17). numpy/scipy/pandas/lmfit
#: are the numeric core the package cannot work without; jinja2 is probed at
#: import time by util/report.py to set ``if_style``.
ALLOWED_MODULE_LEVEL = frozenset({"numpy", "scipy", "pandas", "lmfit", "jinja2"})

#: Documented per-file exceptions, keyed by path relative to the package root:
#: - ``util/hsvd.py``: *which* HSVD backend the module binds is an import-time
#:   rule, and that rule is what ``test_hsvd_backend_selection`` pins.
#: - ``util/crlb.py``: sympy is the CRLB algebra itself (``create_pmatrix``), and
#:   the module is reached only through util/report.py's delayed import — i.e. at
#:   first-report time, never at ``import pyAMARES``.
MODULE_LEVEL_EXCEPTIONS = {
    os.path.join("util", "hsvd.py"): frozenset({"hlsvdpro"}),
    os.path.join("util", "crlb.py"): frozenset({"sympy"}),
}

#: ``pyAMARES/script/`` is out of scope: nothing in the package imports it (the
#: entry points are console scripts), and amaresfit_gui.py carries streamlit,
#: requests and matplotlib at module scope by design.
UNSCANNED_SUBPACKAGES = ("script",)

try:  # Python 3.10+
    STDLIB_MODULE_NAMES = frozenset(sys.stdlib_module_names)
except AttributeError:  # pragma: no cover - Python 3.8/3.9
    STDLIB_MODULE_NAMES = frozenset(
        {
            "__future__", "abc", "argparse", "base64", "collections", "concurrent",
            "contextlib", "copy", "csv", "datetime", "functools", "glob", "hashlib",
            "importlib", "inspect", "io", "itertools", "json", "logging", "math",
            "multiprocessing", "os", "pathlib", "pickle", "random", "re", "shutil",
            "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
            "threading", "time", "traceback", "typing", "uuid", "warnings",
        }
    )  # fmt: skip


def _module_level_imports(tree):
    """Yield ``(top_level_name, lineno)`` for every absolute import at module scope.

    Module scope includes the bodies of module-level ``if``/``try`` blocks — that
    is where both documented exceptions live — but never a function or class body,
    which is exactly where D17 put the heavy imports.
    """
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # level > 0 is a relative import
                yield node.module.split(".")[0], node.lineno
        else:
            pending.extend(ast.iter_child_nodes(node))


def test_no_undocumented_module_level_third_party_imports():
    """The forward-looking half of the D17 guard.

    :func:`test_bare_import_keeps_heavy_modules_unloaded` checks a fixed blocklist
    and gives the better failure message; this one is open-ended, so a cherry-pick
    from upstream that adds a module-level import of something nobody has thought
    of yet fails here instead of shipping.
    """
    import pyAMARES

    package_root = os.path.dirname(os.path.abspath(pyAMARES.__file__))
    offenders = []
    for directory, subdirectories, filenames in os.walk(package_root):
        subdirectories[:] = [
            d
            for d in subdirectories
            if d not in UNSCANNED_SUBPACKAGES and d != "__pycache__"
        ]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, package_root)
            allowed = ALLOWED_MODULE_LEVEL | MODULE_LEVEL_EXCEPTIONS.get(
                relative, frozenset()
            )
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for name, lineno in _module_level_imports(tree):
                if name in STDLIB_MODULE_NAMES or name == "pyAMARES":
                    continue
                if name not in allowed:
                    offenders.append("{}:{}: {}".format(relative, lineno, name))

    assert not offenders, (
        "module-level third-party imports that D17 does not allow:\n  "
        + "\n  ".join(sorted(offenders))
        + "\nImport them inside the function that uses them, or — if the binding "
        "genuinely has to happen at import time — add the file to "
        "MODULE_LEVEL_EXCEPTIONS with a reason and a DIVERGENCE.md entry."
    )
