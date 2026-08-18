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


def test_headless_fft_params_paths_import_no_nmrglue():
    """Pin *where* the nmrglue import sits inside ``fft_params`` (D17).

    It is deliberately below the ``return_mat`` / ``fid=True`` early returns,
    because those are the two branches a headless fit takes — kernel/lmfit.py and
    kernel/PriorKnowledge.py only ever call it that way. A comment says so; this
    test is what enforces it, in a child process so a stray nmrglue loaded by some
    other test cannot mask the regression.
    """
    code = (
        "import sys\n"
        "sys.path = " + repr(list(sys.path)) + "\n"
        "import numpy as np\n"
        "from lmfit import Parameters\n"
        "from pyAMARES.kernel.fid import fft_params\n"
        "params = Parameters()\n"
        "for name, value in (('ak_1', 1.0), ('freq_1', 10.0), ('dk_1', 5.0),\n"
        "                    ('phi_1', 0.0), ('g_1', 0.0)):\n"
        "    params.add(name, value=value)\n"
        "timeaxis = np.arange(16) / 1000.0\n"
        "assert fft_params(timeaxis, params, fid=True) is not None\n"
        "assert fft_params(timeaxis, params, return_mat=True) is not None\n"
        "print('nmrglue' in sys.modules)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, "headless fft_params call failed:\n" + proc.stderr
    assert proc.stdout.strip() == "False", (
        "fft_params' fid=True / return_mat=True branches imported nmrglue — the "
        "import moved above the early returns, putting a plotting-era dependency "
        "back on the critical path of every headless fit (DIVERGENCE.md D17). "
        "Child stdout:\n" + proc.stdout
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

#: Top-level subpackages of ``pyAMARES/`` the scan skips, matched against the
#: first path component only — a ``script`` directory nested anywhere else is
#: still scanned. ``pyAMARES/script/`` is out of scope because nothing in the
#: package imports it (the entry points are console scripts), and
#: amaresfit_gui.py carries streamlit, requests and matplotlib at module scope by
#: design.
UNSCANNED_SUBPACKAGES = ("script",)


def _module_level_imports(tree):
    """Yield ``(top_level_name, lineno)`` for every absolute import at module scope.

    Module scope is everything that runs on import: the module body, the bodies of
    module-level ``if``/``try`` blocks — where both documented exceptions live —
    and class bodies, which execute at class-creation time, i.e. at import. Only
    function bodies are skipped, which is exactly where D17 put the heavy imports.
    """
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # level > 0 is a relative import
                yield node.module.split(".")[0], node.lineno
        else:
            pending.extend(ast.iter_child_nodes(node))


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason=(
        "needs sys.stdlib_module_names (3.10+); the scan reads source, not the "
        "running interpreter, so the 3.10+ legs cover every supported Python"
    ),
)
def test_no_undocumented_module_level_third_party_imports():
    """The forward-looking half of the D17 guard.

    :func:`test_bare_import_keeps_heavy_modules_unloaded` checks a fixed blocklist
    and gives the better failure message; this one is open-ended, so a cherry-pick
    from upstream that adds a module-level import of something nobody has thought
    of yet fails here instead of shipping.
    """
    import pyAMARES

    stdlib_module_names = frozenset(sys.stdlib_module_names)
    package_root = os.path.dirname(os.path.abspath(pyAMARES.__file__))
    offenders = []
    for directory, subdirectories, filenames in os.walk(package_root):
        # Prune by position, not by name: only `pyAMARES/script/` is exempt, not
        # any directory that happens to be called "script".
        depth = os.path.relpath(directory, package_root)
        subdirectories[:] = [
            d
            for d in subdirectories
            if d != "__pycache__"
            and not (depth == os.curdir and d in UNSCANNED_SUBPACKAGES)
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
                if name in stdlib_module_names or name == "pyAMARES":
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


# --------------------------------------------------------------------------------
# Optional dependencies: the message says which extra to install (D18)
# --------------------------------------------------------------------------------
#
# mat73, openpyxl and xlrd left install_requires in 0.5.0. Each import site that
# lost its guaranteed dependency has to fail with a message naming the extra that
# restores it. These tests simulate the missing dependency rather than requiring
# it to be absent, so they assert the same thing on every stack — bare install or
# ``[jupyter]``.


def test_v73_mat_read_without_mat73_names_the_matlab_extra(monkeypatch, tmp_path):
    """``readmrs`` on a v7.3 .mat, with mat73 unimportable."""
    from pyAMARES.fileio import readmat

    # ``None`` in sys.modules makes the import machinery raise ImportError, which
    # is what a genuinely absent mat73 does. monkeypatch removes the key again.
    monkeypatch.setitem(sys.modules, "mat73", None)
    monkeypatch.setattr(readmat, "is_mat_file_v7_3", lambda filename: True)

    with pytest.raises(ImportError) as excinfo:
        readmat.readmrs(str(tmp_path / "v73.mat"))

    assert "pyamares-xmris[matlab]" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_read_fidall_without_mat73_names_the_matlab_extra(monkeypatch, tmp_path):
    """The second v7.3 branch, in ``read_fidall``."""
    from pyAMARES.fileio import readfidall

    monkeypatch.setitem(sys.modules, "mat73", None)
    monkeypatch.setattr(readfidall, "is_mat_file_v7_3", lambda filename: True)

    with pytest.raises(ImportError) as excinfo:
        readfidall.read_fidall(str(tmp_path / "v73.mat"))

    assert "pyamares-xmris[matlab]" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_broken_mat73_keeps_its_own_error(monkeypatch, tmp_path):
    """An installed-but-broken mat73 must not be reported as a missing extra.

    The guard catches ModuleNotFoundError only, so an h5py ABI mismatch — an
    ImportError that is *not* a ModuleNotFoundError — reaches the caller with the
    message that says what is actually wrong.
    """
    import builtins

    from pyAMARES.fileio import readmat

    real_import = builtins.__import__

    def broken_mat73_import(name, *args, **kwargs):
        if name == "mat73":
            raise ImportError("libhdf5.so.310: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_mat73_import)
    monkeypatch.setattr(readmat, "is_mat_file_v7_3", lambda filename: True)

    with pytest.raises(ImportError) as excinfo:
        readmat.readmrs(str(tmp_path / "v73.mat"))

    assert "libhdf5" in str(excinfo.value)
    assert "pyamares-xmris[matlab]" not in str(excinfo.value)


def test_excel_prior_without_an_engine_names_the_excel_extra(monkeypatch, tmp_path):
    """``generateparameter`` on an .xlsx prior, with no pandas Excel engine.

    pandas itself raises ``ImportError("Missing optional dependency 'openpyxl'")``
    in that situation; the wrapper has to turn it into an instruction.
    """
    import pandas as pd

    from pyAMARES.kernel import PriorKnowledge

    def missing_engine(*args, **kwargs):
        raise ImportError(
            "Missing optional dependency 'openpyxl'. Use pip or conda to install "
            "openpyxl."
        )

    monkeypatch.setattr(pd, "read_excel", missing_engine)

    with pytest.raises(ImportError) as excinfo:
        PriorKnowledge.generateparameter(str(tmp_path / "prior.xlsx"))

    message = str(excinfo.value)
    assert "pyamares-xmris[excel]" in message
    assert "CSV" in message  # the no-extra way out
    assert isinstance(excinfo.value.__cause__, ImportError)


#: The logger ``run_parallel_fitting_with_progress`` warns through.
PROGRESS_LOGGER = "pyAMARES.util.multiprocessing"


def test_progress_bar_falls_back_to_text_without_ipywidgets(monkeypatch, caplog):
    """The parallel fit must not die because ipywidgets is behind an extra.

    ``tqdm.notebook`` *imports* fine without ipywidgets and only raises
    ``ImportError("IProgress not found...")`` when a bar is constructed — which
    in ``run_parallel_fitting_with_progress`` happens after the whole process pool
    has been filled. Simulating it through tqdm's own ``IProgress`` flag means
    this asserts the same thing whether or not ipywidgets is installed here.
    """
    import tqdm
    import tqdm.notebook

    from pyAMARES.util.multiprocessing import _select_tqdm

    monkeypatch.setattr(tqdm.notebook, "IProgress", None, raising=False)

    # Name the logger: regression_cases sets every pyAMARES logger to ERROR, and
    # caplog.at_level() without a name only moves the root logger.
    with caplog.at_level("WARNING", logger=PROGRESS_LOGGER):
        selected = _select_tqdm(notebook=True)

    assert selected is tqdm.tqdm
    assert "pyamares-xmris[jupyter]" in caplog.text


def test_progress_bar_falls_back_when_tqdm_notebook_is_absent(monkeypatch, caplog):
    """The same fallback, for a tqdm too old to carry the submodule."""
    import tqdm

    from pyAMARES.util.multiprocessing import _select_tqdm

    monkeypatch.setitem(sys.modules, "tqdm.notebook", None)

    with caplog.at_level("WARNING", logger=PROGRESS_LOGGER):
        selected = _select_tqdm(notebook=True)

    assert selected is tqdm.tqdm
    assert "pyamares-xmris[jupyter]" in caplog.text


def test_progress_bar_uses_the_notebook_class_when_ipywidgets_is_usable():
    """The fallback must not fire when the widget bar would work."""
    import tqdm.notebook

    from pyAMARES.util.multiprocessing import _select_tqdm

    if getattr(tqdm.notebook, "IProgress", False) is None:
        pytest.skip("ipywidgets is not installed here, so there is nothing to pick")

    assert _select_tqdm(notebook=True) is tqdm.notebook.tqdm


def test_progress_bar_honours_notebook_false():
    import tqdm

    from pyAMARES.util.multiprocessing import _select_tqdm

    assert _select_tqdm(notebook=False) is tqdm.tqdm
