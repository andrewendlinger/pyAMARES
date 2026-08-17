"""Shared case runners for the pyAMARES numeric regression corpus.

This module is the single source of truth for *what* gets fitted. Both the golden
capture CLI (``tests/capture_goldens.py``) and the regression tests
(``tests/test_regression.py``, ``tests/test_api_surface.py``) import from here, so a
case can never drift between the value that was frozen and the value that is checked.

Deliberately free of ``pytest`` imports — it must be runnable as a plain library.

Every path is resolved relative to this file, and every input lives under ``tests/``,
so the runners are hermetic: no network, no ``pyAMARES/examples/`` copies, no CWD
dependence.
"""

from __future__ import annotations

import os

# pyAMARES imports matplotlib.pyplot eagerly (kernel/fid.py, libs/MPFIR.py). Pin a
# non-interactive backend before that happens so the runners work headless.
os.environ.setdefault("MPLBACKEND", "Agg")

from copy import deepcopy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pyAMARES  # noqa: E402
from pyAMARES.kernel.lmfit import fitAMARES, result_pd_to_params  # noqa: E402
from pyAMARES.libs.logger import set_log_level  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_FID_PATH = os.path.join(TESTS_DIR, "fid.txt")
EXAMPLE_PRIOR_PATH = os.path.join(TESTS_DIR, "example_human_brain_31P_7T.csv")
SYNTHETIC_PRIOR_PATH = os.path.join(TESTS_DIR, "priors", "synthetic_3peak.csv")
GOLDENS_DIR = os.path.join(TESTS_DIR, "goldens")

# --- Case A: the documented README quick-start acquisition parameters -------------
EXAMPLE_MHZ = 120.0
EXAMPLE_SW = 10000
EXAMPLE_DEADTIME = 300e-6
EXAMPLE_XLIM = (10, -20)

# --- Case B: the synthetic acquisition, frozen ------------------------------------
SYNTHETIC_MHZ = 120.0
SYNTHETIC_SW = 10000.0
SYNTHETIC_DEADTIME = 0.0
SYNTHETIC_N_POINTS = 2048
SYNTHETIC_SEED = 0
#: Noise scales, indexed from 1 in the case names ("noise1" is the high-SNR case).
SYNTHETIC_NOISE_SCALES = {1: 0.002, 2: 0.010}

#: The four synthetic variants, keyed by name. Names rather than ``(int, g_global)``
#: tuples on purpose: ``0.0 == False`` in Python, so ``(1, 0.0)`` and ``(1, False)``
#: collide as dict keys and silently collapse into one entry.
SYNTHETIC_VARIANTS = {
    "noise1_g0": (1, 0.0),
    "noise2_g0": (2, 0.0),
    "noise1_gfree": (1, False),
    "noise2_gfree": (2, False),
}

#: The truth the synthetic FID is built from. ``Beta`` sits exactly 5 ppm above
#: ``Alpha`` because the prior-knowledge CSV ties it there with an ``Alpha+600Hz``
#: expression (600 Hz / 120 MHz = 5 ppm), and ``Beta``'s phase is tied to ``Alpha``'s.
SYNTHETIC_GROUND_TRUTH = pd.DataFrame(
    {
        "amplitude": [1.00, 0.60, 0.30],
        "chem shift(ppm)": [0.00, 5.00, -8.00],
        "LW(Hz)": [20.0, 30.0, 15.0],
        "phase(deg)": [10.0, 10.0, 10.0],
        "g": [0.0, 0.0, 0.0],
    },
    index=["Alpha", "Beta", "Gamma"],
)

# --- Case C: HSVD ------------------------------------------------------------------
HSVD_NUM_COMPONENTS = 8


def quiet() -> None:
    """Silence pyAMARES' INFO chatter so a capture or test run stays readable."""
    set_log_level("error", verbose=False)


def _synthetic_timeaxis() -> np.ndarray:
    """Rebuild the time axis ``initialize_FID`` will construct for the synthetic FID."""
    dwelltime = 1.0 / SYNTHETIC_SW
    return np.arange(0, dwelltime * SYNTHETIC_N_POINTS, dwelltime) + SYNTHETIC_DEADTIME


def synthetic_ground_truth() -> pd.DataFrame:
    """Return a copy of the ground-truth parameter table for the synthetic FID."""
    return SYNTHETIC_GROUND_TRUTH.copy(deep=True)


def make_synthetic_fid(noise_scale: float) -> np.ndarray:
    """Build the synthetic complex FID from :data:`SYNTHETIC_GROUND_TRUTH`.

    The noiseless signal is produced through the library's own forward model
    (``result_pd_to_params`` -> ``multieq6`` -> ``uninterleave``), so the FID is
    exactly what the fitter's model can represent. Noise is deterministic complex
    Gaussian from ``np.random.default_rng(SYNTHETIC_SEED)`` — a fresh generator per
    call, so the two noise scales are independent of the order they are requested in.

    Parameters
    ----------
    noise_scale : float
        Standard deviation of the real and of the imaginary noise channel.

    Returns
    -------
    numpy.ndarray
        Complex FID of length :data:`SYNTHETIC_N_POINTS`.
    """
    params = result_pd_to_params(SYNTHETIC_GROUND_TRUTH, MHz=SYNTHETIC_MHZ)
    pure = pyAMARES.uninterleave(
        pyAMARES.multieq6(params=params, x=_synthetic_timeaxis())
    )
    rng = np.random.default_rng(SYNTHETIC_SEED)
    noise = rng.standard_normal(SYNTHETIC_N_POINTS) + 1j * rng.standard_normal(
        SYNTHETIC_N_POINTS
    )
    return pure + noise_scale * noise


def initialize_example_fid():
    """Load the example FID and prior knowledge exactly as the README quick-start does."""
    quiet()
    fid = pyAMARES.readmrs(EXAMPLE_FID_PATH)
    return pyAMARES.initialize_FID(
        fid=fid,
        priorknowledgefile=EXAMPLE_PRIOR_PATH,
        MHz=EXAMPLE_MHZ,
        sw=EXAMPLE_SW,
        deadtime=EXAMPLE_DEADTIME,
        normalize_fid=False,
        preview=False,
        xlim=EXAMPLE_XLIM,
    )


def run_example_case():
    """Case A — the documented two-stage README fit.

    ``leastsq`` initializes, then ``least_squares`` refines the optimized parameters,
    both with ``inplace=False`` (the default), mirroring README.rst.

    Returns
    -------
    argparse.Namespace
        The fitted FID object returned by the second ``fitAMARES`` call.
    """
    fidobj = initialize_example_fid()
    out1 = fitAMARES(
        fid_parameters=fidobj,
        fitting_parameters=fidobj.initialParams,
        method="leastsq",
        ifplot=False,
    )
    out2 = fitAMARES(
        fid_parameters=out1,
        fitting_parameters=out1.fittedParams,
        method="least_squares",
        ifplot=False,
    )
    return out2


def run_example_xmris_shape_case():
    """Case A' — the exact call shape xmris uses per voxel.

    A ``deepcopy`` of the initialized object is fitted in place with
    ``least_squares``, ``initialize_with_lm=False`` and no LM warm-up, and the
    results are read back off the copy (``fitAMARES`` returns the
    ``MinimizerResult``, not the FID object, when ``inplace=True``).

    Returns
    -------
    argparse.Namespace
        The in-place-fitted copy of the initialized FID object.
    """
    fidobj = initialize_example_fid()
    fidobj_copy = deepcopy(fidobj)
    fitAMARES(
        fid_parameters=fidobj_copy,
        fitting_parameters=fidobj_copy.initialParams,
        method="least_squares",
        initialize_with_lm=False,
        ifplot=False,
        inplace=True,
    )
    return fidobj_copy


def run_synthetic_case(noise_scale: float, g_global):
    """Case B — fit the synthetic 3-peak FID with the hand-written prior knowledge.

    Parameters
    ----------
    noise_scale : float
        Passed to :func:`make_synthetic_fid`.
    g_global : float or bool
        Forwarded to ``initialize_FID``. ``0.0`` pins every ``g`` to zero and fixes
        it; ``False`` lets the CSV's own ``g`` values vary.

    Returns
    -------
    tuple
        ``(fitted_fidobj, ground_truth_dataframe)``.
    """
    quiet()
    fid = make_synthetic_fid(noise_scale)
    fidobj = pyAMARES.initialize_FID(
        fid=fid,
        priorknowledgefile=SYNTHETIC_PRIOR_PATH,
        MHz=SYNTHETIC_MHZ,
        sw=SYNTHETIC_SW,
        deadtime=SYNTHETIC_DEADTIME,
        normalize_fid=False,
        preview=False,
        g_global=g_global,
        xlim=(10, -20),
    )
    fitted = fitAMARES(
        fid_parameters=fidobj,
        fitting_parameters=fidobj.initialParams,
        method="least_squares",
        initialize_with_lm=False,
        ifplot=False,
    )
    return fitted, synthetic_ground_truth()


def run_synthetic_variant(variant: str):
    """Run one named entry of :data:`SYNTHETIC_VARIANTS`."""
    noise_key, g_global = SYNTHETIC_VARIANTS[variant]
    return run_synthetic_case(SYNTHETIC_NOISE_SCALES[noise_key], g_global=g_global)


def run_hsvd_case():
    """Case C — the HSVD initializer on the Case A example FID.

    Structural only: the two HSVD backends (``hlsvdpro`` and the vendored pure-Python
    ``pyAMARES.libs.hlsvd``) differ numerically by design, so nothing here is frozen.

    Returns
    -------
    dict
        ``fidobj``, the returned ``params``, and a ``table`` DataFrame with one row
        per HSVD component (``amplitude``/``ppm``/``linewidth_hz``/``phase_rad``).
    """
    quiet()
    fidobj = initialize_example_fid()
    params = pyAMARES.HSVDinitializer(
        fid_parameters=fidobj,
        fitting_parameters=None,
        num_of_component=HSVD_NUM_COMPONENTS,
        preview=False,
        verbose=False,
    )
    rows = {}
    for name, param in params.items():
        prefix, peak = name.split("_", 1)
        rows.setdefault(peak, {})[prefix] = param.value
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["amplitude"] = table["ak"].abs()
    table["ppm"] = table["freq"] / fidobj.MHz
    table["linewidth_hz"] = table["dk"] / np.pi
    table["phase_rad"] = table["phi"]
    return {"fidobj": fidobj, "params": params, "table": table}


#: The golden cases, by name. Each entry returns the FID object whose
#: ``result_multiplets`` gets frozen. ``capture_goldens.py`` and
#: ``test_regression.py`` both iterate this mapping, so adding a golden case is a
#: one-line change here.
GOLDEN_CASES = {
    "example_readme": run_example_case,
    "example_xmris_shape": run_example_xmris_shape_case,
    "synthetic_noise1_gfree": lambda: run_synthetic_case(
        SYNTHETIC_NOISE_SCALES[1], g_global=False
    )[0],
}

_CASE_CACHE: dict = {}


def golden_case_result(name: str):
    """Run a golden case at most once per process and return its fitted FID object.

    Fitting is cheap but not free, and several tests read the same case, so the
    result is memoized. Callers must treat the returned object as read-only.
    """
    if name not in _CASE_CACHE:
        _CASE_CACHE[name] = GOLDEN_CASES[name]()
    return _CASE_CACHE[name]


#: Columns frozen bit-for-bit. ``sd``/``sd(ppm)``/``sd(Hz)``/``sd(deg)`` are
#: deliberately absent — see :data:`STRUCTURAL_COLUMNS`.
GOLDEN_COLUMNS = [
    "amplitude",
    "chem shift(ppm)",
    "LW(Hz)",
    "phase(deg)",
    "SNR",
    "CRLB(%)",
    "CRLB(cs%) ",
    "CRLB(LW%)",
    "CRLB(phase%)",
]

#: Never golden-compared: the lmfit standard errors travel through an
#: ill-conditioned Fisher matrix and are not reproducible across dependency stacks.
STRUCTURAL_COLUMNS = ["sd", "sd(ppm)", "sd(Hz)", "sd(deg)"]
