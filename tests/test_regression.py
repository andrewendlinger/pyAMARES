# The one supported invocation of this suite — copy it verbatim:
#
#     pytest tests/test_regression.py tests/test_api_surface.py -o addopts=""
#
# `-o addopts=""` is not optional: pytest.ini injects --nbval-lax, which errors out
# in a minimal environment that has no nbval installed.
"""Numeric regression tests: the frozen goldens, plus stack-independent invariants.

What is frozen and what is not
------------------------------
The fitted parameters and every CRLB column are golden-compared cell by cell against
``tests/goldens/*.json``. The four ``sd`` columns never are — they come out of an
ill-conditioned Fisher matrix through ``pinv``/``lstsq`` and are not reproducible
across dependency stacks. They are guarded *structurally* instead, below.

Which golden file a run compares against is platform-dependent:
``tests/goldens/<sys.platform>-<machine>/<case>.json`` wins per file if it exists,
otherwise the canonical ``tests/goldens/<case>.json`` — captured on darwin-arm64 and
the reference for every platform that has no override. Every failure message names
the file it compared.

Tolerances live inside each golden JSON, so loosening one later (say, for cross-BLAS
drift on a CRLB column) is a data-only edit with a comment — never a code change.
They are not bit-for-bit and cannot be: numpy's SIMD complex multiply inside
``equation6`` is memory-alignment sensitive, which makes the fit reproducible only to
about 1e-5 relative even on one machine, one process, back to back. See the comment
on ``DEFAULT_TOLERANCES`` in ``tests/capture_goldens.py`` for the full diagnosis.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pytest

try:  # imported as part of the ``tests`` package
    from tests import regression_cases as rc
except ImportError:  # pragma: no cover - direct invocation from inside tests/
    import regression_cases as rc  # type: ignore[no-redef]


# --------------------------------------------------------------------------------
# Golden comparison
# --------------------------------------------------------------------------------


def golden_path(name: str) -> str:
    """Where this platform reads golden ``name`` from.

    ``tests/goldens/<platform-key>/<name>.json`` wins if it exists, otherwise the
    canonical ``tests/goldens/<name>.json``. The resolution is **per file**, not
    per directory: a platform set that only overrides one case leaves every other
    case reading the canonical arm64 file, which is exactly the behaviour every
    platform had before platform directories existed.
    """
    candidate = os.path.join(rc.platform_goldens_dir(), f"{name}.json")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(rc.GOLDENS_DIR, f"{name}.json")


def load_golden(name: str) -> dict:
    path = golden_path(name)
    if not os.path.exists(path):
        raise AssertionError(
            f"Missing golden {path}. Capture it with:\n"
            f"    python tests/capture_goldens.py --write tests/goldens/"
        )
    with open(path, encoding="utf-8") as handle:
        golden = json.load(handle)
    # Every failure message below quotes this, so a red run says which of the two
    # candidate files it actually compared against.
    golden["_source_path"] = os.path.relpath(path, os.path.dirname(rc.TESTS_DIR))
    return golden


class GoldenComparer:
    """Cell-by-cell comparison against one golden's tolerance block.

    Shared by both golden families — the ``result_multiplets`` tables and the HSVD
    decomposition — because they had started to drift apart: two copies of the
    same resolve-tolerance/isclose/format-the-mismatch loop, one keying its
    overrides ``per_column`` and the other ``per_field``, with different NaN
    guards and different message layouts. One helper, one JSON schema.

    ``per_column`` is the schema's name for the override map even where the
    payload calls its axis something else: the canonical fit goldens are frozen
    and already spell it that way, so it is the spelling the HSVD golden adopted
    rather than the other way round.

    Mismatches accumulate instead of raising, so one red run shows every drifted
    cell — a dependency bump that moves ten metabolites should show all ten, not
    just the alphabetically first.
    """

    def __init__(self, tolerances: dict):
        self.default_rtol = tolerances["default_rtol"]
        self.default_atol = tolerances.get("default_atol", 0.0)
        self.per_column = tolerances.get("per_column", {})
        self.mismatches: list = []

    def tolerances_for(self, column: str):
        """``(rtol, atol)`` for one column/field, per-column override applied."""
        override = self.per_column.get(column, {})
        return (
            override.get("rtol", self.default_rtol),
            override.get("atol", self.default_atol),
        )

    def check(self, label: str, column: str, expected: float, actual: float) -> None:
        """Compare one cell; record a formatted line if it is out of tolerance."""
        rtol, atol = self.tolerances_for(column)
        if np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
            return
        # A relative deviation is meaningless against zero or NaN; say so rather
        # than dividing.
        rel = (
            abs(actual - expected) / abs(expected)
            if expected and not math.isnan(expected)
            else float("nan")
        )
        self.mismatches.append(
            f"  {label:34s} expected {expected!r} got {actual!r} "
            f"(rel {rel:.3e}, abs {abs(actual - expected):.3e}, "
            f"rtol {rtol:.1e} atol {atol:.1e})"
        )


@pytest.mark.parametrize("case_name", sorted(rc.GOLDEN_CASES))
def test_golden_result_multiplets(case_name):
    """Every golden cell of ``result_multiplets`` still matches the frozen value.

    Mismatches are collected and reported together — a dependency bump that moves
    ten metabolites should show all ten, not just the alphabetically first.
    """
    golden = load_golden(case_name)
    source = golden["_source_path"]
    df = rc.golden_case_result(case_name).result_multiplets

    # Guard the golden against the *live* constants, not against its own copies of
    # them (both sides of a JSON-vs-JSON comparison were written by the same
    # capture run, so it can never fail). If someone edits GOLDEN_COLUMNS or
    # RESULT_MULTIPLETS_COLUMNS without re-capturing, these two lines catch it.
    assert set(golden["values"]) == set(rc.GOLDEN_COLUMNS), (
        f"{source}: the golden freezes {sorted(golden['values'])} but "
        f"regression_cases.GOLDEN_COLUMNS says {sorted(rc.GOLDEN_COLUMNS)} — "
        "re-capture the goldens."
    )
    assert golden["columns_exact_order"] == rc.RESULT_MULTIPLETS_COLUMNS, (
        f"{source}: the golden's column order disagrees with "
        "regression_cases.RESULT_MULTIPLETS_COLUMNS — re-capture the goldens."
    )

    assert [str(x) for x in df.index] == golden["index"], (
        f"{source}: the metabolite row index changed.\n"
        f"  expected: {golden['index']}\n"
        f"  actual:   {[str(x) for x in df.index]}"
    )
    assert [str(c) for c in df.columns] == golden["columns_exact_order"], (
        f"{source}: result_multiplets columns changed (label text or order).\n"
        f"  expected: {golden['columns_exact_order']}\n"
        f"  actual:   {[str(c) for c in df.columns]}"
    )

    comparer = GoldenComparer(golden["tolerances"])
    for column, expected_col in golden["values"].items():
        for metabolite, expected in expected_col.items():
            comparer.check(
                f"{column!r} {metabolite}",
                column,
                expected,
                float(df.at[metabolite, column]),
            )

    mismatches = comparer.mismatches
    assert not mismatches, (
        f"{case_name}: {len(mismatches)} golden cell(s) drifted against {source} "
        f"(captured on {golden['meta']['python']}/"
        f"numpy {golden['meta']['numpy']}/pandas {golden['meta']['pandas']}/"
        f"{golden['meta']['machine']}, running on numpy {np.__version__}/"
        f"{rc.platform_goldens_key()}):\n" + "\n".join(mismatches)
    )


def _version_tuple(text: str):
    """``"0.4.0" -> (0, 4, 0)``; ``None`` if any segment is not a plain integer.

    Deliberately not a PEP 440 parser — there is no packaging dependency in this
    suite, and a version this cannot read simply opts out of the comparison it
    feeds rather than failing it.
    """
    parts = text.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def test_every_golden_file_has_a_case():
    """No orphan goldens: a file under tests/goldens/ must map to a live case."""
    on_disk = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(rc.GOLDENS_DIR, "*.json"))
    }
    assert on_disk == set(rc.ALL_GOLDEN_NAMES), (
        "tests/goldens/ and the case set in regression_cases disagree.\n"
        f"  only on disk: {sorted(on_disk - set(rc.ALL_GOLDEN_NAMES))}\n"
        f"  only in code: {sorted(set(rc.ALL_GOLDEN_NAMES) - on_disk)}"
    )


def test_platform_golden_dirs_are_named_and_populated_correctly():
    """Every ``tests/goldens/<subdir>/`` is a well-formed platform override set.

    Three rules, all cheap, and all about files *this* platform may never read —
    which is the point: a wrong linux override is invisible on arm64 until CI
    goes red, so it gets policed everywhere.

    * The directory name must parse as ``<sys.platform>-<machine>`` (the grammar
      lives in ``regression_cases``, shared with the capture script), so a stray
      ``tests/goldens/old/`` cannot masquerade as a platform set that silently
      never applies.
    * Its ``*.json`` names must be a **subset** of the case set — a subset, not a
      bijection, because a platform set is allowed to override only the cases that
      actually drift there and inherit the rest from the canonical arm64 files.
      Anything that is not a ``*.json`` is an orphan too, dotfiles excepted:
      committing a platform set from macOS plants a ``.DS_Store`` next to it, and
      failing the suite over a Finder artefact would teach people to distrust it.
    * An override must not be **older than the canonical it shadows**. Nothing
      else links the two: re-freezing a canonical golden silently leaves every
      committed override in place, still winning the lookup, still asserting
      numbers from before the re-freeze. Compared on ``captured_utc`` (fixed-width
      UTC, so lexicographic order is chronological) and on the recorded pyamares
      version. Both are "not older", not "equal" — the canonical fit goldens still
      carry 0.3.33 from the pre-fork capture, so demanding equality would condemn
      every override captured since.
    """
    problems = []
    for entry in sorted(os.listdir(rc.GOLDENS_DIR)):
        path = os.path.join(rc.GOLDENS_DIR, entry)
        if not os.path.isdir(path):
            continue
        if not rc.looks_like_platform_dir(entry):
            problems.append(
                f"  {entry}/: not a '<sys.platform>-<machine>' directory name "
                f"(this platform's key is {rc.platform_goldens_key()!r})"
            )
            continue
        names = sorted(n for n in os.listdir(path) if not n.startswith("."))
        stray = [n for n in names if not n.endswith(".json")]
        if stray:
            problems.append(f"  {entry}/: non-golden file(s) {stray}")
        goldens = [n for n in names if n.endswith(".json")]
        orphans = sorted(
            {os.path.splitext(n)[0] for n in goldens} - set(rc.ALL_GOLDEN_NAMES)
        )
        if orphans:
            problems.append(f"  {entry}/: golden(s) with no case {orphans}")
        for name in goldens:
            canonical_path = os.path.join(rc.GOLDENS_DIR, name)
            if not os.path.exists(canonical_path):
                continue
            with open(os.path.join(path, name), encoding="utf-8") as handle:
                override_meta = json.load(handle).get("meta", {})
            with open(canonical_path, encoding="utf-8") as handle:
                canonical_meta = json.load(handle).get("meta", {})
            problems.extend(
                _stale_override_problems(entry, name, override_meta, canonical_meta)
            )
    assert not problems, "malformed platform golden directories:\n" + "\n".join(
        problems
    )


def _stale_override_problems(entry, name, override_meta, canonical_meta):
    """Report an override golden that predates the canonical it shadows."""
    problems = []
    override_when = override_meta.get("captured_utc")
    canonical_when = canonical_meta.get("captured_utc")
    if override_when and canonical_when and override_when < canonical_when:
        problems.append(
            f"  {entry}/{name}: captured {override_when}, but the canonical it "
            f"shadows was re-captured later ({canonical_when}) — re-capture the "
            "override or drop it."
        )
    override_version = _version_tuple(override_meta.get("pyamares", ""))
    canonical_version = _version_tuple(canonical_meta.get("pyamares", ""))
    if override_version and canonical_version and override_version < canonical_version:
        problems.append(
            f"  {entry}/{name}: captured on pyamares "
            f"{override_meta['pyamares']}, older than the canonical it shadows "
            f"({canonical_meta['pyamares']}) — re-capture the override or drop it."
        )
    return problems


@pytest.mark.parametrize("case_name", sorted(rc.GOLDEN_CASES))
def test_goldens_never_freeze_the_sd_columns(case_name):
    """A re-capture must not quietly start freezing the non-reproducible sd columns."""
    golden = load_golden(case_name)
    frozen = set(golden["values"])
    leaked = frozen & set(rc.STRUCTURAL_COLUMNS)
    assert not leaked, (
        f"{case_name}: sd column(s) {sorted(leaked)} were golden-compared. They are "
        "not reproducible across dependency stacks — guard them structurally instead."
    )
    # Against the live constant, never against the golden's own copy of it — a
    # JSON-vs-JSON comparison was written by one capture run and cannot fail.
    assert frozen == set(rc.GOLDEN_COLUMNS)


# --------------------------------------------------------------------------------
# Case B — synthetic 3-peak fit: accuracy against the known ground truth
# --------------------------------------------------------------------------------

VARIANT_IDS = list(rc.SYNTHETIC_VARIANTS)


@pytest.fixture(scope="module")
def synthetic_fits():
    """Every synthetic variant, fitted once per process (shared with the goldens)."""
    return {name: rc.synthetic_variant_result(name) for name in rc.SYNTHETIC_VARIANTS}


@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_synthetic_recovers_ground_truth(synthetic_fits, variant):
    """The fit recovers the parameters the synthetic FID was built from.

    Stack-independent by construction: this checks physics, not float bits, so it
    keeps meaning after a dependency bump that legitimately perturbs the last digits.

    The linewidth check applies only to the ``g_global=0.0`` variants. With ``g``
    free the model carries a Gaussian/Lorentzian mixing parameter the noiseless
    truth does not use, and ``g`` trades off against the damping factor: the fit
    stays excellent in amplitude and chemical shift but parks some of the decay in
    ``g``, so the reported ``LW(Hz)`` legitimately drifts (measured up to +77% on
    the broadest peak at the low-SNR scale). Asserting a bound wide enough to pass
    there would assert nothing.
    """
    noise_key, g_global = rc.SYNTHETIC_VARIANTS[variant]
    check_linewidth = g_global is not False
    fitted, truth = synthetic_fits[variant]
    result = fitted.result_multiplets
    assert list(result.index) == list(truth.index)

    problems = []
    for name in truth.index:
        amp_rel = abs(result.at[name, "amplitude"] - truth.at[name, "amplitude"]) / abs(
            truth.at[name, "amplitude"]
        )
        if amp_rel > 0.05:
            problems.append(f"  {name}: amplitude off by {amp_rel:.2%} (> 5%)")

        shift_abs = abs(
            result.at[name, "chem shift(ppm)"] - truth.at[name, "chem shift(ppm)"]
        )
        if shift_abs > 0.05:
            problems.append(f"  {name}: chem shift off by {shift_abs:.4f} ppm (> 0.05)")

        if check_linewidth:
            lw_rel = abs(result.at[name, "LW(Hz)"] - truth.at[name, "LW(Hz)"]) / abs(
                truth.at[name, "LW(Hz)"]
            )
            if lw_rel > 0.10:
                problems.append(f"  {name}: LW off by {lw_rel:.2%} (> 10%)")

    assert not problems, (
        f"synthetic fit (noise scale {rc.SYNTHETIC_NOISE_SCALES[noise_key]}, "
        f"g_global={g_global!r}) missed the ground truth:\n" + "\n".join(problems)
    )


# --------------------------------------------------------------------------------
# Case B — the sd columns, structurally only
# --------------------------------------------------------------------------------

#: ``(value column, sd column, CRLB column, lmfit parameter prefix)`` — the four
#: parameter families ``report_amares`` reports a standard deviation and a CRLB for.
SD_CRLB_FAMILIES = [
    ("amplitude", "sd", "CRLB(%)", "ak"),
    ("chem shift(ppm)", "sd(ppm)", "CRLB(cs%) ", "freq"),
    ("LW(Hz)", "sd(Hz)", "CRLB(LW%)", "dk"),
    ("phase(deg)", "sd(deg)", "CRLB(phase%)", "phi"),
]

#: Agreement with the fit-wide sd/CRLB scale factor demanded of a freely varying
#: parameter. Measured worst deviation over 30 repeat runs: 1.5e-6.
FREE_K_RTOL = 1e-4
#: ...and of a parameter pinned by an ``expr`` tie, whose standard error is
#: propagated while its CRLB is not. Measured worst deviation: 5.8e-3.
TIED_K_RTOL = 2e-2


@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_synthetic_sd_columns_are_well_formed(synthetic_fits, variant):
    """All four sd columns are finite, strictly positive, and smaller than the signal."""
    result = synthetic_fits[variant][0].result_multiplets
    problems = []
    for column in rc.STRUCTURAL_COLUMNS:
        values = result[column]
        for name, value in values.items():
            if not np.isfinite(value):
                problems.append(f"  {column!r} {name}: not finite ({value!r})")
            elif value <= 0:
                problems.append(f"  {column!r} {name}: not positive ({value!r})")
    for name in result.index:
        if not result.at[name, "sd"] < result.at[name, "amplitude"]:
            problems.append(
                f"  sd {name}: {result.at[name, 'sd']!r} is not smaller than the "
                f"amplitude {result.at[name, 'amplitude']!r}"
            )
    assert not problems, "malformed sd columns:\n" + "\n".join(problems)


@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_sd_and_crlb_are_proportional(synthetic_fits, variant):
    """``sd = k * (CRLB/100) * |value|`` with one constant ``k`` per fit.

    Measured, not assumed. The obvious guess — ``CRLB(%) == 100*sd/|value|``, i.e.
    ``k == 1`` — is **false**: ``report_amares`` fills ``sd`` from lmfit's own
    standard errors (scaled by the reduced chi-square) and ``CRLB(%)`` from
    ``evaluateCRB``'s Fisher-matrix bound (scaled by the OXSA noise-variance
    estimate). Both are the same square root of the same inverse Fisher diagonal
    under different noise normalisations, so they are exactly proportional with a
    single per-fit constant. On the baseline stack that constant is ~0.9976 for the
    synthetic cases and ~0.72875 for the documented example fit — nowhere near 1.

    ``k`` is read off the amplitude family, where it is reproducible to 4e-7 over 30
    repeat runs. Two tolerances follow, both measured over those runs:

    * a parameter the prior knowledge lets vary freely agrees with the fit-wide
      ``k`` to 1.5e-6, so it is held to ``FREE_K_RTOL`` (1e-4, a 60x margin);
    * a parameter pinned by an ``expr`` tie — ``Beta``'s chemical shift and phase
      here — does not. lmfit *propagates* the tied parameter's standard error from
      the peak it is tied to, while ``evaluateCRB`` computes that peak's CRLB
      independently, and the two disagree by up to 5.8e-3 in a way that is itself
      not reproducible run to run. Tied cells are held to ``TIED_K_RTOL`` (2e-2).
    """
    fitted = synthetic_fits[variant][0]
    result = fitted.result_multiplets
    tied = {
        tuple(name.split("_", 1))
        for name, param in fitted.fittedParams.items()
        if param.expr
    }

    def implied_k(value_col, sd_col, crlb_col):
        return result[sd_col] / (result[crlb_col] / 100.0 * result[value_col].abs())

    amplitude_k = implied_k(*SD_CRLB_FAMILIES[0][:3])
    assert np.allclose(amplitude_k, amplitude_k.iloc[0], rtol=1e-5), (
        f"k is not constant within the amplitude family: {amplitude_k.to_dict()}"
    )
    k = float(amplitude_k.median())
    assert 0.0 < k < 10.0, f"implausible sd/CRLB scale factor k={k!r}"

    problems = []
    for value_col, sd_col, crlb_col, prefix in SD_CRLB_FAMILIES:
        for name, value in implied_k(value_col, sd_col, crlb_col).items():
            is_tied = (prefix, name) in tied
            rtol = TIED_K_RTOL if is_tied else FREE_K_RTOL
            if not np.isclose(value, k, rtol=rtol):
                problems.append(
                    f"  {sd_col!r} {name} ({'expr-tied' if is_tied else 'free'}): "
                    f"k={value!r} deviates from the fit-wide k={k!r} by "
                    f"{abs(value - k) / k:.2e} (> {rtol:.0e})"
                )
    assert not problems, (
        f"sd and CRLB are no longer proportional (fit-wide k={k!r}):\n"
        + "\n".join(problems)
    )


@pytest.mark.parametrize("suffix", ["g0", "gfree"])
def test_sd_grows_with_noise(synthetic_fits, suffix):
    """More noise, larger standard deviations — for every metabolite, every column."""
    low = synthetic_fits[f"noise1_{suffix}"][0].result_multiplets
    high = synthetic_fits[f"noise2_{suffix}"][0].result_multiplets
    problems = []
    for column in rc.STRUCTURAL_COLUMNS:
        for name in low.index:
            if not high.at[name, column] > low.at[name, column]:
                problems.append(
                    f"  {column!r} {name}: noise2 sd {high.at[name, column]!r} is not "
                    f"greater than noise1 sd {low.at[name, column]!r}"
                )
    assert not problems, (
        "the sd columns did not grow with the noise scale "
        f"({rc.SYNTHETIC_NOISE_SCALES[1]} -> {rc.SYNTHETIC_NOISE_SCALES[2]}):\n"
        + "\n".join(problems)
    )


# --------------------------------------------------------------------------------
# Case C — the HSVD path: structural only, no goldens
# --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hsvd_result():
    return rc.run_hsvd_case()


def test_hsvd_components_are_plausible(hsvd_result):
    """The HSVD decomposition of the example FID stays physically sane.

    No numbers are frozen here: the two backends (``hlsvdpro`` and the vendored
    pure-Python ``pyAMARES.libs.hlsvd``) differ numerically by design.
    """
    table = hsvd_result["table"]
    fidobj = hsvd_result["fidobj"]
    assert 0 < len(table) <= rc.HSVD_NUM_COMPONENTS

    raw = table[["ak", "freq", "dk", "phi", "g"]].to_numpy(dtype=float)
    assert np.isfinite(raw).all(), f"non-finite HSVD parameters:\n{table}"

    assert (table["amplitude"] > 0).all(), f"non-positive amplitudes:\n{table}"
    # >= 0, not > 0: HSVDinitializer's dk filter is one-sided (dk < lw_threshold,
    # pyAMARES/util/hsvd.py), so a dk <= 0 component from an unbounded curve_fit
    # survives it and the lmfit parameter's min=0 then clips it to exactly 0.0.
    # Only the vendored backend has been observed here; hlsvdpro (x86_64, numpy 1.x)
    # differs numerically by design and may produce that clipped edge case.
    assert (table["linewidth_hz"] >= 0).all(), f"negative linewidths:\n{table}"
    # HSVDinitializer filters on dk < lw_threshold (500 rad/s) before returning.
    assert (table["linewidth_hz"] < 500.0 / np.pi).all(), (
        f"a component survived the linewidth filter it should not have:\n{table}"
    )

    # Sanity bound, not a contract: the library never constrains freq, so the
    # curve_fit refinement may nudge a component slightly past the +-sw/2 window.
    # 2x Nyquist would mean the decomposition lost the plot entirely.
    nyquist_ppm = rc.EXAMPLE_SW / 2.0 / rc.EXAMPLE_MHZ
    assert table["ppm"].abs().max() < 2.0 * nyquist_ppm, (
        f"a component sits far outside the +-{nyquist_ppm:.2f} ppm spectral window:\n"
        f"{table}"
    )

    # HSVDinitializer records how much of the data the components explain. The
    # library does not bound it above by 1, so assert only positive-and-finite.
    assert np.isfinite(fidobj.relativeNorm) and fidobj.relativeNorm > 0.0, (
        f"HSVD residual norm ratio {fidobj.relativeNorm!r} is not a positive number"
    )


def test_hsvd_dominant_component_is_pcr(hsvd_result):
    """The strongest component of the 31P example FID is PCr, at 0 ppm."""
    table = hsvd_result["table"]
    dominant = table["amplitude"].idxmax()
    ppm = float(table.at[dominant, "ppm"])
    assert abs(ppm) <= 0.5, (
        f"the dominant HSVD component sits at {ppm:.3f} ppm, more than 0.5 ppm from "
        f"PCr at 0 ppm:\n{table}"
    )


def test_hsvd_backend_selection():
    """The ``hlsvd`` symbol resolves per ``pyAMARES/util/hsvd.py``'s own rule.

    ``util/hsvd.py`` binds ``hlsvd`` to the vendored ``pyAMARES.libs.hlsvd`` when
    numpy is 2.x or when ``hlsvdpro`` is not installed, and to ``hlsvdpro``
    otherwise. Both bindings are *modules*, so the identity check is on
    ``__name__``; a module has no ``__module__`` attribute.
    """
    import pyAMARES.util.hsvd as hsvd_module

    numpy_major = int(np.__version__.split(".")[0])
    try:
        import hlsvdpro  # noqa: F401

        hlsvdpro_importable = True
    except ImportError:
        hlsvdpro_importable = False

    expect_vendored = numpy_major >= 2 or not hlsvdpro_importable
    expected = "pyAMARES.libs.hlsvd" if expect_vendored else "hlsvdpro"
    assert hsvd_module.hlsvd.__name__ == expected, (
        f"numpy {np.__version__} (major {numpy_major}), hlsvdpro importable="
        f"{hlsvdpro_importable} should select {expected!r}, but util/hsvd.py bound "
        f"{hsvd_module.hlsvd.__name__!r}"
    )


# --------------------------------------------------------------------------------
# Case D — the vendored HSVD backend, frozen
# --------------------------------------------------------------------------------


def test_hsvd_vendored_backend_matches_golden():
    """The vendored pure-Python HSVD decomposition still returns the frozen numbers.

    Unlike the structural Case C tests above, this one *does* freeze values — but
    of ``pyAMARES.libs.hlsvd.hlsvd`` called directly, so "the two backends differ
    by design" never applies: whichever backend ``util/hsvd.py`` happens to bind,
    this test measures the vendored one.

    Tolerances live in the golden and were measured across dependency stacks and
    platforms; the golden's ``comment`` records the numbers.
    """
    golden = load_golden(rc.HSVD_VENDORED_CASE)
    source = golden["_source_path"]
    result = rc.run_hsvd_vendored_backend_case()

    # Against the live constant, never against the golden's copy of it.
    assert golden["component_fields"] == list(rc.HSVD_COMPONENT_FIELDS), (
        f"{source}: the golden freezes {golden['component_fields']} but "
        f"regression_cases.HSVD_COMPONENT_FIELDS says "
        f"{list(rc.HSVD_COMPONENT_FIELDS)} — re-capture the golden."
    )
    assert result["nsv_found"] == golden["nsv_found"], (
        f"{source}: the vendored HSVD backend found {result['nsv_found']} singular "
        f"value(s), the golden froze {golden['nsv_found']}"
    )

    components = result["components"]
    assert len(components) == len(golden["components"]), (
        f"{source}: the backend returned {len(components)} component(s), the golden "
        f"froze {len(golden['components'])} — the rest of the comparison is "
        "meaningless, so it is not attempted."
    )

    comparer = GoldenComparer(golden["tolerances"])
    for row, expected_row in enumerate(golden["components"]):
        for field in rc.HSVD_COMPONENT_FIELDS:
            comparer.check(
                f"component[{row}].{field}",
                field,
                float(expected_row[field]),
                float(components.at[row, field]),
            )

    expected_sv = golden["top_singular_values"]
    actual_sv = result["top_singular_values"]
    assert len(actual_sv) == len(expected_sv), (
        f"{source}: {len(actual_sv)} singular value(s) returned, "
        f"{len(expected_sv)} frozen"
    )
    for i, (expected, actual) in enumerate(zip(expected_sv, actual_sv)):
        comparer.check(
            f"singular_value[{i}]", "top_singular_values", expected, float(actual)
        )

    mismatches = comparer.mismatches
    assert not mismatches, (
        f"{rc.HSVD_VENDORED_CASE}: {len(mismatches)} frozen value(s) drifted against "
        f"{source} (captured on {golden['meta']['python']}/"
        f"numpy {golden['meta']['numpy']}/scipy {golden['meta']['scipy']}/"
        f"{golden['meta']['machine']}, running on numpy {np.__version__}/"
        f"{rc.platform_goldens_key()}):\n" + "\n".join(mismatches)
    )


# --------------------------------------------------------------------------------
# Prior-knowledge dtype handling
# --------------------------------------------------------------------------------


def _initialize_rejecting_lossy_setitem(prior_path):
    """``initialize_FID`` with pandas' lossy-setitem FutureWarning promoted to an error.

    ``unitconverter`` writes float64 conversion results into columns that
    ``safe_convert_to_numeric``'s ``downcast="float"`` left at float32. Where the
    value has no float32 representation that is a lossy setitem, which pandas <=
    2.3 performs anyway after widening the column itself -- warning only -- while
    pandas 3 raises ``TypeError: Invalid value ... for dtype 'float32'``.

    Promoting just that warning is what makes these tests bite on **every**
    supported pandas rather than only on pandas 3: without
    ``_widen_columns_that_cannot_hold`` the call below raises here too. The filter
    is matched on the message rather than the category so an unrelated future
    deprecation inside ``initialize_FID`` cannot turn these tests red.
    """
    import warnings

    import pyAMARES

    rc.quiet()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*incompatible dtype.*", category=FutureWarning
        )
        return pyAMARES.initialize_FID(
            fid=None, priorknowledgefile=prior_path, preview=False
        )


def test_non_float32_representable_phase_survives_unit_conversion():
    """A 180-degree prior phase must load on every pandas.

    Pinned on ``tests/Table1.csv``, the shipped example that declares two peaks at
    180 degrees; neither golden prior reaches this path, because both declare
    every phase as 0 or as an expression. The assert is on the exact float64
    value -- a silent float32 round trip would yield 3.1415927410125732.

    Note that Table1 states those phases under *equal* bounds, so
    ``generateparameter`` takes its ``lval == uval`` shortcut and reads the value
    from the object-dtype bounds frame, never from the converted column. That is
    why this test alone does not pin the widening (no-op the helper and it still
    passes on the value assert) and why the warning filter above, plus
    :func:`test_unitconverter_survives_a_lossy_float32_writeback`, are needed.
    """
    obj = _initialize_rejecting_lossy_setitem(os.path.join(rc.TESTS_DIR, "Table1.csv"))
    assert obj.initialParams["phi_Tau"].value == math.radians(180.0)
    assert obj.initialParams["phi_Tau2"].value == math.radians(180.0)


def test_unitconverter_survives_a_lossy_float32_writeback():
    """All three unit conversions must survive a float64 row over float32 columns.

    ``tests/priors/lossy_setitem.csv`` is built so that every one of
    ``unitconverter``'s writes is lossy for the ``Wide`` column: an amplitude that
    overflows float32 keeps the sibling ``Big`` column at float64, so each row
    cross-section is float64, and ``Wide`` carries a 180 degree phase under
    unequal bounds so the converted value is read back out of the column rather
    than out of the bounds frame.

    Every assert is on an exact float64 value, each of which a float32 round trip
    would visibly change (12.000000178813934 -> 12.0, 7*pi -> 21.991148, pi ->
    3.1415927410125732).
    """
    prior = os.path.join(rc.TESTS_DIR, "priors", "lossy_setitem.csv")
    obj = _initialize_rejecting_lossy_setitem(prior)

    # The stored cell is float32(0.1), widened -- not re-rounded -- on the way out.
    assert obj.initialParams["freq_Wide"].value == float(np.float32(0.1)) * 120.0
    assert obj.initialParams["dk_Wide"].value == 7.0 * np.pi
    assert obj.initialParams["phi_Wide"].value == math.radians(180.0)
    # Unequal bounds, i.e. the value really did come through the converted column.
    assert obj.initialParams["phi_Wide"].min < obj.initialParams["phi_Wide"].max


def test_hsvd_fit_on_a_prior_object_keeps_its_fitted_rows():
    """A fitted peak set that diverges from ``peaklist`` must not be reindexed away.

    ``report_amares`` reorders ``result_multiplets`` to the prior-knowledge peak
    order. Binding that ``reindex`` is only safe while the two label sets agree:
    on a divergent set it drops every fitted peak and injects an all-NaN row per
    unfitted prior name.

    Two supported workflows diverge. This one is ``amaresFit --use_hsvd``: a
    FIDobj built from a prior knowledge file -- so it *has* a ``peaklist`` -- is
    then fitted with HSVD-derived parameters whose peaks are named "1".."N". The
    two sets share no label at all, so an unguarded reindex empties the table
    completely.
    """
    import pyAMARES

    rc.quiet()
    fidobj = rc.initialize_example_fid()
    assert fidobj.peaklist, "the example FIDobj should carry a prior peaklist"

    hsvd_params = pyAMARES.HSVDinitializer(
        fid_parameters=fidobj,
        num_of_component=rc.HSVD_NUM_COMPONENTS,
        preview=False,
        verbose=False,
    )
    fitted = pyAMARES.fitAMARES(
        fid_parameters=fidobj,
        fitting_parameters=hsvd_params,
        method="least_squares",
        ifplot=False,
        inplace=False,
    )
    result = fitted.result_multiplets
    assert not set(result.index) & set(fidobj.peaklist), (
        "this test is only meaningful while the HSVD peak names and the prior "
        f"peak names are disjoint, got {list(result.index)!r}"
    )
    assert len(result) > 0, "the fitted table lost every row"
    assert result["amplitude"].notna().all(), (
        f"result_multiplets was reindexed against the prior peaklist:\n{result}"
    )
