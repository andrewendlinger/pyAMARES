# `pyamares-xmris` — divergence ledger

## What this package is

A repackage of [pyAMARES](https://github.com/hawkMRS/pyAMARES), published under a separate
name on PyPI so that its dependency metadata can be corrected. **The import name is
unchanged — you still write `import pyAMARES`.** Only the distribution name differs, the
same way `scikit-learn` is imported as `sklearn`.

- Upstream project: `HawkMRS/pyAMARES`
- Upstream license: BSD 3-Clause, Copyright (c) 2023-2025 Jia Xu, Magnetic Resonance
  Research Facility, University of Iowa. Preserved verbatim in [`LICENSE.txt`](LICENSE.txt).
- Upstream base for this release: **0.3.33**
- Last verified against upstream: **2026-07-21**

Please cite the original authors — this repackage adds nothing to cite:

> Xu, J.; Vaeggemose, M.; Schulte, R.F.; Yang, B.; Lee, C.-Y.; Laustsen, C.; Magnotta, V.A.
> PyAMARES, an Open-Source Python Library for Fitting Magnetic Resonance Spectroscopy Data.
> *Diagnostics* **2024**, *14*, 2668. https://doi.org/10.3390/diagnostics14232668

## Versioning policy

| Series | Meaning |
|---|---|
| `0.3.x` | Faithful repackage. Packaging metadata only; nothing under `pyAMARES/` is modified. |
| `0.3.x.postN` | Repackage-only re-release on identical upstream source. |
| `0.4.0+` | Behaviour diverges from upstream. Every difference gets a `D`-entry below. |

PyPI versions are immutable, so `0.3.33` remains permanently available as the
zero-divergence artifact even after later versions diverge. Cite that version if you need a
build that is provably upstream-equivalent.

## Install hazard

`pyamares-xmris` and upstream `pyAMARES` both install into `site-packages/pyAMARES/`.
Installing both into one environment silently overwrites files, and pip/uv will **not**
warn. Uninstalling either one afterwards removes files the other still claims. Install
exactly one.

---

# D — Divergences (shipped in 0.3.33)

Nothing under `pyAMARES/` is modified. All entries below are `setup.py` metadata.

## D1 — `hlsvdpro` declared with a PEP 508 marker

    Status:    shipped
    Commit:    cfb3a8c
    Symptom:   `pip install` fails on Apple Silicon and any non-x86_64 platform
    Cause:     hlsvdpro 2.0.0 ships x86_64-only wheels and no sdist, so there is
               nothing for pip to install or build; upstream declares it
               unconditionally via a build-time platform.machine() check, which
               bakes the builder's architecture into the published wheel
    Change:    hlsvdpro>=2.0.0; platform_machine == 'x86_64' or platform_machine == 'amd64'
    Rationale: pyAMARES does not require it. pyAMARES/util/hsvd.py falls back to the
               bundled pure-Python pyAMARES/libs/hlsvd.py. Cost is speed on the HSVD
               path only; no correctness cost.
    Upstream:  PR filed, maintainer unresponsive (issue #15)
    Evidence:  verified on arm64 — installs with no hlsvdpro present, hsvd resolves
               to pyAMARES.libs.hlsvd, documented example fit converges

## D2 — `numpy<2.0` and `pandas<2.2` caps

    Status:    shipped
    Commit:    8300ee5
    Symptom:   LossySetitemError and AttributeError on empty DataFrames during
               standard pyAMARES workflows
    Cause:     pandas tightened implicit-upcast rules on setitem; pyAMARES writes
               scalars into columns of differing dtype in PriorKnowledge.py
    Change:    pandas>=1.1.0,<2.2.0 and numpy>=1.18.1,<2.0.0
    Side effect: this is what caps the package at Python 3.12 — see C1
    Caveat:    the original failure was observed in an xmris workflow and has not
               been reduced to a minimal reproducer; the exact trigger is unconfirmed
    Update:    reduced in 0.4.0 — see D11. The LossySetitemError is real and its
               cause is as stated, but the threshold in this entry is wrong:
               pandas 2.1.4, which these caps resolve to, warns on exactly the
               same write. Only pandas 3.0 makes it fatal. The AttributeError
               half was never reproduced. The caps themselves are untouched here;
               revisiting them is a packaging decision.

## D3 — distribution renamed, attribution added

    Status:    shipped
    Change:    name="pyamares-xmris"; description, url and project_urls updated to
               attribute upstream; license="BSD-3-Clause" and license_files declared
               explicitly (previously conveyed only via the trove classifier, and the
               file is LICENSE.txt rather than bare LICENSE)
    Rationale: BSD attribution requirement; PyPI needs a free name

## D4 — `tests` excluded from the distribution

    Status:    shipped
    Symptom:   upstream installs a top-level `tests` package into site-packages,
               where it can shadow an unrelated `tests` module
    Cause:     tests/__init__.py exists and find_packages() had no exclude
    Change:    find_packages(exclude=["tests", "tests.*"])
    Evidence:  wheel top_level.txt now contains `pyAMARES` only

## D5 — Python 3.13/3.14 classifiers removed

    Status:    shipped
    Cause:     D2's caps resolve to numpy 1.26.4 / pandas 2.1.4, whose wheels stop
               at cp312; the classifiers advertised versions that cannot install
    Change:    dropped the 3.13 and 3.14 trove classifiers
    Not done:  python_requires still says >=3.8 — see C4

---

# D — Divergences (shipped in 0.4.0, unreleased)

The first entries that touch code under `pyAMARES/`. Their shared purpose is to
make the source correct under numpy 2, pandas ≥2.2 and pandas 3 **without moving
the shipped stack**: every entry below is verified against the frozen regression
corpus (`tests/goldens/`, captured on py3.12 / numpy 1.26.4 / pandas 2.1.4), and
every golden is byte-identical before and after. The dependency metadata in
`setup.py` still carries D2's caps; lifting them is a separate decision.

Verified stacks: py3.12 with pandas 2.1.4 / 2.2.3 / 2.3.3 under numpy 1.26.4, and
py3.13 / py3.14 with pandas 2.3.3 / 3.0.3 / 3.0.5 under numpy 2.5.2.

## D6 — `parse_bounds` accumulator frames declare `dtype=object`

    Status:    shipped in 0.4.0
    Symptom:   none observed
    Cause:     the two bound frames were built with the dtype left implicit, so
               it came from whatever pandas infers for an empty frame, while the
               loop fills them with a mixture of floats, NaN and raw strings
    Change:    pd.DataFrame(..., dtype=object) for df_lb and df_ub
    Rationale: truthful dtype; removes a dependence on inference that has
               changed across pandas majors
    Evidence:  measured object on pandas 2.1.4 / 2.2.3 / 2.3.3 / 3.0.3 both
               before and after, so this is a no-op today. Eight adversarial
               priors (no bounds section, bounds header only, all-string,
               mixed float/str/NaN, all-float, all-NaN, float-then-string,
               malformed) behave identically on all four.

## D7 — `extract_expr` works on an object-dtype frame

    Status:    shipped in 0.4.0
    Symptom:   pandas 3: TypeError: object of type 'float' has no len(), from lmfit
    Cause:     PDEP-14 makes a text column the `str` dtype, where the None that
               process_expression returns for a non-string cell is stored as a
               missing value and read back as float NaN — defeating both that
               `return None` and the `is None` check in process_df_corrected
    Change:    deepcopy(pk.iloc[1:6]).astype(object)
    Evidence:  no-op on pandas ≤2.3, where the frame is already object
    Resolves:  the break recorded as C2

## D8 — a non-string `expr` is normalised to `None`

    Status:    shipped in 0.4.0
    Symptom:   as D7, at the lmfit call site in generateparameter
    Cause:     df_expr[peak].iloc[i] can yield NaN under pandas 3; lmfit accepts
               only a string expression or None
    Change:    if not isinstance(expr, str): expr = None
    Evidence:  no-op on pandas ≤2.3 — only str and None ever reached it there
    Resolves:  the break recorded as C2, with D7

## D9 — bounds are tested with `pd.isna`, not `np.isnan`

    Status:    shipped in 0.4.0
    Symptom:   TypeError: ufunc 'isnan' not supported for the input types,
               whenever a bound survives safe_convert_to_numeric as a string
    Cause:     np.isnan rejects strings and None
    Change:    np.isnan(lval)/np.isnan(uval) -> pd.isna(...)
    Behaviour: identical for floats. A non-numeric string bound — invalid input
               either way — now fails further downstream with a different error
               instead of at the isnan call.

## D10 — `uniquify_dataframe` selects the grouping column explicitly

    Status:    shipped in 0.4.0
    Symptom:   pandas 3: every HSVD component silently loses its name and
               HSVDinitializer returns an empty Parameters object; downstream
               KeyError: 'ak'
    Cause:     pandas 2.2 deprecated and pandas 3 removed passing the grouping
               column into DataFrameGroupBy.apply, so `group.loc[..., "name"]`
               stops assigning and starts *creating* an all-NaN column
    Change:    .groupby("name", group_keys=False)[list(df_named.columns)].apply(...)
    Evidence:  bit-identical output on pandas 2.1.4 / 2.2.3 / 2.3.3, verified
               side by side; also silences the pandas 2.2+ FutureWarning

## D11 — `unitconverter` widens float32 columns that cannot hold a conversion

    Status:    shipped in 0.4.0
    Symptom:   pandas 3: TypeError: Invalid value '3.141592653589793' for dtype
               'float32' — this is D2's LossySetitemError
    Cause:     safe_convert_to_numeric calls pd.to_numeric(..., downcast="float"),
               so numeric prior cells are float32, while every unit conversion
               produces float64. np.deg2rad(180) has no float32 representation,
               and neither has `shift * MHz` or `linewidth * pi` whenever a
               sibling column stays float64/object and promotes the row.
    Change:    each of the three conversions computes its values at the row's own
               dtype, widens to float64 only the columns that cannot hold the
               result (_widen_columns_that_cannot_hold), then writes
    Rationale: that is precisely what pandas ≤2.3 did on its own. Widening
               unconditionally instead moves chemicalshift and linewidth to
               float64 arithmetic and shifts fitted amplitudes by ~1.3e-3
               relative — measured, and rejected for that reason.
    Threshold: pandas 2.1.4, the version D2's cap resolves to, emits the same
               FutureWarning as 2.2 and 2.3. The <2.2 cap never avoided this
               path; it only sat upstream of the release that made it fatal.
    Evidence:  tests/Table1.csv (two peaks at 180°) raised on pandas 3 before
               this. Pinned by two tests on Table1.csv and
               tests/priors/lossy_setitem.csv, both run with pandas'
               incompatible-dtype FutureWarning promoted to an error so they
               fail on every supported pandas if the widening is removed.
    Reduces:   D2's "not been reduced to a minimal reproducer" caveat

## D12 — `fircls1`'s scipy version gate compares numbers

    Status:    shipped in 0.4.0
    Symptom:   on scipy 1.2–1.9, fircls1 takes the ≥1.14 firls branch and raises
    Cause:     `scipy.__version__ >= "1.14.0"` is a lexicographic comparison, and
               "1.9.0" sorts after "1.14.0"
    Change:    compare (major, minor) as an int tuple, assuming a version string
               with a non-numeric component is modern
    Evidence:  no behaviour change on any scipy ≥1.14; checked against eleven
               version strings including rc and dev suffixes

## D13 — `util/hsvd` imports `scipy.optimize` explicitly

    Status:    shipped in 0.4.0
    Symptom:   none currently
    Cause:     scipy.optimize.curve_fit is used but only `import scipy` was
               present. It resolves through lmfit's side-effect import, and on
               scipy ≥1.9 through scipy's own lazy submodule __getattr__ —
               neither of which holds across the declared scipy>=1.2.1 floor.
    Change:    import scipy.optimize

## D14 — `report_amares` applies its peaklist reindex, under a set guard

    Status:    shipped in 0.4.0
    Symptom:   the documented "reorder to the prior-knowledge peak order" never
               happened: the reindex return value was discarded
    Cause:     `result.reindex(fid_parameters.peaklist)` called for effect
    Change:    bind the result, but only when set(peaklist) == set(result.index)
    Rationale: binding it unconditionally is destructive on a divergent label
               set — it drops every fitted peak absent from peaklist and injects
               an all-NaN row per unfitted prior name. Two supported workflows
               diverge: fitting a prior-built FIDobj with HSVD-derived parameters
               (peaks named "1".."N", i.e. `amaresFit --use_hsvd`) shares no label
               at all and empties the table, and filter_param_by_ppm drops the
               peaks outside the fitted window.
    Evidence:  no-op on every corpus case — all goldens byte-identical. Both
               divergent paths produce output identical to the pre-0.4.0
               discarded-reindex behaviour, verified side by side, and the HSVD
               one is pinned by a regression test.

## D15 — the vendored `hlsvdpropy` copy carries its attribution

    Status:    shipped in 0.4.0
    Symptom:   pyAMARES/libs/hlsvd.py shipped with no copyright or license notice
    Cause:     the file is a vendored copy of hlsvdpropy/hlsvd.py from hlsvdpropy
               2.0.2 (the only PyPI release, 2023-07-24), Copyright (c) 2020
               Brian J Soher, BSD 3-Clause — whose clause 1 requires source
               redistributions to retain the notice, conditions and disclaimer
    Change:    a header giving the upstream project, author, copyright, license,
               why pyAMARES vendors it, and the three local modifications
               relative to 2.0.2 (np.mat -> .copy() for numpy 2, np.complex128 ->
               complex, an explicit return in create_hlsvd_fids) plus the
               whole-file reformat; a matching entry in docs/source/license.rst
    Note:      the full license text is reproduced inline in the header rather
               than as a sibling LICENSE file, because MANIFEST.in ships no
               non-Python files from pyAMARES/libs/ and a sibling .txt does not
               reach the wheel (verified with `uv build` + `unzip -l`). The
               header does reach both wheel and sdist.
    Evidence:  token-level diff against the 2.0.2 wheel: 97.4% identical,
               algorithm unchanged
    Rationale: BSD attribution requirement; no code change

## Considered and deliberately not changed

`kernel/lmfit.py: load_parameter_from_csv` — `df.where(pd.notnull(df), None)` was
a candidate for the same PDEP-14 treatment as D7, but measurement refuted the
need and exposed a cost. On pandas 3 that line already yields NaN for a missing
`expr`, which is exactly what `dataframe_to_parameters` tests for with `pd.isna`;
adding `.astype(object)` first would instead turn a blank *numeric* cell in a
hand-edited `params.csv` into a real None, which lmfit reads as `min=-inf`/
`max=+inf` and which clamps a None value to the lower bound. Verified identical
on pandas 2.1.4, 2.3.3, 3.0.3 and 3.0.5. The line is unchanged; only a comment
recording this was added.

---

# C — Candidates (undecided)

Not commitments. Recorded so the cost is known when the question comes up.

## C1 — numpy 2.x support (would lift the Python ceiling)

    Status:    blocked on a third party
    Blocker:   nmrglue 0.11 (last PyPI release, 2024-10-29) uses np.dtype('a8'),
               removed in numpy 2.0 -> import-time TypeError
    Upstream:  already fixed on nmrglue master (np.dtype('S8'), commit 5a6c58e),
               but unreleased. A published wheel cannot depend on a git ref, so
               this cannot be worked around in packaging alone.
    pyAMARES:  clean. Zero numpy-2 removed-alias usages in the source; the
               pandas applymap/map calls are already version-guarded.
    Notes:     pyAMARES uses nmrglue for exactly one function, ng.proc_base.fft,
               at 6 call sites (util/hsvd.py:64,74; util/visualization.py:44,48,100,101).
               That function is a one-line numpy wrapper:
                 np.fft.fftshift(np.fft.fft(data, axis=-1).astype(data.dtype), -1)
               So the dependency, and this entire blocker, exists for one line.
    Options:   (a) ask nmrglue to cut a release — free if it lands, unbounded wait
               (b) drop nmrglue and inline the FFT — ~half a day incl. validation,
                   but ends the "zero source changes" property and adds sync burden
    Evidence:  with nmrglue installed from git, Python 3.13 and 3.14 both work —
               see "Compatibility matrix" below

## C2 — pandas 3.x support

    Status:    source addressed in 0.4.0; the dependency metadata is not
    Symptom:   TypeError: object of type 'float' has no len()
    Cause:     pandas 3.0 defaults string columns to the `str` dtype (PDEP-14), where
               missing values read back as NaN rather than None. The `expr` read in
               PriorKnowledge.generateparameter took such a column and handed a
               float to lmfit.
    Addressed: D7 and D8 fix that read. Two further breaks surfaced behind it, as
               this entry's estimate warned they might: D10 (HSVDinitializer
               silently returned no parameters) and D11 (a lossy float32 setitem,
               which turned out to be D2's LossySetitemError and is fatal only
               from pandas 3.0). The regression corpus and all six example
               notebooks now pass on pandas 3.0.3 and 3.0.5 under numpy 2.
    Not done:  setup.py still declares pandas<2.2 — see D2. Whether to lift that
               cap, and to what floor, is a packaging decision.
    Note:      independent of C1 — pandas 2.3 works fine under numpy 2

## C3 — `sd` columns are not reproducible across dependency versions

    Status:    open, needs a domain decision
    Symptom:   sd, sd(ppm), sd(Hz), sd(deg) differ by up to 238% between dependency
               sets on identical input; some values come out negative
    Cause:     util/crlb.py:56-68 inverts a Fisher matrix that pyAMARES itself flags
               as ill-conditioned, via scipy.linalg.pinv or np.linalg.lstsq. Their
               rank cutoffs shift between LAPACK/scipy builds.
    Not a regression: pre-existing fragility. The current shipped stack produces
               sd(Hz) = 47107 on the documented example.
    Unaffected: every fitted parameter and every CRLB column agrees to ~1e-7
    Blocks:    this, not the code, is the real cost of C1/C2 — bumping deps needs a
               regression corpus and a judgement on whether `sd` is trusted at all
    Open question: does xmris consume the `sd` column? If not, the validation
               burden largely disappears.

## C4 — `python_requires` does not reflect the real ceiling

    Status:    open, trivial
    Symptom:   on Python 3.13+, pip attempts a from-source numpy 1.26 build and fails
               with a compiler error instead of a clean "requires Python <3.13"
    Change:    python_requires=">=3.8,<3.13" while D2 stands
    Not done:  tightening it changes upstream behaviour, which 0.3.x avoids by policy.
               Revisit at 0.4.0, or drop entirely if C1 lands.

---

# Compatibility matrix

Documented example fit (`pyAMARES/examples/fid.txt` +
`example_human_brain_31P_7T.csv`), verified 2026-07-21 on macOS arm64.

| Python | numpy | pandas | scipy | nmrglue | Result |
|---|---|---|---|---|---|
| 3.12 | 1.26.4 | 2.1.4 | 1.17.1 | 0.11 | **ships today** — fit converges |
| 3.13 | 2.5.1 | 3.0.3 | 1.18.0 | git | fails — C2 |
| 3.13 | 2.5.1 | 2.3.3 | 1.18.0 | 0.11 | fails — C1 |
| 3.13 | 2.5.1 | 2.3.3 | 1.18.0 | git | fit converges |
| 3.14 | 2.5.1 | 2.3.3 | 1.18.0 | git | fit converges |

Cross-checking the last row against the first: all fitted parameters and all CRLB columns
agree to ~1e-7 (float roundoff); only the `sd` columns diverge (C3).

Caveat: this is one example fit, not a regression suite. Treat it as evidence that the
blockers are correctly identified, not as validation that numpy 2 is safe for production use.
