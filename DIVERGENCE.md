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
- Last verified against upstream: **2026-08-18** — `hawkMRS/pyAMARES@145f817` is still the
  upstream tip and is an ancestor of this branch, so every difference is one listed below.

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

`0.4.0` is the first diverging release, and divergence is the accepted long-term direction:
upstream is tracked and cherry-picked, not followed. PyPI versions are immutable, so `0.3.33`
remains permanently available as the zero-divergence artifact even after later versions
diverge. Cite that version if you need a build that is provably upstream-equivalent.

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
    Update:    in 0.5.0, D18 moves this requirement — marker and all, verbatim —
               out of install_requires and into an extra of its own, `hlsvd`. The
               default install no longer mentions hlsvdpro on any platform. This
               entry's outcome is unchanged (a clean install on arm64); what
               changed is that x86_64 no longer gets, by default, a binary that
               only Python 3.8 environments can import — setuptools >= 82 removed
               the pkg_resources that hlsvdpro 2.0.0 imports at module scope, and
               3.8 is the last Python whose setuptools stays below that. Users on
               x86_64 py3.8 with numpy 1.x, the one configuration where the
               dependency was ever live, opt back in with
               `pip install 'pyamares-xmris[hlsvd]'`. util/hsvd.py is untouched
               and still binds a user-installed hlsvdpro under numpy<2.

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
               half was never reproduced. The caps themselves are untouched by
               D11; they are lifted in D16, which supersedes this entry.

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
    Update:    reversed in 0.4.0. D16 lifts the caps and restores both
               classifiers; python_requires stays >=3.8, now on evidence.

---

# D — Divergences (shipped in 0.4.0)

D6–D15 are the first entries that touch code under `pyAMARES/`. Their shared
purpose is to make the source correct under numpy 2, pandas ≥2.2 and pandas 3
**without moving the shipped stack**: every one of them is verified against the
frozen regression corpus (`tests/goldens/`, captured on py3.12 / numpy 1.26.4 /
pandas 2.1.4), and every golden is byte-identical before and after. D16 is the
packaging half — with the source correct and nmrglue 0.12 published, it lifts
D2's caps and lets the new stack actually be installed.

Verified stacks: py3.12 with pandas 2.1.4 / 2.2.3 / 2.3.3 under numpy 1.26.4, and
py3.13 / py3.14 with pandas 2.3.3 / 3.0.3 / 3.0.5 under numpy 2.5.2. See the
compatibility matrix at the end for the post-D16 resolutions.

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

## D16 — the dependency ceilings come off; nmrglue is floored at 0.12

    Status:    shipped in 0.4.0
    Symptom:   `pip install pyamares-xmris` on Python 3.13+ tried to build numpy
               1.26 from source and failed with a compiler error; on any Python
               it pinned the environment to numpy 1.x / pandas 2.1, which no
               longer matched what the source (D6-D15) supports
    Cause:     D2's numpy<2.0.0 / pandas<2.2.0 caps, and C1's blocker underneath
               them — nmrglue 0.11 used np.dtype('a8'), removed in numpy 2, and
               pyAMARES imports nmrglue eagerly, so the failure was fatal at
               `import pyAMARES` rather than at the one function it needs
    Change:    numpy>=1.18.1,<2.0.0        -> numpy>=1.18.1
               pandas>=1.1.0,<2.2.0        -> pandas>=1.1.0
               nmrglue                     -> nmrglue>=0.12
               classifiers: Python 3.13 and 3.14 restored
               python_requires: unchanged at >=3.8 (see below)
               description: no longer claims "zero algorithm changes"; it now
               says minimal, ledger-documented compatibility fixes and points at
               this file. README.rst's opening note reworded to match.
               requirements.txt: emptied to a comment. It duplicated the
               dependency list without versions, so it could only ever
               contradict setup.py; the file is kept so upstream merges stay
               quiet.
    Floors:    only the ceilings are removed. Every floor, and every other
               dependency, is untouched — dependency slimming is deliberately
               out of scope, see C5.
    hlsvdpro:  its D1 marker is unchanged and needs no python_version guard.
               hlsvdpro 2.0.0's py38-none wheels resolve on cp313 and cp314 for
               x86_64 manylinux and Windows (uv dry-run, 2026-08-17).
    python_requires: stays ">=3.8". The cap C4 proposed is no longer needed, and
               the floor was tested rather than assumed: with the ceilings off,
               Python 3.8 resolves to numpy 1.24.4 / pandas 2.0.3 / scipy 1.10.1
               / nmrglue 0.12, and all 53 corpus tests pass there. nmrglue 0.12
               declares no python_requires of its own and still classifies 3.8;
               its source compiles clean under a 3.8 interpreter. So no 3.8 or
               3.9 classifier is trimmed.
    Evidence:  nmrglue 0.12 (PyPI, 2026-08-16) carries the fix — the wheel's
               fileio/tecmag.py reads `TNTMAGIC = np.dtype('S8')`, and 0.12
               imports fine on numpy 1.26.4, so the floor costs old-stack users
               nothing. The 53-test corpus is green on py3.8/3.9/3.12/3.13/3.14
               under the resolutions in the compatibility matrix, and still green
               on the frozen baseline (py3.12 / numpy 1.26.4 / pandas 2.1.4) with
               nmrglue 0.11 force-installed and with 0.12 — identical goldens
               either way, so bumping the floor is not itself a numeric change.
               The old stack also still resolves through the published metadata:
               `--with . --with 'numpy<2' --with 'pandas<2.2'` gives numpy 1.26.4
               / pandas 2.1.4, i.e. dropping the ceilings did not raise a floor.
    Resolves:  C1, C4; completes C2

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

# D — Divergences (staged for 0.5.0)

## D17 — the heavy imports are deferred to first use

    Status:    staged for 0.5.0 (unreleased)
    Symptom:   `import pyAMARES` eagerly loaded nmrglue, matplotlib.pyplot and
               mat73. Two costs. An import failure in any of them was fatal to
               the whole package rather than to the one function that needs it —
               exactly the incident D16 describes, where nmrglue 0.11 under
               numpy 2 made `import pyAMARES` raise a TypeError even though only
               a one-line FFT wrapper was involved. And every consumer paid the
               full import cost of a plotting stack, an NMR file-format library
               and a HDF5 MATLAB reader for a single headless fitAMARES call.
    Change:    the module-level imports move into the function bodies that use
               them, following the precedents already in the source (the delayed
               crlb import in util/report.py, the IPython import in
               kernel/PriorKnowledge.py, the tqdm import in
               util/multiprocessing.py, the nibabel import in
               fileio/readnifti.py). Nine files:
                 kernel/fid.py           nmrglue -> fft_params, process_fid,
                                         simulate_fid; pyplot -> the ifplot /
                                         preview blocks
                 kernel/objective_func.py nmrglue -> objective_range's fit_range
                                         branch
                 kernel/PriorKnowledge.py pyplot -> initialize_FID's preview block
                 libs/MPFIR.py           pyplot + nmrglue -> MPFIR's ifplot block
                 util/visualization.py   nmrglue -> preview_HSVD, plot_fit;
                                         pyplot -> preview_HSVD, combined_plot
                 util/hsvd.py            nmrglue -> HSVDp0; pyplot -> HSVDp0's
                                         ifplot block and HSVDinitializer's
                                         preview block
                 util/crlb.py            pyplot -> evaluateCRB's verbose block and
                                         create_pmatrix's ifplot block
                 fileio/readfidall.py    mat73 -> read_fidall's v7.3 branch
                 fileio/readmat.py       mat73 -> readmrs' v7.3 branch
               pyplot always lands in the narrowest plotting block. util/crlb.py
               is the one that makes that worth anything: every fitAMARES call
               runs report_amares, which reaches crlb through util/report.py's
               delayed import, so leaving pyplot at crlb's module scope would
               have kept the whole plotting stack on the critical path of a
               headless fit.
               nmrglue lands at the top of the function body, except in
               fft_params, where it sits *below* the return_mat / fid=True early
               returns: those two branches are the ones a fit takes (kernel/
               lmfit.py:487,504,548,559 and PriorKnowledge.py:665 all pass
               fid=True or return_mat=True), so the import must not sit above
               them. With that placement a default headless fit imports no
               nmrglue at all — measured, see Behaviour.
    Not moved: util/report.py's `try: import jinja2` probe — it is a core
               dependency and its result feeds the module-level `if_style` flag
               that the styled-table path reads.
               util/crlb.py's sympy — it *is* the CRLB algebra (create_pmatrix
               parses the prior-knowledge expressions with sympy_parser), so no
               call into that module does not want it, and the module itself is
               only ever reached at first-report time. Measured: a default
               headless fit loads sympy and nothing else from the guarded list.
               util/hsvd.py's backend selection (hlsvdpro vs the vendored
               pyAMARES/libs/hlsvd) — which backend the module binds is an
               import-time rule, and that rule is what
               test_hsvd_backend_selection pins. A comment above it says so.
               lmfit's own module-scope `try: import matplotlib` (lmfit.model's
               _HAS_MATPLOTLIB), which is why the guard below forbids
               matplotlib.pyplot rather than bare matplotlib: the bare package
               loads on `import pyAMARES` no matter what this repo does, and the
               expensive half is pyplot.
    Behaviour: numerics identical — the corpus is green and every golden is
               byte-identical, on the frozen py3.12 / numpy 1.26.4 /
               pandas 2.1.4 stack and on an unpinned py3.13 resolution. Three
               timing differences, all deliberate:
                 1. an import error in nmrglue / matplotlib / mat73 now surfaces
                    at the first call that needs it, not at `import pyAMARES`
                 2. MPLBACKEND, or matplotlib.use(), set *after* `import
                    pyAMARES` is now honoured — previously pyplot had already
                    chosen a backend by then
                 3. incidental module attributes are gone: pyAMARES.kernel.fid.plt
                    and .ng, kernel.objective_func.ng, kernel.PriorKnowledge.plt,
                    util.visualization.plt/.ng, util.crlb.plt, and the same on
                    libs/MPFIR.py, util/hsvd.py, fileio/readfidall.py,
                    fileio/readmat.py. None is public API and none is in
                    __all__, but a consumer reaching through the module
                    namespace for them breaks.
               Measured, py3.13 / numpy 2.5.2 / pandas 3.0.5: `import pyAMARES`
               loads none of the guarded modules, and a default headless
               fitAMARES adds only sympy (see Not moved). On the frozen py3.12
               stack the same fit also ends up with matplotlib.pyplot — imported
               by *pandas*, not by pyAMARES: pandas 2.1.4's
               pandas/io/formats/style.py has a module-scope `import
               matplotlib.pyplot as plt`, and report_amares touches
               `DataFrame.style`. pandas 3 dropped that, and nothing this repo
               can do would have avoided it.
               Per-call cost: kernel/objective_func.objective_range is a fit-loop
               callback when fit_range is set, so its import runs once per
               residual evaluation — a resolved import is one sys.modules dict
               lookup, negligible against the FFT on the next line.
    Guarded:   two tests in tests/test_api_surface.py, deliberately overlapping.
               test_bare_import_keeps_heavy_modules_unloaded imports pyAMARES in
               a child process (handed this process' sys.path, so it resolves the
               same package pytest did) and asserts that none of
               matplotlib.pyplot, nmrglue, mat73, sympy, IPython, tqdm, requests,
               xlrd, openpyxl or nibabel is in sys.modules afterwards — a fixed
               blocklist, with the better failure message.
               test_no_undocumented_module_level_third_party_imports is the
               open-ended half: it AST-scans pyAMARES/**/*.py and fails on any
               module-scope import of a distribution outside
               {numpy, scipy, pandas, lmfit, jinja2} plus the two documented
               per-file exceptions (hlsvdpro in util/hsvd.py, sympy in
               util/crlb.py). pyAMARES/script/ is excluded — nothing in the
               package imports it, and amaresfit_gui.py carries streamlit,
               requests and matplotlib at module scope by design. So an upstream
               cherry-pick that adds a module-level import of something nobody
               has thought of yet fails here rather than shipping.
    Enables:   C5. The import graph is now verifiable, which is the cost that
               entry names as the blocker for moving dependencies behind extras.

## D18 — the runtime dependency list is slimmed to what the package imports

    Status:    staged for 0.5.0 (unreleased)
    Symptom:   `pip install pyamares-xmris` resolved 62 packages on py3.13 for a
               library whose own imports are numpy/scipy/pandas/matplotlib/
               lmfit/sympy/nmrglue. A Jupyter stack (ipython, ipykernel, the two
               ipywidgets marker lines), an HTTP client (requests) and two file
               readers (mat73, xlrd) were installed into every environment that
               only wanted to fit spectra. hlsvdpro was worse than unused on
               almost every stack: on x86_64 it installed a binary wheel that
               cannot import — hlsvdpro 2.0.0 does `import pkg_resources` at
               module scope and setuptools >= 82 removed pkg_resources, so
               util/hsvd.py's except-ImportError silently bound the vendored
               backend anyway (C5's 2026-08-18 update, found wiring the
               regression CI and verified on real linux/amd64: it imports only
               with `setuptools<82` force-installed). And openpyxl, which pandas
               needs to read the .xlsx priors this package documents as its
               primary input format, was never declared at all — a clean install
               could read a CSV prior and not an Excel one (issue #12).
    Change:    install_requires is now exactly, and only, what the package
               imports:
                 numpy>=1.18.1, scipy>=1.2.1, pandas>=1.1.0,
                 matplotlib>=3.1.3, lmfit, sympy, nmrglue>=0.12, jinja2, tqdm
               Demoted to extras: ipython, ipykernel, ipywidgets (both marker
               lines), requests, mat73, xlrd. Newly declared: openpyxl, behind
               the same extra as xlrd — that is the fix for issue #12.
               Demoted out of the default install, to an extra of its own:
               hlsvdpro, carrying D1's marker unchanged. It is in no other extra
               — not jupyter, not dev — because it is a niche numeric opt-in, and
               because on nearly every environment it is inert: it cannot import
               under setuptools >= 82, and util/hsvd.py never looks for it under
               numpy >= 2. There is exactly one live configuration left, and it
               is the reason the extra exists rather than nothing at all:
               **x86_64 Python 3.8 with numpy 1.x**. setuptools stopped
               supporting 3.8 after the 75.x line, so a 3.8 environment resolves
               setuptools < 82, pkg_resources is present, hlsvdpro imports, and
               util/hsvd.py binds it in preference to the vendored copy. Such a
               user upgrading 0.4.0 -> 0.5.0 therefore *does* change HSVD
               backend unless they install `pyamares-xmris[hlsvd]`; that is the
               documented opt-back-in. On any modern stack the extra installs a
               wheel that raises on import and util/hsvd.py's except-ImportError
               steps over it — harmless, and no worse than 0.4.0 was.
               Extras layout (plain list concatenation, not self-referential
               extras, so the metadata stays readable on old pip):
                 matlab  = mat73
                 excel   = openpyxl, xlrd
                 hlsvd   = hlsvdpro>=2.0.0 (x86_64/amd64 marker, D1)
                 jupyter = notebook, ipykernel, ipython, ipywidgets (markers),
                           requests + matlab + excel
                 docs    = unchanged
                 ruff    = unchanged
                 dev     = jupyter + docs + ruff
               `jupyter` also picks up `notebook`, which the old extra declared
               and install_requires did not, and it no longer declares ipykernel
               twice (C5's "the extra and the runtime dependency disagree about
               whose job it is").
    Behaviour: a bare install now *fails* on two paths it used to serve, with a
               message naming the extra rather than a bare ModuleNotFoundError:
                 fileio/readmat.py::readmrs and fileio/readfidall.py::read_fidall
                 — the v7.3 (HDF5) .mat branch, chained from mat73's
                 ModuleNotFoundError: "Reading MATLAB v7.3 .mat files requires
                 mat73. Install it with: pip install 'pyamares-xmris[matlab]'".
                 Traditional .mat files go through scipy.io and are unaffected.
                 kernel/PriorKnowledge.py::generateparameter — the .xlsx/.xls
                 branch, chained from pandas' own "Missing optional dependency"
                 ImportError, naming `pyamares-xmris[excel]` and pointing at CSV
                 as the no-extra way out.
               Both guards are deliberately narrow about *which* failure they
               reinterpret. The mat73 ones catch ModuleNotFoundError only, so an
               installed-but-broken mat73 (an h5py ABI mismatch, say) keeps its
               own message instead of being misreported as a missing extra. The
               Excel one has to catch ImportError, because that is what pandas
               raises for a missing *and* for a too-old engine — so its message
               says "missing or too old" and offers upgrade as well as install.
               D17 is what makes those the only two: it moved every heavy import
               into the function body that needs it *first*, so no demotion can
               turn into an import-time failure of the whole package. That
               ordering is the entire safety argument for this entry.
               tqdm stays a hard dependency, so the progress bars stay — but the
               notebook bar needed a code change, added here. `tqdm.notebook`
               does *not* degrade on its own: without ipywidgets it imports
               cleanly and then raises ImportError("IProgress not found...") from
               `status_printer` when the first bar is constructed (measured on
               tqdm 4.70.0). In util/multiprocessing.py that construction happens
               inside `run_parallel_fitting_with_progress` — public API, exported
               top level, `notebook=True` by default — after the process pool has
               already been filled, so demoting ipywidgets would have crashed a
               parallel fit that worked in every earlier release. The selection
               now lives in `_select_tqdm`, which reads tqdm's own `IProgress`
               flag and falls back to the text bar with a warning naming
               `pyamares-xmris[jupyter]`. IPython was already imported
               defensively in PriorKnowledge.py. requests is imported only by
               script/amaresfit_gui.py, which nothing in the package imports.
    Restores:  `pip install 'pyamares-xmris[jupyter]'` is a superset of the 0.4.0
               install minus hlsvdpro, plus notebook and openpyxl; add `,hlsvd`
               and it is a superset of 0.4.0 outright on x86_64 — which is where
               0.4.0 installed hlsvdpro at all. Nobody who installs
               `[jupyter,hlsvd]` loses anything.
    Guarded:   eight tests in tests/test_api_surface.py, all of which simulate the
               absent dependency rather than requiring it to be absent, so they
               assert the same thing on a bare install and under [jupyter]:
                 - both v7.3 branches with mat73 made unimportable via
                   `sys.modules["mat73"] = None`, asserting the extra's name and
                   that the ModuleNotFoundError survives as `__cause__`
                 - a *broken* mat73 (an ImportError that is not a
                   ModuleNotFoundError, injected through builtins.__import__),
                   asserting its own message reaches the caller untouched
                 - the Excel branch with pandas' read_excel monkeypatched to raise
                   the real "Missing optional dependency 'openpyxl'" ImportError
                 - four on `_select_tqdm`: the two fallback routes (tqdm's
                   IProgress set to None, and tqdm.notebook made unimportable),
                   each asserting the text class is chosen and that the warning
                   names the extra; that the notebook class is still chosen when
                   ipywidgets is usable; and that notebook=False is honoured
               CI: `.github/workflows/test-install.yml` asserts after a plain
               `pip install .` that none of IPython, ipykernel, ipywidgets,
               requests, mat73, xlrd, openpyxl, hlsvdpro or notebook resolves
               through importlib.util.find_spec, and then that a bare `import
               pyAMARES` still loads none of the D17 blocklist. A new
               `test-extras` job (py3.8 and py3.13) installs `.[jupyter]` and
               imports everything the extra promises. test-notebooks.yml installs
               `-e ".[jupyter]"` — step1_download.ipynb uses requests.
    Evidence:  measured on py3.13 by the same method on both sides
               (importlib.metadata.distributions() in a fresh resolution):
               64 distributions for 0.4.0 from PyPI, 27 for this branch —
               Jinja2, MarkupSafe, asteval, certifi, contourpy, cycler, dill,
               fonttools, kiwisolver, lmfit, matplotlib, mpmath, nmrglue, numpy,
               packaging, pandas, pillow, pip, pyamares-xmris, pyparsing,
               python-dateutil, scipy, six, sympy, tqdm, uncertainties, wheel.
               The corpus (55 tests + 9 new ones, of which one skips where
               ipywidgets is absent) is green on that bare py3.13 install, with
               all nine demoted names confirmed absent via
               importlib.util.find_spec, and on the frozen golden stack (py3.12 /
               numpy 1.26.4 / pandas 2.1.4) — 63 passed, 1 skipped either way,
               every golden untouched. All six example notebooks pass under
               `.[jupyter]` (93 cells, step1_download's requests included).
               `.[hlsvd]` resolves on arm64 and correctly installs nothing (the
               marker), leaving util/hsvd.py on the vendored backend. The
               published metadata was read back out of the built wheel's
               METADATA, not just off setup.py.
    Resolves:  C5
    Supersedes: D1's placement, not its content — the marker moves verbatim from
               install_requires to the `hlsvd` extra. The arm64 install problem D1
               fixed stays fixed, now twice over: the marker still excludes arm64,
               and the default install does not mention hlsvdpro at all.

---

# C — Candidates (undecided)

Not commitments. Recorded so the cost is known when the question comes up.

## C1 — numpy 2.x support (would lift the Python ceiling) — RESOLVED in 0.4.0

    Resolution: option (a) happened. nmrglue 0.12 was released to PyPI on
               2026-08-16 carrying exactly the fix this entry was waiting on
               (np.dtype('a8') -> np.dtype('S8')), so no source change and no
               inlined FFT was needed — D16 floors the dependency at 0.12 and
               drops the numpy ceiling. nmrglue remains in use — and note the
               historical Notes below undercount it: besides ng.proc_base.fft
               (12 call sites across kernel/objective_func.py, kernel/fid.py,
               util/hsvd.py, util/visualization.py, libs/MPFIR.py), pyAMARES
               also calls ng.proc_base.em (kernel/fid.py, 2 sites), so a future
               "drop nmrglue" (option b) must inline both wrappers, not one.
               The Python ceiling this entry describes is gone with it: 3.13 and
               3.14 are supported, classified and corpus-verified, and the
               package no longer pins anyone to numpy 1.x.

    Status:    resolved by D16 — history below
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

## C2 — pandas 3.x support — RESOLVED in 0.4.0

    Resolution: fully resolved. D7 and D8 fix the read this entry describes,
               D10 and D11 fix the two breaks that surfaced behind it, and D16
               drops the pandas<2.2 cap so the supported source can actually
               meet a supported pandas. A fresh install on py3.12-3.14 now
               resolves to pandas 3.0.5, with the corpus green.

    Status:    resolved by D7/D8/D10/D11 + D16 — history below
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
               (Done in D16: the cap is gone, the floor stays at 1.1.0.)
    Note:      independent of C1 — pandas 2.3 works fine under numpy 2

## C3 — `sd` columns are not reproducible across dependency versions — CLOSED by policy

    Resolution: closed as a documented limitation, not fixed. The four `sd`
               columns — sd, sd(ppm), sd(Hz), sd(deg) — are **not validated
               across dependency stacks**, and 0.4.0 ships saying so. Nothing in
               util/crlb.py is touched: the ill-conditioning is upstream's
               numerics, and "improve" here means changing fitted uncertainties,
               which is a domain decision no packaging release should make.
    Guarded:   structurally only, never golden-compared, and
               test_goldens_never_freeze_the_sd_columns fails the suite if a
               re-capture quietly starts freezing them. What is asserted:
                 - every value finite, strictly positive, and sd < amplitude
                 - the *measured* proportionality sd = k*(CRLB/100)*|value| with
                   one fit-wide k (k is not 1 — report_amares fills sd from
                   lmfit's stderr and CRLB from the Fisher matrix separately)
                 - monotone growth with the noise scale
    Locked:    every fitted parameter and every CRLB column is regression-locked
               against the frozen goldens at rtol 1e-4.
    Open question, answered: yes, xmris consumes all four. fitting/amares.py maps
               each of amplitude/chem_shift/linewidth/phase to its (sd, CRLB%)
               pair and reads all eight columns. But it asserts only relative
               properties of them, never an absolute sd value, so the columns
               being stack-dependent does not make xmris stack-dependent — and
               the validation burden this entry feared does not materialise.
    Status:    closed; revisit only if someone needs trustworthy absolute sd

## C4 — `python_requires` does not reflect the real ceiling — RESOLVED in 0.4.0

    Resolution: resolved by D16 without the change this entry proposed. The
               ">=3.8,<3.13" cap existed only to describe D2's caps honestly;
               D2's caps are gone, so there is no ceiling left to declare.
               python_requires stays ">=3.8" — now on evidence rather than by
               inheritance: Python 3.8 resolves (numpy 1.24.4 / pandas 2.0.3 /
               nmrglue 0.12) and the 53-test corpus passes there. The symptom
               this entry described — a from-source numpy 1.26 build on 3.13+ —
               cannot occur any more, because numpy is no longer capped.
    Status:    resolved by D16

## C5 — dependency slimming (deliberately out of 0.4.0) — RESOLVED in 0.5.0

    Resolution: taken in 0.5.0 as D18, in the shape this entry sketched. Every
               candidate named below is now behind an extra, including hlsvdpro,
               which gets one of its own rather than staying in the default
               install — it is inert everywhere except x86_64 py3.8 with numpy
               1.x, which is exactly who the `hlsvd` extra is for. The
               double-declared ipykernel is gone. The extras are additive, so
               `pyamares-xmris[jupyter,hlsvd]` reproduces the 0.4.0 install. The
               cost this entry names — import-graph verification — was paid
               first, by D17. One correction to the note below: hlsvdpro is dead
               weight on every *current* stack, not literally every stack; a
               Python 3.8 environment caps setuptools below 82 and imports it
               fine.

    Status:    resolved by D18 — history below
    Symptom:   `pip install pyamares-xmris` pulls a Jupyter stack and an HTTP
               client into any environment that only wants to fit spectra. On a
               py3.13 resolution that is 62 packages for a library whose own
               imports are numpy/scipy/pandas/matplotlib/lmfit/sympy/nmrglue.
    Candidates: ipython, ipykernel, ipywidgets (the two ipywidgets marker lines
               together), requests, mat73, xlrd. Two of them are already odd:
               the `jupyter` extra declares ipykernel a second time, so the
               runtime dependency and the extra disagree about whose job it is;
               and hlsvdpro is dead weight under numpy>=2, where util/hsvd.py
               never imports it at all (D1's note) — the new default resolution
               therefore installs an x86_64 binary that nothing loads.
    Update (2026-08-18, found wiring the regression CI): hlsvdpro is now dead
               weight on *every* stack, not just numpy>=2. hlsvdpro 2.0.0 does
               `import pkg_resources` at module scope, and setuptools >=82
               removed pkg_resources — so on any current environment it
               installs but fails to import (`ModuleNotFoundError`), and
               util/hsvd.py's except-ImportError silently binds the vendored
               backend. Verified on real linux/amd64: import succeeds only
               with `setuptools<82` force-installed. The 0.5.0 slimming case
               for dropping the marker entirely is therefore stronger than
               when this entry was written.
    Cost:      import-graph verification. `import pyAMARES` is eager (kernel/fid.py
               and libs/MPFIR.py pull nmrglue and matplotlib at import time), so a
               demotion that misses one module turns a missing extra into a fatal
               ImportError rather than a deferred one. Beyond that, the six
               example notebooks and script/amaresfit_gui.py are real workflows
               that would break if the Jupyter/HTML-display path is demoted
               without an extra to restore it.
    Shape:     if taken — keep the hard six in install_requires, move the rest
               behind extras (`notebook`, `io`), and make the extras additive so
               `pyamares-xmris[jupyter]` reproduces today's install exactly.
    Not now:   0.4.0's job is the numpy 2 / pandas 3 lift, which is verified by
               the corpus. Slimming changes what an existing user's `pip install
               -U` leaves them with, and needs its own verification pass.

---

# Compatibility matrix

The 53-test regression corpus (`tests/test_regression.py` + `tests/test_api_surface.py`),
verified 2026-08-18 on macOS arm64. "Resolved" rows are what `pip install pyamares-xmris`
actually produces on that Python after D16 — nothing is pinned.

| Python | numpy | pandas | scipy | nmrglue | How reached | Result |
|---|---|---|---|---|---|---|
| 3.8 | 1.24.4 | 2.0.3 | 1.10.1 | 0.12 | resolved (the floor) | 53 green |
| 3.9 | 2.0.2 | 2.3.3 | 1.13.1 | 0.12 | resolved | 53 green |
| 3.12 | 1.26.4 | 2.1.4 | 1.17.1 | 0.11 | pinned — **the golden stack** | 53 green |
| 3.12 | 1.26.4 | 2.1.4 | 1.17.1 | 0.12 | pinned numpy/pandas only | 53 green |
| 3.12 | 2.5.2 | 3.0.5 | 1.18.0 | 0.12 | resolved | 53 green |
| 3.13 | 2.5.2 | 2.3.3 | 1.18.0 | 0.12 | pinned pandas only | 53 green |
| 3.13 | 2.5.2 | 3.0.5 | 1.18.0 | 0.12 | resolved | 53 green |
| 3.14 | 2.5.2 | 3.0.5 | 1.18.0 | 0.12 | resolved | 53 green |

Phase-2 verification additionally covered py3.12 with pandas 2.2.3 under numpy 1.26.4, and
py3.13 with pandas 3.0.3 under numpy 2.5.2, plus all six example notebooks under pandas 3.

Rows 3 and 4 differ only in nmrglue, and produce identical goldens — D16's floor bump is not
a numeric change. Across every row: all fitted parameters and all CRLB columns match the
frozen goldens at their 1e-4 default rtol (1e-3 for `CRLB(cs%) `, 1e-6 atol for
`chem shift(ppm)`); only the `sd` columns vary, which is C3 and is why they are guarded
structurally rather than frozen.

This is no longer one example fit. `tests/` is a regression suite whose goldens are frozen on
the 0.3.33 stack (row 3), so a row being green means the fitted output is the shipped output —
not merely that the fit converged.
