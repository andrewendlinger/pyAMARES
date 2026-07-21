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

    Status:    open, small
    Symptom:   TypeError: object of type 'float' has no len()
    Cause:     pandas 3.0 defaults string columns to the `str` dtype (PDEP-14), where
               missing values read back as NaN rather than None. PriorKnowledge.py:371
               reads `expr` from such a column and hands a float to lmfit.
    Estimate:  ~1-2 h for the known break; unknown what surfaces behind it
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
