# About `pyamares-xmris`

This distribution is a **faithful repackage of [pyAMARES](https://github.com/hawkMRS/pyAMARES)**,
published under a different name on PyPI so that it can be installed on Apple Silicon (arm64)
machines. **There are no source or algorithm changes.**

- **Upstream project:** [HawkMRS/pyAMARES](https://github.com/hawkMRS/pyAMARES)
- **Upstream license:** BSD 3-Clause — Copyright (c) 2023-2025, Jia Xu, Magnetic Resonance
  Research Facility, University of Iowa. See [`LICENSE.txt`](LICENSE.txt), which is preserved
  verbatim.
- **Import name is unchanged:** you still write `import pyAMARES`. Only the *distribution*
  name differs, the same way `scikit-learn` is imported as `sklearn`.

## What differs from upstream

The entire delta is in `setup.py`. Three things:

1. **`hlsvdpro` is declared with a PEP 508 environment marker** instead of being appended
   unconditionally:

   ```
   hlsvdpro>=2.0.0; platform_machine == 'x86_64' or platform_machine == 'amd64'
   ```

2. **`numpy<2.0` and `pandas<2.2` caps**, which reflect pyAMARES' real compatibility limits.

3. **The distribution name**, plus metadata/attribution: `name="pyamares-xmris"`, an explicit
   `license`/`license_files` declaration, project URLs pointing back at upstream, and
   `find_packages(exclude=["tests", "tests.*"])` so the test suite is not installed as a
   top-level `tests` package in `site-packages`.

Nothing under `pyAMARES/` is modified.

## Why this exists

pyAMARES depends on [`hlsvdpro`](https://pypi.org/project/hlsvdpro/), a compiled library.
`hlsvdpro` 2.0.0 ships wheels **only for x86_64** (`macosx_10_9_x86_64`,
`manylinux2014_x86_64`, `win_amd64`) and publishes **no source tarball**. On an Apple Silicon
Mac there is therefore nothing for `pip` to install or build, and the whole install fails.

pyAMARES does not actually require it. It carries a bundled pure-Python fallback in
`pyAMARES/libs/hlsvd.py` and selects it automatically at import time
(`pyAMARES/util/hsvd.py`). With `hlsvdpro` absent, pyAMARES works correctly — the only cost
is speed on the HSVD code path. But because upstream's `setup.py` *declares* the dependency
unconditionally, `pip` obeys the declaration rather than the runtime reality, and the install
dies before any of that matters.

The correct fix is to merge the environment marker into upstream and let PyPI carry it. That
was attempted: the upstream maintainer has not responded to the pull requests filed against
`HawkMRS/pyAMARES`. Separately, upstream's PyPI releases are frozen at **0.3.28** while the
source tree has moved on to 0.3.33.

A downstream project cannot patch a *transitive* dependency's platform markers for its own
`pip install` users, and a published wheel may only depend on packages that exist on PyPI — a
git source is stripped at build time. So the fix has to live in a package on PyPI. Hence this
repackage, maintained for the [`xmris`](https://github.com/andrewendlinger/xmris) project.

If upstream ever merges the marker and cuts a release, this package becomes unnecessary and
consumers should move back to `pyamares`.

## Versioning

The version mirrors the upstream source tree this is built from (currently `0.3.33`). A
repackage-only re-release on identical upstream code gets a `.postN` suffix.

## Python version support

The `numpy<2.0` cap resolves to numpy 1.26.4, whose wheels stop at CPython 3.12; `pandas<2.2`
has the same ceiling. In practice this package installs on **Python 3.8–3.12**.

## Citation

Please cite the original authors' paper — this repackage adds nothing to cite:

> Xu, J.; Vaeggemose, M.; Schulte, R.F.; Yang, B.; Lee, C.-Y.; Laustsen, C.; Magnotta, V.A.
> PyAMARES, an Open-Source Python Library for Fitting Magnetic Resonance Spectroscopy Data.
> *Diagnostics* **2024**, *14*, 2668. https://doi.org/10.3390/diagnostics14232668
