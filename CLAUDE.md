# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this repo is

A fork of [hawkMRS/pyAMARES](https://github.com/hawkMRS/pyAMARES), published on PyPI as
**`pyamares-xmris`**. The **import name is unchanged** — `import pyAMARES`. It exists to serve
[xmris](https://github.com/andrewendlinger/xmris) as its AMARES-fitting engine with corrected
packaging metadata and, from 0.4.0 on, targeted source fixes.

- **`DIVERGENCE.md` is the law.** Every behavioral difference from upstream gets a `D`-entry;
  undecided candidates get `C`-entries. No source change ships without its ledger entry.
- **Divergence is accepted policy.** Long-term this fork diverges from upstream and that is
  fine — upstream is tracked and cherry-picked, not followed. `0.3.33` remains the permanent
  zero-divergence artifact on PyPI.
- Branch **`pyamares-xmris`** is the release line. Branch off it, PR back into it. The user
  merges PRs.

## Build & environments

- Metadata lives in legacy **`setup.py`** (no `[project]` table; `pyproject.toml` holds only
  `[build-system]` and `[tool.ruff]`). The wheel is pure Python (`py3-none-any`).
- **Not a uv project** — there must be no committed `uv.lock` (it is gitignored). Test against
  specific stacks with `uv run --no-project --python 3.X --with ... --with .` or scratch venvs
  plus `uv pip install --no-deps -e .`.
- Version single source of truth: `pyAMARES/__init__.py` (`__version__`), AST-parsed by
  `setup.py`.
- Lint: `ruff check .` and `ruff format --check .` (line length 88; `pyAMARES/libs/hlsvd.py`
  is excluded as vendored third-party code).

## Tests

- **Numeric regression corpus** (the safety net for any dependency or source change):
  `pytest tests/test_regression.py tests/test_api_surface.py -o addopts=""`
  (`-o addopts=""` is required — `pytest.ini` injects `--nbval-lax`, which errors in minimal
  envs without nbval).
- **Goldens under `tests/goldens/` are frozen** — captured on the 0.3.33 stack (py3.12,
  numpy 1.26.4, pandas 2.1.4). Never regenerate them without an explicit decision; tolerance
  adjustments are data-only edits inside the golden JSON, with a comment.
- Notebook smoke tests: `pytest --nbval-lax --current-env tests/` (execution-only, no output
  comparison; needs nbval + ipykernel).

## The xmris contact surface is API

xmris consumes exactly: `initialize_FID`, `fitAMARES` (`least_squares`/`leastsq`),
`result_pd_to_params`, `multieq6`, `uninterleave`, `pyAMARES.libs.logger.set_log_level` /
`DEFAULT_LOG_LEVEL`, and reads only `FIDobj.result_multiplets`. Its 13 column labels —
including the **trailing space in `"CRLB(cs%) "`** — the metabolite row index, and the FIDobj
Namespace attribute names (`initialParams`, `peaklist`, `styled_df`, `simple_df`, `out_obj`,
`fitted_fid`, `result_multiplets`) are load-bearing API. `tests/test_api_surface.py` pins them;
do not rename any of them without a coordinated xmris release.

## Gotchas

- Since D17, `import pyAMARES` **defers** nmrglue, `matplotlib.pyplot` and mat73 to first
  use — they are imported inside the function bodies that need them, so a broken one now
  fails at that call instead of at import. Bare `matplotlib` still loads (lmfit imports it
  at module scope), and so does jinja2 (`util/report.py` probes it by design).
  Two tests in `tests/test_api_surface.py` pin the import graph — a fixed blocklist
  (`test_bare_import_keeps_heavy_modules_unloaded`) and an AST scan of `pyAMARES/**`
  (`test_no_undocumented_module_level_third_party_imports`, allow-list
  numpy/scipy/pandas/lmfit/jinja2 plus documented per-file exceptions). Put a new heavy
  import in a function body, not at module level. numpy 2 requires `nmrglue>=0.12`.
- `hlsvdpro` is optional by design: `util/hsvd.py` never imports it under numpy ≥2 (the
  vendored pure-Python `pyAMARES/libs/hlsvd.py` is used), and falls back to the vendored copy
  when it is absent under numpy 1.x. Its PEP 508 marker restricts it to x86_64/amd64 — it
  ships no arm64 wheel and no sdist.
- The `sd`, `sd(ppm)`, `sd(Hz)`, `sd(deg)` result columns are **not reproducible across
  dependency stacks** (ill-conditioned Fisher matrix through `pinv`/`lstsq`; up to ~238%
  spread observed). They are guarded structurally only, never golden-compared. Fitted
  parameters and all CRLB columns are regression-locked.

## Releases

- Never bump the version before CI is green. The bump is the final commit of a release PR.
- **The user pushes the `v*` tag**; `publish.yml` then builds and publishes to PyPI via
  trusted publishing automatically. A pushed tag is an irreversible release.
