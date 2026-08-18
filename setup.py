import ast
import glob
import os

from setuptools import find_packages, setup
from setuptools.command.sdist import sdist as _sdist

pyamares_init_file = os.path.join(os.path.dirname(__file__), "pyAMARES", "__init__.py")


def get_version(init_file_path, *variables):
    with open(init_file_path, "r") as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    vars_dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in variables:
                    value = eval(
                        compile(ast.Expression(node.value), init_file_path, "eval")
                    )
                    vars_dict[target.id] = value
    return vars_dict


vars_dict = get_version(pyamares_init_file, "__version__", "__author__")
__version__ = vars_dict.get("__version__")
__author__ = vars_dict.get("__author__")


# __version__ = "0.2"
# __author__ = "Jia Xu"
# from pyAMARES import __author__, __version__
class CustomSDist(_sdist):
    user_options = _sdist.user_options + [
        ("include-docs", None, "Include documentation in the distribution")
    ]

    def initialize_options(self):
        _sdist.initialize_options(self)
        self.include_docs = False

    def finalize_options(self):
        _sdist.finalize_options(self)

    def run(self):
        if self.include_docs:
            self.distribution.data_files.extend(
                [
                    ("docs", glob.glob("docs/source/**", recursive=True)),
                    ("docs", glob.glob("docs/*.*")),
                    ("docs", ["docs/Makefile"]),
                ]
            )
        _sdist.run(self)


# MATLAB v7.3 (HDF5) .mat reading -- fileio/readmat.py::readmrs and
# fileio/readfidall.py::read_fidall import mat73 inside the v7.3 branch only.
matlab_requirements = [
    "mat73",
]

# Excel prior knowledge -- kernel/PriorKnowledge.py hands .xlsx/.xls to
# pandas.read_excel, which needs openpyxl (.xlsx) or xlrd (legacy .xls).
excel_requirements = [
    "openpyxl",
    "xlrd",
]

# The optional native HSVD backend. Deliberately its own extra and part of no
# other: on any environment with setuptools >= 82 -- i.e. everything from Python
# 3.9 up -- hlsvdpro 2.0.0 installs but cannot import (module-scope
# `import pkg_resources`), and util/hsvd.py never looks for it under numpy >= 2.
# It is still live on x86_64 Python 3.8 with numpy 1.x, where setuptools is
# capped below 82, and that is who this extra is for. The marker is D1's: the
# project publishes x86_64/amd64 wheels only, and no sdist. See D18.
hlsvd_requirements = [
    "hlsvdpro>=2.0.0; platform_machine == 'x86_64' or platform_machine == 'amd64'",
]

# Everything the example notebooks and the interactive workflows need. Plain
# list concatenation rather than self-referential extras, so the metadata stays
# readable on old pip/setuptools: `pip install 'pyamares-xmris[jupyter]'`
# reproduces the pre-0.5.0 install (minus hlsvdpro, which has its own extra),
# plus notebook and openpyxl. See D18.
jupyter_requirements = (
    [
        "notebook",
        "ipykernel",
        "ipython",
        "ipywidgets>=7.6.0,<8.0.0;python_version<'3.11'",  # For older Python versions
        "ipywidgets>=8.0.0;python_version>='3.11'",  # For newer Python versions
        "requests",
    ]
    + matlab_requirements
    + excel_requirements
)

doc_requirements = [
    "sphinx",
    "nbsphinx",
    "sphinx_tabs",
    "sphinx_rtd_theme",
    "Pygments",
    "ipywidgets",
]

ruff_requirements = [
    "ruff",
    "pre-commit",
    "pytest",
]


# What the fitting engine itself imports. Everything else -- the Jupyter stack,
# the HTTP client, the optional file readers -- lives behind an extra (D18).
install_requires = [
    "numpy>=1.18.1",
    "scipy>=1.2.1",
    "pandas>=1.1.0",
    "matplotlib>=3.1.3",
    "lmfit",
    "sympy",
    # 0.12 replaced np.dtype('a8') with np.dtype('S8'); anything older fails to
    # import under numpy 2. The floor is still required after D17 made the import
    # lazy -- laziness only moves the failure from `import pyAMARES` to the first
    # ng.proc_base call, which is harder to diagnose, not less fatal. See D16.
    "nmrglue>=0.12",
    "jinja2",
    "tqdm",
]


setup(
    name="pyamares-xmris",
    version=__version__,
    author=__author__,
    author_email="jia-xu-1@uiowa.edu",
    description=(
        "PyAMARES repackaged for clean pip installs on Apple Silicon (arm64) and "
        "for numpy 2 / pandas 3: a faithful BSD repackage of HawkMRS/pyAMARES "
        "carrying only minimal, ledger-documented compatibility fixes "
        "(see DIVERGENCE.md); still 'import pyAMARES'."
    ),
    long_description=open("README.rst", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    license="BSD-3-Clause",
    license_files=["LICENSE.txt"],
    url="https://github.com/andrewendlinger/pyAMARES",
    project_urls={
        "Upstream (original project)": "https://github.com/hawkMRS/pyAMARES",
        "Upstream documentation": "https://pyamares.readthedocs.io/en/latest/index.html",
        "Divergence from upstream": (
            "https://github.com/andrewendlinger/pyAMARES/blob/pyamares-xmris/"
            "DIVERGENCE.md"
        ),
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    # Kept at upstream's floor: with the caps lifted, 3.8 still resolves (numpy
    # 1.24.4 / pandas 2.0.3 / nmrglue 0.12) and the regression corpus passes there.
    # No upper bound — 3.13 and 3.14 are verified. See D16.
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "matlab": matlab_requirements,
        "excel": excel_requirements,
        "hlsvd": hlsvd_requirements,
        "jupyter": jupyter_requirements,
        "docs": doc_requirements,
        "ruff": ruff_requirements,
        "dev": jupyter_requirements + doc_requirements + ruff_requirements,
    },
    cmdclass={
        "sdist": CustomSDist,
    },
    data_files=[],
    zip_safe=False,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "amaresFit=pyAMARES.script.amaresfit:main",
        ],
    },
)
