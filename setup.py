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


jupyter_requirements = [
    "ipykernel",
    "notebook",
]

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


install_requires = [
    "pandas>=1.1.0,<2.2.0",
    "matplotlib>=3.1.3",
    "lmfit",
    "numpy>=1.18.1,<2.0.0",
    "scipy>=1.2.1",
    "sympy",
    "nmrglue",
    "xlrd",
    "jinja2",
    "tqdm",
    "mat73",
    "ipython",
    "ipykernel",
    "requests",
    "ipywidgets>=7.6.0,<8.0.0;python_version<'3.11'",  # For older Python versions
    "ipywidgets>=8.0.0;python_version>='3.11'",  # For newer Python versions
    # Use the better-performing 'hlsvdpro' package if running on supported platforms
    # (e.g., x86_64 or amd64 architectures). Otherwise, fall back to the custom
    # 'hlsvdpropy' implementation located in pyAMARES/libs/hlsvd.py.
    # (Refactored to PEP 508 markers in Issue #15 for Apple Silicon/uv compatibility)
    "hlsvdpro>=2.0.0; platform_machine == 'x86_64' or platform_machine == 'amd64'",
]


setup(
    name="pyamares-xmris",
    version=__version__,
    author=__author__,
    author_email="jia-xu-1@uiowa.edu",
    description=(
        "PyAMARES repackaged for clean pip installs on Apple Silicon (arm64): "
        "hlsvdpro is made platform-conditional. A faithful BSD repackage of "
        "HawkMRS/pyAMARES with zero algorithm changes; still 'import pyAMARES'."
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
        # 3.13/3.14 are intentionally not listed: the numpy<2.0 / pandas<2.2 caps
        # below resolve to numpy 1.26.4 and pandas 2.1.4, whose wheels stop at cp312.
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version requirement
    install_requires=install_requires,
    extras_require={
        "docs": doc_requirements,
        "jupyter": jupyter_requirements,
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
