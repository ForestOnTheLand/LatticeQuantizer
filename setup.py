from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "lattice.core.csrc",
        ["csrc/bind.cpp", "csrc/core.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-ffast-math"],
    ),
]

setup(
    name="lattice",
    version=__version__,
    description="Best Lattice Construction via Gradient Descent",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.10",
)
