from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Get path from active Conda environment
conda_env_path = os.environ.get('CONDA_PREFIX', '')
include_dir = os.path.join(conda_env_path, 'Library', 'include')
lib_dir = os.path.join(conda_env_path, 'Library', 'lib')

ext_modules = [
    Pybind11Extension(
        "tcov_module",
        ["tcov_wrapper.cpp", "scatters.cpp"],  # C++ files
        include_dirs=[include_dir],  # Path to Armadillo headers
        library_dirs=[lib_dir],  # Path to Armadillo libraries
        libraries=["lapack", "blas", "armadillo"],  # Armadillo libraries
        language="c++",
    )
]

setup(
    name="tcov_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
