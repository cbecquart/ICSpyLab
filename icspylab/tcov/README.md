# Steps to create the tcov module with **Pybind11** 🐍

## 1. Install Armadillo

It is possible to use Conda for this step. Conda simplifies the library installation process, allowing you to install 
Armadillo without having to compile the library yourself.

````bash
conda install -c conda-forge armadillo
````

> **♩** The ``math.h`` file is part of the standard C library, and is automatically included in almost 
> all C/C++ environments. If you get an error saying that math.h is not found, this probably means that 
> your C++ development environment is not configured properly. 

## 2. Write C++ code

This is the file ``scatters.cpp`` coded by Andreas Alfons in [ICS](https://github.com/AuroreAA/ICSClust/blob/main/src/scatters.cpp).
The command:
````c++
#include <RcppArmadillo.h>
````
is replaced by the command:

````c++
#include <armadillo>
````

## 3. Create a wrapper 
To use ``scatters.cpp`` with **Pybind11** and Python, you need to adapt the file so that it can be correctly exposed 
as a Python module. This means adding special declarations to **Pybind11** to expose your C++ functions to Python.
This is what we do in ``tcov_wrapper.cpp`` to leave ``scatters.cpp`` unchanged and keep a clear separation between 
the original C++ code and the Python interface logic.

Special attention must be paid to how the C++ code reads input data in **Numpy array** format.
**Pybind11** can handle the **Numpy** format, and the following code converts it to an Armadillo array.
````c++
// Convert pybind11 array -> Armadillo matrix
auto buf = x.request();
arma::mat x_mat(reinterpret_cast<double*>(buf.ptr), buf.shape[1], buf.shape[0], false);
x_mat = x_mat.t();
````

To ensure conversion, the line of code below displays the first line of the Armadillo matrix.
````c++
std::cout << "Armadillo Matrix (First row): " << x_mat.row(0) << std::endl;
````

## 4. Create a ``setup.py`` file
Set up your compilation to include Armadillo.

> **♩** In ``setup.py``, don't forget to update the **include_dirs** and **library_dirs** include paths.
> These are in the project's conda environment.

## 5. Compile the module

To compile the C++ wrapper and generate the shared module file:

````bash
python setup.py build_ext --inplace
````
For this step, Python calls a C++ compiler, by default [Microsoft Visual C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/). 

---

## Tools for integrating C++ into Python

- [Pybind11](https://github.com/pybind/pybind11): a library that exposes C++ types in Python and vice versa, mainly for creating Python links from existing C++ code. 
- [Nanobind](https://github.com/wjakob/nanobind): a library that exposes C++ types in Python and vice versa. It is reminiscent of **Boost.Python** and **Pybind11** and uses almost identical syntax. Unlike these existing tools, nanobind is more efficient: bindings compile in less time, produce smaller binaries and have better runtime performance.
- [CARMA](https://carma.readthedocs.io/en/latest/introduction.html): CARMA provides fast bidirectional conversions between Numpy arrays and Armadillo matrices, vectors and cubes, just as RcppArmadillo does for R and Armadillo.
- [Cython](https://github.com/cython/cython/wiki): a programming language for writing C extensions to the Python language. Source code is translated into optimized C/C++ code and compiled into Python extension modules.
