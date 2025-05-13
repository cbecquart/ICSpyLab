#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace py = pybind11;

// Inclure votre fonction tcov_cpp
arma::mat tcov_cpp(const arma::mat& x, const double& beta);

py::array_t<double> tcov_py(py::array_t<double, py::array::c_style | py::array::forcecast> x, double beta) {
    // Convertir pybind11 array -> Armadillo matrix
    auto buf = x.request();
    arma::mat x_mat(reinterpret_cast<double*>(buf.ptr), buf.shape[1], buf.shape[0], false);
    x_mat = x_mat.t();

    // Appeler la fonction C++
    arma::mat result = tcov_cpp(x_mat, beta);

    // Convertir Armadillo matrix -> pybind11 array
    return py::array_t<double>(
        {result.n_rows, result.n_cols}, // Shape
        {sizeof(double) * result.n_cols, sizeof(double)}, // Strides (C-contiguous)
        result.memptr() // Pointer to data
    );
}

PYBIND11_MODULE(tcov_module, m) {
    m.def("tcov", &tcov_py, "Compute TCOV scatter matrix");
}
