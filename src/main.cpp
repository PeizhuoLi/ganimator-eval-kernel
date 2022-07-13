#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> np_double;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> np_int;


void prepare_group_cost(np_double group_cost, np_double cost)
{
    auto gc = group_cost.mutable_unchecked<3>();
    auto c = cost.unchecked<2>();
    auto L = cost.shape(0);
    auto L2 = cost.shape(1);

    for (int i = 0; i < L; i++) {
        for (int j = i+1; j <= L; j++) {
            for (int k = 0; k + j - i - 1 < L2; k++) {
                gc(i, j, k) = gc(i, j - 1, k) + c(j-1, k + j-i-1);
            }
        }
    }
}


void nn_dp(np_double G, np_int E, np_int F, np_double Cost, int tmin, int L, int Nt) {
    auto g = G.mutable_unchecked<1>();
    auto e = E.mutable_unchecked<1>();
    auto f = F.mutable_unchecked<1>();
    auto cost = Cost.unchecked<3>();

    g(0) = 0;
    for (int i = tmin; i <= L; i++) {
        for (int k = 0; k < Nt; k++) {
            for (int l = 0; l <= i - tmin; l++) {
                auto new_val = g(l) + cost(l, i, k);
                if (new_val < g(i)) {
                    g(i) = new_val;
                    e(i) = k;
                    f(i) = l;
                }
            }
        }
    }
}


PYBIND11_MODULE(ganimator_eval_kernel, m) {
    m.doc() = R"pbdoc(
        GANimator evaluation kernel
        -----------------------

        .. currentmodule:: ganimator_eval_kernel

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("prepare_group_cost", &prepare_group_cost, "Prepare group cost");

    m.def("nn_dp", &nn_dp, "Nearest neighbor dp");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
