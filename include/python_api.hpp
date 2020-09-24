#ifndef PYTHON_API_HPP
#define PYTHON_API_HPP

#include "fcm.hpp"
#include "parameters.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>
#include <stdexcept>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace std;

inline tuple<vector<Py_intptr_t>, Py_intptr_t> array_shape_size(const np::ndarray& array) {
    auto shape_ptr = array.get_shape();
    const vector<Py_intptr_t> shape(shape_ptr, shape_ptr + array.get_nd());
    const auto size = accumulate(shape.cbegin(), shape.cend(), 1, multiplies<Py_intptr_t>()); // <numeric>, <functional>
    
    return make_tuple(shape, size);
}

template<class T>
tuple<vector<T>, vector<Py_intptr_t>> np_to_vector(const np::ndarray& array) {
    const auto dtype = np::dtype::get_builtin<T>();
    const auto array_type_T = (dtype == array.get_dtype()) ? array : array.astype(dtype);
    auto array_ptr = reinterpret_cast<T*>(array_type_T.get_data());

    const auto [shape, size] = array_shape_size(array_type_T);
    const vector<T> vec(array_ptr, array_ptr + size);

    return make_tuple(vec, shape);
}

template<class T>
std::vector<T> list_to_vector(const py::list& input) {
    std::vector<T> v;
    v.reserve(py::len(input));
    for (int i=0; i<py::len(input); i++) {
        v.push_back(py::extract<T>(input[i]));
    }
    return v;
}

template<class T>
np::ndarray c_array_to_np(T* array, const vector<Py_intptr_t>& shape) {
    const auto dtype = np::dtype::get_builtin<T>();

    py::list shape_list;
    for (const auto& s : shape) {
        shape_list.append(s);
    }
    const auto shape_tuple = py::tuple(shape_list);

    const auto strides_ptr = np::empty(shape_tuple, dtype).get_strides();
    py::list strides_list;
    for (unsigned long i=0; i<shape.size(); i++) {
        strides_list.append(strides_ptr[i]);
    }
    const auto strides_tuple = py::tuple(strides_list);

    return np::from_data(array, dtype, shape_tuple, strides_tuple, py::object());
}

// template<class T>
// np::ndarray vector_to_np(vector<T>& vec, const vector<Py_intptr_t>& shape=vector<Py_intptr_t>()) {
//     const auto result = c_array_to_np<T>(
//         vec.data(), shape.empty() ? vector<Py_intptr_t> {Py_intptr_t(vec.size())} : shape
//     );
//     return result.copy();   // copy since input vector will get destructed
// }                           // after exposed c++ function returns => vec.data() dangling

template<class T>
np::ndarray vector_to_np(const vector<T>& vec, const vector<Py_intptr_t>& shape=vector<Py_intptr_t>()) {
    // if (vec.empty()) {
    //     throw std::invalid_argument("vec must not be empty");
    // }
    py::list shape_list;
    if (shape.empty()) {
        shape_list.append(vec.size());
    } else {
        unsigned long size = 1;
        for (const auto s : shape) {
            shape_list.append(s);
            size *= s;
        }
        if (size != vec.size()) {
            throw std::invalid_argument("shape is incompatible with vec.size()");
        }
    }
    const auto shape_tuple = py::tuple(shape_list);
    const auto dtype = np::dtype::get_builtin<T>();

    auto result = np::empty(shape_tuple, dtype);
    std::copy(vec.cbegin(), vec.cend(), reinterpret_cast<T*>(result.get_data()));

    return result;
}

namespace fcm {

struct PyStructuralGroup : public StructuralGroup {
    PyStructuralGroup(const StructuralGroup& g);
    PyStructuralGroup(const double mass_fraction, const pieces_t pieces, const double strength,
                      const double density, const double cloud_mass_frac,
                      const double strength_scaler, const py::list& subfragment_mass_fractions);
    
    py::list subfragment_mass_fractions() const;
};

struct PyMeteoroid : public Meteoroid {
    PyMeteoroid(const double density, const double velocity, const double radius,
                const double angle, const double strength, const double cloud_mass_frac,
                const py::list& structural_groups);
    
    py::list groups() const;
};

py::tuple solve_impact(const PyMeteoroid& impactor, double z_start, double z_ground,
                       const FCM_params& params, const FCM_settings& settings,
                       const np::ndarray& height, const np::ndarray& density,
                       id_type seed, bool craters, bool dedz, bool final_states, bool timeseries);

} // namespace fcm

#endif // !PYTHON_API_HPP
