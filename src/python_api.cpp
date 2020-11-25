#include "python_api.hpp"

#include "atmospheric_density.hpp"
#include "fcm.hpp"
#include "parameters.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace fcm;

PyStructuralGroup::PyStructuralGroup(const StructuralGroup& g) : StructuralGroup {g} {}

PyStructuralGroup::PyStructuralGroup(
    const double mass_fraction, const pieces_t pieces, const double strength,
    const double density,  const double cloud_mass_frac, const double strength_scaler,
    const py::list& subfragment_mass_fractions
) : StructuralGroup {mass_fraction, pieces, strength, density, 1, strength_scaler,
                     std::vector<double>{}}
{
    this->fragment_mass_fractions = list_to_vector<double>(subfragment_mass_fractions);
    this->cloud_mass_frac = cloud_mass_frac;
    if (cloud_mass_frac < 1) {
        if (strength_scaler <= 0) throw std::invalid_argument("strength_scaler must be > 0");
        if (std::abs(std::accumulate(fragment_mass_fractions.cbegin(),
                                     fragment_mass_fractions.cend(), 1) - 1) > 1e-10) {
            throw std::invalid_argument("sum of fragment_mass_fractions must equal 1");
        }
        std::sort(this->fragment_mass_fractions.begin(), this->fragment_mass_fractions.end());
    }
}

py::list PyStructuralGroup::subfragment_mass_fractions() const {
    py::list result;
    for (const auto& frac : this->fragment_mass_fractions) {
        result.append(frac);
    }
    return result;
}

PyMeteoroid::PyMeteoroid(const double density, const double velocity, const double radius,
                         const double angle, const double strength, const double cloud_mass_frac,
                         const py::list& structural_groups)
    : Meteoroid {density, velocity, radius, angle, strength, cloud_mass_frac}
{
    for (int i=0; i<py::len(structural_groups); i++) {
        this->structural_groups.push_back(py::extract<PyStructuralGroup>(structural_groups[i]));
    }
}

py::list PyMeteoroid::groups() const {
    py::list result;
    for (const auto& group : this->structural_groups) {
        result.append(PyStructuralGroup(group));
    }
    return result;
}

// Since this is meant for returning these timeseries as numpy arrays to python:
// -> Use C-style array instead of std::vector
// -> Allocating on the heap with new
// -> No delete[] is intended to be called!
std::tuple<double*, std::vector<Py_intptr_t>> _concat_timeseries_arr(
    const std::list<std::array<double, data_size>>& data
) {
    auto length = data_size * data.size();
    const auto timeseries = new double[length];

    auto iterator = timeseries;
    for (const auto& row : data) {
        iterator = std::copy(row.cbegin(), row.cend(), iterator); // <algorithm>
    }
    assert(iterator == timeseries + length); // <cassert>
    const std::vector<Py_intptr_t> shape {Py_intptr_t(data.size()), data_size};

    return std::make_tuple(timeseries, shape);
}

std::tuple<std::vector<double>, std::vector<Py_intptr_t>> _concat_timeseries_vec(
    const std::list<std::array<double, data_size>>& data
) {
    std::vector<double> timeseries;
    timeseries.reserve(data_size * data.size());

    for (const auto& row : data) {
        timeseries.insert(timeseries.end(), row.cbegin(), row.cend());
    }
    assert(timeseries.size() == timeseries.capacity());
    const std::vector<Py_intptr_t> shape {Py_intptr_t(data.size()), data_size};

    return std::make_tuple(timeseries, shape);
}

AtmosphericDensity _rho_a_from_np(const np::ndarray& height, const np::ndarray& density) {
    const auto [height_vec, height_shape] = np_to_vector<double>(height);
    const auto [density_vec, density_shape] = np_to_vector<double>(density);
    if (height_vec.empty() || density_vec.empty()) {
        throw std::invalid_argument("height and density must not be empty"); // <stdexcept>
    }
    if (height_vec.size() != density_vec.size()) {
        throw std::invalid_argument("height and density must have the same size");
    }
    if (height_shape.size() != 1
        && *std::max_element(height_shape.cbegin(), height_shape.cend()) != height_vec.size()) {
        throw std::invalid_argument("height must be one dimensional array");
    }
    if (density_shape.size() != 1
        && *std::max_element(density_shape.cbegin(), density_shape.cend()) != density_vec.size()) {
        throw std::invalid_argument("density must be one dimensional array");
    }

    return AtmosphericDensity(std::move(height_vec), density_vec); // <utility>
}

py::dict _fragment_info(const FragmentInfo& info) {
    py::list parent_ids, daughter_ids;
    auto it = info.parent_ids.crbegin();
    while (it != info.parent_ids.crend()) {
        parent_ids.append(*it++);
    }
    for (const auto id : info.daughter_ids) {
        daughter_ids.append(id);
    }
    py::dict result;
    result["strength"] = info.strength;
    result["ID"] = info.id;
    result["parent IDs"] = parent_ids;
    result["daughter IDs"] = daughter_ids;
    result["is cloud"] = info.is_cloud;
    result["did impact"] = info.impact;
    result["escaped atmosphere again"] = info.escape;

    return result;
}

py::tuple _craters_arrays(const std::list<Crater>& craters, bool record_data) {
    std::vector<double> x_vec, y_vec, r_vec;
    x_vec.reserve(craters.size());
    y_vec.reserve(craters.size());
    r_vec.reserve(craters.size());
    py::list fragment_ids;

    for (const auto& crater : craters) {
        x_vec.push_back(crater.x);
        y_vec.push_back(crater.y);
        r_vec.push_back(crater.r);
        if (record_data) {
            if (crater.fragment_ids.size() > 1) {
                py::list ids;
                for (const auto& id : crater.fragment_ids) {
                    ids.append(id);
                }
                fragment_ids.append(py::tuple(ids));
            } else {
                fragment_ids.append(crater.fragment_ids.front());
            }
        }
    }
    return py::make_tuple(vector_to_np(x_vec), vector_to_np(y_vec), vector_to_np(r_vec), fragment_ids);
}

py::tuple _final_states_array(
    const std::list<std::pair<FragmentInfo, std::list<std::array<double, data_size>>>>& data
) {
    std::vector<double> data_vec;
    data_vec.reserve(data_size * data.size());
    std::vector<Py_intptr_t> data_shape {Py_intptr_t(data.size()), data_size};

    std::vector<id_type> index_vec;
    index_vec.reserve(data.size());
    
    for (const auto& [info, timeseries] : data) {
        data_vec.insert(data_vec.end(), timeseries.back().cbegin(), timeseries.back().cend());
        index_vec.push_back(info.id);
    }

    return py::make_tuple(vector_to_np(index_vec), vector_to_np(data_vec, data_shape));
}

py::tuple fcm::solve_impact(const PyMeteoroid& impactor, double z_start, double z_ground,
                            const FCM_params& params, const FCM_settings& settings,
                            const np::ndarray& height, const np::ndarray& density, id_type seed,
                            bool craters, bool dedz, bool final_states) {

    const auto rho_a = _rho_a_from_np(height, density);

    const auto [dEdz, data] = solve_entry(impactor, z_start, z_ground, rho_a, params, settings, dedz, seed);
    const auto craters_list = craters ? calculate_craters(data, params, settings) : std::list<Crater>();

    py::list fragment_data;
    if (settings.record_data) {
        for (const auto& ts : data) {
            const auto [vec, shape] = _concat_timeseries_vec(ts.second);
            const auto info = _fragment_info(ts.first);
            fragment_data.append(py::make_tuple(info, vector_to_np(vec, shape)));
        }
    }
    const auto craters_tuple = craters ? _craters_arrays(craters_list, final_states || settings.record_data)
                                       : py::object();
    const auto final_states_tuple = final_states ? _final_states_array(data)
                                                 : py::object();
    
    const auto dEdz_array = dedz ? vector_to_np(dEdz) : py::object();

    return py::make_tuple(craters_tuple, dEdz_array, final_states_tuple, fragment_data);
}

auto max_seed() {
    return UINT_FAST32_MAX;
}


BOOST_PYTHON_MODULE(core)
{
    Py_Initialize();
    np::initialize();

    py::class_<FCM_settings>(
        "FcmSettings",
        py::init<CloudDispersionModel, ODEsolver, bool, bool, double, double, bool>()
    ).def_readwrite("cloud_dispersion_model", &FCM_settings::cloud_dispersion_model)
        .def_readwrite("ode_solver", &FCM_settings::ode_solver)
        .def_readwrite("flat_planet", &FCM_settings::flat_planet)
        .def_readwrite("fixed_timestep", &FCM_settings::fixed_timestep)
        .def_readwrite("precision", &FCM_settings::precision)
        .def_readwrite("dh", &FCM_settings::dh)
        .def_readwrite("record_data", &FCM_settings::record_data)
        .def_readwrite("cloud_stopping_criterion", &FCM_settings::cloud_stopping_criterion)
        .def_readwrite("max_iterations", &FCM_settings::max_iterations);
    
    py::class_<FCM_crater_coeff>(
        "FcmCraterCoeff",
        py::init<double, double, double, double, double, double, double, double, double>()
    ).def_readwrite("min_crater_radius", &FCM_crater_coeff::min_crater_radius)
        .def_readwrite("ground_density", &FCM_crater_coeff::ground_density)
        .def_readwrite("ground_strength", &FCM_crater_coeff::ground_strength)
        .def_readwrite("K1", &FCM_crater_coeff::K1)
        .def_readwrite("K2", &FCM_crater_coeff::K2)
        .def_readwrite("Kr", &FCM_crater_coeff::Kr)
        .def_readwrite("mu", &FCM_crater_coeff::mu)
        .def_readwrite("nu", &FCM_crater_coeff::nu)
        .def_readwrite("rim_factor", &FCM_crater_coeff::rim_factor);
    
    py::class_<FCM_params>(
        "FcmParams",
        py::init<double, double, double, double, double, double, double, double, double, double,
                 FCM_crater_coeff>()
    ).def_readwrite("g0", &FCM_params::g0)
        .def_readwrite("Rp", &FCM_params::Rp)
        .def_readwrite("ablation_coeff", &FCM_params::ablation_coeff)
        .def_readwrite("drag_coeff", &FCM_params::drag_coeff)
        .def_readwrite("lift_coeff", &FCM_params::lift_coeff)
        .def_readwrite("max_strength", &FCM_params::max_strength)
        .def_readwrite("frag_velocity_coeff", &FCM_params::frag_velocity_coeff)
        .def_readwrite("cloud_disp_coeff", &FCM_params::cloud_disp_coeff)
        .def_readwrite("strengh_scaling_disp", &FCM_params::strengh_scaling_disp)
        .def_readwrite("fragment_mass_disp", &FCM_params::fragment_mass_disp)
        .def_readwrite("min_crater_radius", &FCM_params::min_crater_radius)
        .def_readwrite("ground_density", &FCM_params::ground_density)
        .def_readwrite("ground_strength", &FCM_params::ground_strength)
        .def_readwrite("K1", &FCM_params::K1)
        .def_readwrite("K2", &FCM_params::K2)
        .def_readwrite("Kr", &FCM_params::Kr)
        .def_readwrite("mu", &FCM_params::mu)
        .def_readwrite("nu", &FCM_params::nu)
        .def_readwrite("rim_factor", &FCM_params::rim_factor);
    
    py::class_<PyStructuralGroup>("StructuralGroup", py::init<double, unsigned int, double, double,
                                                              double, double, py::list>())
        .def_readwrite("mass_fraction", &PyStructuralGroup::mass_fraction)
        .def_readwrite("pieces", &PyStructuralGroup::pieces)
        .def_readwrite("strength", &PyStructuralGroup::strength)
        .def_readwrite("density", &PyStructuralGroup::density)
        .def_readwrite("cloud_mass_frac", &PyStructuralGroup::cloud_mass_frac)
        .def_readwrite("strength_scaler", &PyStructuralGroup::strength_scaler)
        .add_property("subfragment_mass_fractions", &PyStructuralGroup::subfragment_mass_fractions);
    
    py::class_<PyMeteoroid>("Meteoroid", py::init<double, double, double, double, double, double,
                                                  py::list>())
        .def_readwrite("density", &PyMeteoroid::density)
        .def_readwrite("velocity", &PyMeteoroid::velocity)
        .def_readwrite("radius", &PyMeteoroid::radius)
        .def_readwrite("angle", &PyMeteoroid::angle)
        .def_readwrite("strength", &PyMeteoroid::strength)
        .def_readwrite("cloud_mass_frac", &PyMeteoroid::cloud_mass_frac)
        .add_property("mass", &PyMeteoroid::mass)
        .add_property("groups", &PyMeteoroid::groups);
    
    py::enum_<ODEsolver>("OdeSolver")
        .value("forward_euler", ODEsolver::forwardEuler)
        .value("improved_euler", ODEsolver::improvedEuler)
        .value("RK4", ODEsolver::RK4)
        .value("AB2", ODEsolver::AB2);
    
    py::enum_<CloudDispersionModel>("CloudDispersionModel")
        .value("pancake", CloudDispersionModel::pancake)
        .value("debris_cloud", CloudDispersionModel::debrisCloud)
        .value("chain_reaction", CloudDispersionModel::chainReaction);

    py::def("solve_impact", solve_impact);
    py::def("max_seed", max_seed);
}