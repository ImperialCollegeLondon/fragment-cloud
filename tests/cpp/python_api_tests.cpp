#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

#include "ode.hpp"
#include "python_api.hpp"
#include "parameters.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <vector>
#include <cassert>

BOOST_AUTO_TEST_SUITE(conversions)

BOOST_AUTO_TEST_CASE(arrayShapeSize)
{
    const auto shape1_ = py::make_tuple(2, 3);       // size 6
    const auto shape2_ = py::make_tuple(7);          // size 7
    const auto shape3_ = py::make_tuple(1, 6, 1, 2); // size 12
    const auto dtype1 = np::dtype::get_builtin<unsigned long>();
    const auto dtype2 = np::dtype::get_builtin<double>();
    const auto dtype3 = np::dtype::get_builtin<bool>();
    const auto array1 = np::zeros(shape1_, dtype1);
    const auto array2 = np::zeros(shape2_, dtype2);
    const auto array3 = np::zeros(shape3_, dtype3);

    const auto [shape1, size_1] = array_shape_size(array1);
    const auto [shape2, size_2] = array_shape_size(array2);
    const auto [shape3, size_3] = array_shape_size(array3);
    BOOST_TEST(size_1 == 6);
    BOOST_TEST((shape1 == std::vector<Py_intptr_t> {2, 3}));
    BOOST_TEST(size_2 == 7);
    BOOST_TEST((shape2 == std::vector<Py_intptr_t> {7}));
    BOOST_TEST(size_3 == 12);
    BOOST_TEST((shape3 == std::vector<Py_intptr_t> {1, 6, 1, 2}));
}

BOOST_AUTO_TEST_CASE(npToVector, * utf::tolerance(1e-8))
{
    const auto data = py::make_tuple(1, 2, 3);
    const auto dtypeInt = np::dtype::get_builtin<int>();
    const auto dtypeDouble = np::dtype::get_builtin<double>();
    const std::vector<Py_intptr_t> expected_shape {3};

    const auto arrayInt = np::array(data, dtypeInt);
    const auto arrayDouble = np::array(data, dtypeDouble);

    const auto [vectorDouble1, shapeDouble1] = np_to_vector<double>(arrayDouble);
    const auto [vectorDouble2, shapeDouble2] = np_to_vector<double>(arrayInt);
    const std::vector<double> expectedDouble {1, 2, 3};

    BOOST_TEST(shapeDouble1 == expected_shape);
    BOOST_TEST(shapeDouble2 == expected_shape);
    BOOST_TEST(vectorDouble1.size() == expectedDouble.size());
    BOOST_TEST(vectorDouble2.size() == expectedDouble.size());
    for (int i=0; i<expectedDouble.size(); i++) {
        BOOST_TEST(vectorDouble1[i] == expectedDouble[i]);
        BOOST_TEST(vectorDouble2[i] == expectedDouble[i]);
    }

    const auto [vectorInt1, shapeInt1] = np_to_vector<int>(arrayInt);
    const auto [vectorInt2, shapeInt2] = np_to_vector<int>(arrayDouble);
    const std::vector<int> expectedInt {1, 2, 3};

    BOOST_TEST(vectorInt1 == expectedInt);
    BOOST_TEST(vectorInt2 == expectedInt);
    BOOST_TEST(vectorInt1.size() == expectedInt.size());
    BOOST_TEST(vectorInt2.size() == expectedInt.size());
}

BOOST_AUTO_TEST_CASE(cArrayToNp, * utf::tolerance(1e-8))
{
    const auto arrayDouble = new double[4] {1, 2, 3, 4};
    const auto arrayInt = new int[4] {1, 2, 3, 4};
    const std::vector<Py_intptr_t> shape1 {4};
    const std::vector<Py_intptr_t> shape2 {2, 2};
    const std::vector<Py_intptr_t> shape3 {1, 4, 1};

    const auto resultDouble1 = c_array_to_np(arrayDouble, shape1);
    const auto resultDouble2 = c_array_to_np(arrayDouble, shape2);
    const auto resultDouble3 = c_array_to_np(arrayDouble, shape3);

    const auto resultInt1 = c_array_to_np(arrayInt, shape1);
    const auto resultInt2 = c_array_to_np(arrayInt, shape2);
    const auto resultInt3 = c_array_to_np(arrayInt, shape3);

    const auto [converted_back_double_1, shape_double_1] = np_to_vector<double>(resultDouble1);
    const auto [converted_back_double_2, shape_double_2] = np_to_vector<double>(resultDouble2);
    const auto [converted_back_double_3, shape_double_3] = np_to_vector<double>(resultDouble3);
    const std::vector<double> expected_double {1, 2, 3, 4};

    BOOST_TEST(converted_back_double_1.size() == 4);
    BOOST_TEST(converted_back_double_2.size() == 4);
    BOOST_TEST(converted_back_double_3.size() == 4);
    for (int i=0; i<4; i++) {
        BOOST_TEST(converted_back_double_1[i] == expected_double[i]);
        BOOST_TEST(converted_back_double_2[i] == expected_double[i]);
        BOOST_TEST(converted_back_double_3[i] == expected_double[i]);
    }
    BOOST_TEST(shape_double_1 == shape1);
    BOOST_TEST(shape_double_2 == shape2);
    BOOST_TEST(shape_double_3 == shape3);

    const auto [converted_back_int_1, shape_int_1] = np_to_vector<int>(resultInt1);
    const auto [converted_back_int_2, shape_int_2] = np_to_vector<int>(resultInt2);
    const auto [converted_back_int_3, shape_int_3] = np_to_vector<int>(resultInt3);
    const std::vector<int> expected_int {1, 2, 3, 4};

    BOOST_TEST(converted_back_int_1 == expected_int);
    BOOST_TEST(converted_back_int_2 == expected_int);
    BOOST_TEST(converted_back_int_3 == expected_int);
    BOOST_TEST(shape_int_1 == shape1);
    BOOST_TEST(shape_int_2 == shape2);
    BOOST_TEST(shape_int_3 == shape3);

    delete[] arrayDouble;
    delete[] arrayInt;
}

BOOST_AUTO_TEST_CASE(vecToNp)
{
    std::vector<int> vec {1, 2, 3, 4};
    const std::vector<Py_intptr_t> shape1 {4};
    const std::vector<Py_intptr_t> shape2 {2, 2};

    const auto result_0 = vector_to_np(vec);
    const auto result_1 = vector_to_np(vec, shape1);
    const auto result_2 = vector_to_np(vec, shape2);

    const auto [converted_back_1, shape_0] = np_to_vector<int>(result_0);
    const auto [converted_back_2, shape_1] = np_to_vector<int>(result_1);
    const auto [converted_back_3, shape_2] = np_to_vector<int>(result_2);

    BOOST_TEST(converted_back_1 == vec);
    BOOST_TEST(converted_back_2 == vec);
    BOOST_TEST(converted_back_3 == vec);
    BOOST_TEST(shape_0 == shape1);
    BOOST_TEST(shape_1 == shape1);
    BOOST_TEST(shape_2 == shape2);
}

BOOST_AUTO_TEST_CASE(listToVec)
{
    const std::vector<int> vec {1, 2, 3, 4};
    py::list input_list;
    for (const auto i : vec) {
        input_list.append(i);
    }

    const auto result_vec = list_to_vector<int>(input_list);
    BOOST_TEST(result_vec == vec);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(exposedFunctions)

BOOST_AUTO_TEST_CASE(solveImpact)
{
    fcm::FCM_settings s(fcm::CloudDispersionModel::debrisCloud, fcm::ODEsolver::AB2, false, false,
                        0.001, 10, true);
    const fcm::FCM_crater_coeff crater_coeff(0.75, 1.5e3, 1e4, 0.15, 1, 1.1, 0.4, 0.33, 1.3);
    const fcm::FCM_params p(5, 6371e3, 2e-9, 0.5, 5e-4, 330e6, 0.2, 1.5, 1, 0.9, crater_coeff);

    const double mass = 0.34;
    const double radius = 0.03;
    const double density = fcm::sphere_rho(mass, radius);
    const py::list subfragment_fractions(py::make_tuple(0.4, 0.6));

    const fcm::PyStructuralGroup group(1, 2, 1e5, density, 0.5, 0.25, subfragment_fractions);
    py::list structural_groups(py::make_tuple(group));

    fcm::PyMeteoroid m(density, 10e3, radius, 1, 1e4, 0.25, structural_groups);

    const double rho_0 = 0.1;
    const double H = 8e3;
    const std::vector<double> h {0, 10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3};
    std::vector<double> d(h.size());
    std::transform(h.cbegin(), h.cend(), d.begin(),
                   [=](const double z){ return rho_0 * std::exp(-z/H); });
    
    const auto h_array = vector_to_np(h);
    const auto d_array = vector_to_np(d);

    const auto result = fcm::solve_impact(m, 70e3, 10e3, p, s, h_array, d_array, 0,
                                          true, true, true);
}

BOOST_AUTO_TEST_SUITE_END()

bool init_unit_test() {
    Py_Initialize();
    np::initialize();

    py::class_<fcm::PyStructuralGroup>(
        "StructuralGroup", py::init<double, unsigned int, double, double, double, double, py::list>()
    );
    
    py::class_<fcm::PyMeteoroid>(
        "Meteoroid", py::init<double, double, double, double, double, double, py::list>()
    );

    return true;
}
