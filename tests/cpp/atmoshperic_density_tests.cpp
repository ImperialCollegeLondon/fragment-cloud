#define BOOST_TEST_MODULE atmospheric density interpolator
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include "atmospheric_density.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

std::tuple<std::vector<double>, std::vector<double>> test_data(const double rho_0,
                                                               const double scale_height) {
    const std::vector<double> h {-1, 0, 1, 2, 3, 4, 5};
    std::vector<double> rho_a(h.size());
    std::transform(h.cbegin(), h.cend(), rho_a.begin(),                               // <algorithm>
                   [=](const double z){ return rho_0 * std::exp(-z/scale_height); }); // <cmath>

    return std::tuple(h, rho_a);
}

BOOST_AUTO_TEST_CASE(move_constructor, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(std::move(h), density); // <utility>

    BOOST_TEST(rho_a.calc(0) == rho_0);
    BOOST_TEST(rho_a.calc(-0.5) > rho_0);
    BOOST_TEST(rho_a.calc(2) < rho_0);
}

BOOST_AUTO_TEST_CASE(qualitative, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(h, density);

    BOOST_TEST(rho_a.calc(0) == rho_0);
    BOOST_TEST(rho_a.calc(-0.5) > rho_0);
    BOOST_TEST(rho_a.calc(2) < rho_0);
}

BOOST_AUTO_TEST_CASE(exact, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(h, density);

    const auto expected = std::exp(-1.3) * rho_0; // <cmath>
    BOOST_TEST(rho_a.calc(scale_height * 1.3) == expected);
}

BOOST_AUTO_TEST_CASE(bounds, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(h, density);

    BOOST_TEST(rho_a.calc(h.front()) == density.front());
    BOOST_TEST(rho_a.calc(h.back()) == density.back());
}

BOOST_AUTO_TEST_CASE(out_of_bounds, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(h, density);

    const auto expected_positive_direction = /*0;*/ density.back() * std::exp(-1/scale_height);
    BOOST_TEST(rho_a.calc(h.back() + 1) == expected_positive_direction);

    const auto expected_negative_direction = std::exp(2/scale_height) * rho_0; // <cmath>
    BOOST_TEST(rho_a.calc(-2) == expected_negative_direction);
}

BOOST_AUTO_TEST_CASE(scale_height, * utf::tolerance(1e-8))
{
    const double rho_0 = 5;
    const double scale_height = 2;
    const auto [h, density] = test_data(rho_0, scale_height);
    const fcm::AtmosphericDensity rho_a(h, density);

    BOOST_TEST(rho_a.scale_height(0.5) == scale_height);
    BOOST_TEST(rho_a.scale_height(0) == scale_height);
    BOOST_TEST(rho_a.scale_height(h.front() - 1) == scale_height);
    BOOST_TEST(rho_a.scale_height(h.back()) == scale_height);
    BOOST_TEST(rho_a.scale_height(h.back() + 1) == scale_height);
}
