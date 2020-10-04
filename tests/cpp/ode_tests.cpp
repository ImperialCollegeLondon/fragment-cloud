#define BOOST_TEST_MODULE ODEs
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include "ode.hpp"
#include <cmath>
#include <tuple>

struct TestValues {
    const double C_D, rho_a, A, v, m, g, theta, phi, C_ab, C_L, Rp, z, rho_m, C_disp, rf;
    
    TestValues () : C_D(0.5), rho_a(1.2), A(10), v(100), m(70e3), g(5), theta(1), phi(-0.5),
                    C_ab(2e-9), C_L(5e-4), Rp(6371e3), z(50e3), rho_m(3000), C_disp(1.5), rf(1.3) {}
};

BOOST_AUTO_TEST_SUITE(Conversion_Functions)

BOOST_AUTO_TEST_CASE(square_cube, * utf::tolerance(1e-8))
{
    const int l = 3;
    const double la = -2.5;
    BOOST_TEST_REQUIRE(std::square(l) == l*l);
    BOOST_TEST_REQUIRE(std::square(l) == std::pow(l, 2));
    BOOST_TEST_REQUIRE(std::square(la) == std::abs(la)*std::abs(la));
    BOOST_TEST_REQUIRE(std::square(la) == std::pow(la, 2));

    BOOST_TEST_REQUIRE(std::cube(l) == l*l*l);
    BOOST_TEST_REQUIRE(std::cube(l) == std::pow(l, 3));
    BOOST_TEST_REQUIRE(std::cube(la) == la*std::abs(la)*std::abs(la));
    BOOST_TEST_REQUIRE(std::cube(la) == std::pow(la, 3));
}

BOOST_AUTO_TEST_CASE(v_theta_phi, * utf::tolerance(1e-8))
{
    const TestValues p;
    const auto vx = fcm::vx(p.v, std::cos(p.theta), std::cos(p.phi));
    const auto vy = fcm::vy(p.v, std::cos(p.theta), std::sin(p.phi));
    const auto vz = fcm::vz(p.v, std::sin(p.theta));

    const auto vh = std::hypot(vx, vy);

    const auto v = fcm::v(vh, vz);
    const auto cos_phi = fcm::cos_phi(vx, vh);
    const auto sin_phi = fcm::sin_phi(vy, vh);
    const auto theta = fcm::theta(vh, vz);

    BOOST_TEST(v == p.v);
    BOOST_TEST(cos_phi == std::cos(p.phi));
    BOOST_TEST(sin_phi == std::sin(p.phi));
    BOOST_TEST(theta == p.theta);
}

BOOST_AUTO_TEST_CASE(ram_pressure, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = 1.2e4;
    const auto result = fcm::ram_pressure(p.rho_a, p.v);
    BOOST_TEST_REQUIRE(result == expected);
}

BOOST_AUTO_TEST_CASE(g, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = 4.9224336921;
    const auto g_flat = fcm::g_flat(p.g);
    const auto g_curved = fcm::g(p.g, p.z, p.Rp);
    const auto g_curved_z0 = fcm::g(p.g, 0, p.Rp);
    const auto g_curved_neg = fcm::g(p.g, -p.z, p.Rp);

    BOOST_TEST(g_flat == p.g);
    BOOST_TEST(g_curved_z0 == g_flat);
    BOOST_TEST(g_curved == expected);
    BOOST_TEST(g_flat > g_curved);
    BOOST_TEST(g_curved_neg > g_flat);
}

BOOST_AUTO_TEST_CASE(sphere, * utf::tolerance(1e-8))
{
    const double r = 2;
    const double rho = 3;
    const double m = 4.0 * M_PI / 3.0 * rho * std::pow(r, 3);

    BOOST_TEST(fcm::sphere_r(m, rho) == r);
    BOOST_TEST(fcm::sphere_rho(m, r) == rho);
    BOOST_TEST(fcm::sphere_m(rho, r) == m);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(breakup)

BOOST_AUTO_TEST_CASE(v_t, * utf::tolerance(1e-8))
{
    const double C = 0.8;
    const double r_meteoroid = 1.5;
    const double r_fragment = 0.3;
    const double rho_air = 0.1;
    const double rho_fragment = 2000;
    const double v = 300;

    const double expected = std::sqrt(3.0/2.0 * C * r_meteoroid / r_fragment * rho_air / rho_fragment) * v;
    BOOST_TEST(fcm::V_T(C, r_meteoroid, r_fragment, rho_air, rho_fragment, v) == expected);
}

BOOST_AUTO_TEST_CASE(Weibull, * utf::tolerance(1e-8))
{
    const double sigma0 = 10;
    const double alpha = 0.5;
    const double m0 = 1000;
    const double m_fragment = 10;

    const double expected = 100;
    BOOST_TEST(fcm::fragment_mean_strength(m_fragment/m0, sigma0, alpha) == expected);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(energy)

BOOST_AUTO_TEST_CASE(kinectic, * utf::tolerance(1e-8))
{
    const TestValues p;
    const auto expected = 0.5 * p.m * p.v * p.v;
    BOOST_TEST_REQUIRE(fcm::kinetic_energy(p.m, p.v) == expected);
}

BOOST_AUTO_TEST_CASE(kinectic_and_gravitational, * utf::tolerance(1e-8))
{
    const TestValues p;
    const auto expected = p.m * (0.5 * p.v*p.v - p.g * p.Rp * p.Rp / (p.Rp + p.z));
    const auto expected_flat = p.m * (p.v*p.v / 2 + p.g*p.z);
    BOOST_TEST(fcm::energy(p.m, p.v, p.z, p.g, p.Rp) == expected);
    BOOST_TEST(fcm::energy_flat(p.m, p.v, p.z, p.g) == expected_flat);
}

BOOST_AUTO_TEST_CASE(velocity_upper_bound, * utf::tolerance(1e-8))
{
    const TestValues p;
    const auto energy_1 = fcm::energy_flat(p.m, p.v, p.z, p.g);

    const double h = 10e3;
    const auto v_2 = std::sqrt(fcm::v2_ground_upper_bound(p.v, p.g, p.z, h));
    const auto energy_2 = fcm::energy_flat(p.m, v_2, h, p.g);

    BOOST_TEST(energy_1 == energy_2);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(craters)

BOOST_AUTO_TEST_CASE(regolith, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double ground_density = 2e3;
    const double r = std::sqrt(p.A / M_PI);
    const double expected = 1.03 * p.rf * std::pow(r, 0.83) * std::pow(p.g / (p.v*p.v), -0.17)
                            * std::pow(p.rho_m / ground_density, 0.33);
    BOOST_TEST(fcm::radius_in_regolith(r, p.g, p.v*p.v, p.rho_m, ground_density, p.rf) == expected);
}

BOOST_AUTO_TEST_CASE(granular_soil, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double ground_density = 2e3;
    const double ground_strength = 50e3;
    const double r = std::sqrt(p.A / M_PI);
    const double expected = 1.03 * p.rf * r * std::pow(ground_strength / (ground_density * p.v*p.v), -0.205)
                            * std::pow(p.rho_m / ground_density, 0.4);
    BOOST_TEST(fcm::radius_in_granular_soil(r, ground_strength, p.v*p.v, p.rho_m, ground_density, p.rf) == expected);
}

BOOST_AUTO_TEST_CASE(holsapple_regolith, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double ground_density = 1.5e3;
    const double ground_strength = 1e4;
    const double vz_squared = std::square(p.v);
    const double r = std::sqrt(p.A / M_PI) / 10;
    const auto r_regolith = fcm::radius_in_regolith(r, p.g, vz_squared, p.rho_m, ground_density, p.rf);
    const auto r_granular = fcm::radius_in_granular_soil(r, ground_strength, vz_squared, p.rho_m,
                                                         ground_density, p.rf);
    const auto [K1, K2, Kr, mu, nu] = std::make_tuple(0.15, 1.0, 1.1, 0.4, 0.4);
    const auto r_holsapple = fcm::Holsapple_crater_radius(
        ground_strength, ground_density, std::sqrt(vz_squared), p.g, r, p.rho_m, K1, nu, mu, K2,
        Kr, p.m/1e3, p.rf
    );
    BOOST_TEST(r_holsapple <= r_regolith);
    BOOST_TEST(r_holsapple >= r_granular);
}

BOOST_AUTO_TEST_CASE(holsapple_granular_soil, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double ground_density = 2.1e3;
    const double ground_strength = 1.3e5;
    const double vz_squared = std::square(p.v);
    const double r = std::sqrt(p.A / M_PI) / 10;
    const auto r_regolith = fcm::radius_in_regolith(r, p.g, vz_squared, p.rho_m, ground_density, p.rf);
    const auto r_granular = fcm::radius_in_granular_soil(r, ground_strength, vz_squared, p.rho_m,
                                                         ground_density, p.rf);
    const auto [K1, K2, Kr, mu, nu] = std::make_tuple(0.04, 1.0, 1.1, 0.55, 0.4);
    const auto r_holsapple = fcm::Holsapple_crater_radius(
        ground_strength, ground_density, std::sqrt(vz_squared), p.g, r, p.rho_m, K1, nu, mu, K2,
        Kr, p.m/1e3, p.rf
    );
    BOOST_TEST(r_holsapple <= r_regolith);
    BOOST_TEST(r_holsapple >= r_granular);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(ODE_Functions)

BOOST_AUTO_TEST_CASE(dvdt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = -3.0/7.0 + 4.207354924;
    const auto result = fcm::dvdt(p.C_D, fcm::ram_pressure(p.rho_a, p.v),
                                  p.A, p.m, p.g, std::sin(p.theta));
    BOOST_TEST(result == expected);
}

BOOST_AUTO_TEST_CASE(dmdt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = -1.2e-2;
    const auto result = fcm::dmdt(p.C_ab, fcm::ram_pressure(p.rho_a, p.v), p.A, p.v);
    BOOST_TEST(result == expected);
}

BOOST_AUTO_TEST_CASE(dthetadt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected_flat = 2.7015115293e-2 - 3e-5/7.0;
    const double expected_curved = expected_flat - 8.41461308e-6;
    const auto result_curved = fcm::dthetadt(p.g, std::cos(p.theta), p.v, p.C_L,
                                             fcm::ram_pressure(p.rho_a, p.v), p.A, p.m, p.Rp, p.z);
    const auto result_flat = fcm::dthetadt_flat(p.g, std::cos(p.theta), p.v, p.C_L,
                                                fcm::ram_pressure(p.rho_a, p.v), p.A, p.m);

    BOOST_TEST(result_flat == expected_flat);
    BOOST_TEST(result_curved == expected_curved);
}

BOOST_AUTO_TEST_CASE(vz_dzdt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = -84.1470984808;
    const auto result_dzdt = fcm::dzdt(p.v, std::sin(p.theta));
    const auto result_vz = fcm::vz(p.v, std::sin(p.theta));
    BOOST_TEST(result_dzdt == expected);
    BOOST_TEST(result_dzdt == result_vz);
}

BOOST_AUTO_TEST_CASE(vx_dxdt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected_flat = 47.4159881779;
    const double expected_curved = expected_flat * 0.9922130509;
    const auto result_dxdt_flat = fcm::dxdt_flat(p.v, std::cos(p.theta), std::cos(p.phi));
    const auto result_dxdt_z0 = fcm::dxdt(p.v, std::cos(p.theta), std::cos(p.phi), p.Rp, 0);
    const auto result_dxdt = fcm::dxdt(p.v, std::cos(p.theta), std::cos(p.phi), p.Rp, p.z);
    const auto result_vx = fcm::vx(p.v, std::cos(p.theta), std::cos(p.phi));

    BOOST_TEST(result_dxdt_flat == expected_flat);
    BOOST_TEST(result_dxdt_z0 == result_dxdt_flat);
    BOOST_TEST(result_vx == result_dxdt_flat);
    BOOST_TEST(result_dxdt == expected_curved);
}

BOOST_AUTO_TEST_CASE(vy_dydt, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected_flat = -25.9034724;
    const double expected_curved = expected_flat * 0.9922130509;
    const auto result_dydt_flat = fcm::dydt_flat(p.v, std::cos(p.theta), std::sin(p.phi));
    const auto result_dydt_z0 = fcm::dydt(p.v, std::cos(p.theta), std::sin(p.phi), p.Rp, 0);
    const auto result_dydt = fcm::dydt(p.v, std::cos(p.theta), std::sin(p.phi), p.Rp, p.z);
    const auto result_vy = fcm::vy(p.v, std::cos(p.theta), std::sin(p.phi));

    BOOST_TEST(result_dydt_flat == expected_flat);
    BOOST_TEST(result_dydt_z0 == result_dydt_flat);
    BOOST_TEST(result_vy == result_dydt_flat);
    BOOST_TEST(result_dydt == expected_curved);
}

BOOST_AUTO_TEST_CASE(drdt_pancake, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = 1.6814973649;
    const auto result_big_strength = fcm::d2rdt2_pancake(
        fcm::ram_pressure(p.rho_a, p.v), 1e6, p.C_disp, p.rho_m, std::sqrt(p.A/M_PI)
    );
    const auto result_small_strength = fcm::d2rdt2_pancake(
        fcm::ram_pressure(p.rho_a, p.v), 1e4, p.C_disp, p.rho_m, std::sqrt(p.A/M_PI)
    );
    BOOST_TEST(result_big_strength == 0);
    BOOST_TEST(result_small_strength == expected);
}

BOOST_AUTO_TEST_CASE(drdt_debriscloud, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double expected = std::sqrt(0.0021) * 100;
    const auto result_big_strength = fcm::drdt_debriscloud(
        fcm::ram_pressure(p.rho_a, p.v), 1e6, p.C_disp, p.rho_a, p.rho_m, p.v
    );
    const auto result_small_strength = fcm::drdt_debriscloud(
        fcm::ram_pressure(p.rho_a, p.v), 1e4, p.C_disp, p.rho_a, p.rho_m, p.v
    );
    BOOST_TEST(result_big_strength == 0);
    BOOST_TEST(result_small_strength == expected);
}

BOOST_AUTO_TEST_CASE(drdt_chainreaction, * utf::tolerance(1e-8))
{
    const TestValues p;
    const double dmdt = -1.2e-2;
    const double expected_big_strength = 8.4958291245e-6 * dmdt;
    const double expected_small_strength = expected_big_strength + 119.6826841204 / 313.0277825825;
    const auto result_big_strength = fcm::drdt_chainreaction(
        fcm::ram_pressure(p.rho_a, p.v), 1e6, std::sqrt(p.A/M_PI), p.m, dmdt, p.C_disp, p.rho_m
    );
    const auto result_small_strength = fcm::drdt_chainreaction(
        fcm::ram_pressure(p.rho_a, p.v), 1e4, std::sqrt(p.A/M_PI), p.m, dmdt, p.C_disp, p.rho_m
    );
    BOOST_TEST(result_big_strength == expected_big_strength);
    BOOST_TEST(result_small_strength == expected_small_strength);
}

BOOST_AUTO_TEST_SUITE_END()
