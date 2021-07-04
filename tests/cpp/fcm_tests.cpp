#define BOOST_TEST_MODULE fragment-cloud model
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include "fcm.hpp"

#include "atmospheric_density.hpp"
#include "fragment.hpp"
#include "ode.hpp"
#include "parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <iostream>
#include <iomanip>
#include <string>

auto make_settings(const fcm::CloudDispersionModel cloud_model=fcm::CloudDispersionModel::debrisCloud,
                   const fcm::ODEsolver solver=fcm::ODEsolver::AB2, const bool flat_earth=false,
                   const bool fixed_timestep=true) {
    return std::make_shared<const fcm::FCM_settings>(cloud_model, solver, 1e-2, 10, true, flat_earth,
                                                     fixed_timestep);
}

auto test_params(const std::shared_ptr<const fcm::FCM_settings>& settings, const double strength,
                 const bool is_cloud=true, const double phi=0, const double theta=1,
                 const double cloud_frac=1) {
    const fcm::FCM_crater_coeff crater_coeff(0.75, 1.5e3, 1e4, 0.15, 1, 1.1, 0.4, 0.33, 1.3);
    const fcm::FCM_params p(5, 6371e3, 2e-9, 0.5, 5e-4, 330e6, 0.2, 1.5, 1, 0.9, crater_coeff);

    const double rho_0 = 0.1;
    const double H = 8e3;
    const std::vector<double> h {0, 10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3};
    std::vector<double> d(h.size());
    std::transform(h.cbegin(), h.cend(), d.begin(),
                   [=](const double z){ return rho_0 * std::exp(-z/H); });

    const double mass = 0.34;
    const double radius = 0.03;
    const double density = fcm::sphere_rho(mass, radius);
    const fcm::AtmosphericDensity rho_a(std::move(h), d);

    auto structural_group = cloud_frac < 1 ? std::list<fcm::StructuralGroup> {
        fcm::StructuralGroup(1, 1, strength, density, cloud_frac, 0.25, std::vector<double>{0.4, 0.6})
    } : std::list<fcm::StructuralGroup> {};
    const fcm::Meteoroid m(density, 10e3, radius, std::abs(theta), strength, cloud_frac,
                           std::move(structural_group));
    // Fragment ctor
    fcm::Fragment f(m.mass(), m.velocity, m.radius, theta, 20e3, m.strength, m.cloud_mass_frac,
                    m.density, std::make_shared<const fcm::AtmosphericDensity>(rho_a),
                    std::make_shared<const fcm::FCM_params>(p), settings, 123U,
                    std::list<fcm::SubFragment>(), 0, 0, 0, std::cos(phi), std::sin(phi));
    if (is_cloud) {
        const auto tmp = f.break_apart();
        BOOST_TEST_REQUIRE(tmp.size() == 1);
        f = tmp.front();
    }

    return std::make_tuple(p, m, f, rho_a);
}

template<class T>
void print_result(const std::array<T, data_size>& result, const fcm::ODEsolver solver,
                  const fcm::CloudDispersionModel cloud) {
    
    std::string cloud_name, solver_name;
    switch (solver) {
        case fcm::ODEsolver::forwardEuler: {
            solver_name = "forwardEuler";
            break;
        }
        case fcm::ODEsolver::improvedEuler: {
            solver_name = "improvedEuler";
            break;
        }
        case fcm::ODEsolver::RK4: {
            solver_name = "RK4";
            break;
        }
        case fcm::ODEsolver::AB2: {
            solver_name = "AB2";
            break;
        }
        default:
            throw std::invalid_argument("invalid ODE solver");
    }
    switch (cloud) {
        case fcm::CloudDispersionModel::pancake: {
            cloud_name = "pancake";
            break;
        }
        case fcm::CloudDispersionModel::debrisCloud: {
            cloud_name = "debrisCloud";
            break;
        }
        case fcm::CloudDispersionModel::chainReaction: {
            cloud_name = "chainReaction";
            break;
        }
        default:
            throw std::invalid_argument("invalid cloud dispersion model");
    }

    std::cout << std::setprecision(4) << "\nImpact Time: " << result[0] << " seconds" << std::endl
              << "Impact Coordinates: (" << result[5]/1e3 << ", " << result[6]/1e3 << ") km" << std::endl
              << "Impact Elevation: " << result[4]/1e3 << " km" << std::endl
              << "Impact Mass: " << result[1]/1e3 << " t" << std::endl
              << "Impact Velocity: " << result[2] << " m/s" << std::endl
              << "Impact Angle: " << result[3] * 180 / M_PI << " degrees" << std::endl
              << "Radius at impact: " << result[7] << " m" << std::endl
              << "Cloud dispersion model: " << cloud_name << std::endl
              << "ODE solver: " << solver_name << std::endl;
}

BOOST_AUTO_TEST_SUITE(dfdt)

BOOST_AUTO_TEST_CASE(pancake, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::pancake);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6);
    const auto result = fcm::dfdt(fragment, params, *settings, 0);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy == 0);
    BOOST_TEST(result.dz < 0);
    BOOST_TEST(result.dr == 0);
    BOOST_TEST(result.d2r == 0);

    const auto [params_2, m_2, fragment_2, rho_a_] = test_params(settings, 1e4);
    const auto result_2 = fcm::dfdt(fragment_2, params_2, *settings, 1000);

    BOOST_TEST(result_2.dm < 0);
    BOOST_TEST(result_2.dx > 0);
    BOOST_TEST(result_2.dy == 0);
    BOOST_TEST(result_2.dz < 0);
    BOOST_TEST(result_2.dr == 0);
    BOOST_TEST(result_2.d2r > 0);
}

BOOST_AUTO_TEST_CASE(debrisCloud, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::debrisCloud);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6);
    const auto result = fcm::dfdt(fragment, params, *settings, -200);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy == 0);
    BOOST_TEST(result.dz < 0);
    BOOST_TEST(result.dr == 0);
    BOOST_TEST(result.d2r == 0);

    const auto [params_2, m_2, fragment_2, rho_a_] = test_params(settings, 1e4);
    const auto result_2 = fcm::dfdt(fragment_2, params_2, *settings, 0);

    BOOST_TEST(result_2.dm < 0);
    BOOST_TEST(result_2.dx > 0);
    BOOST_TEST(result_2.dy == 0);
    BOOST_TEST(result_2.dz < 0);
    BOOST_TEST(result_2.dr > 0);
    BOOST_TEST(result_2.d2r == 0);
}

BOOST_AUTO_TEST_CASE(chainReaction, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::chainReaction);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6);
    const auto result = fcm::dfdt(fragment, params, *settings, 400);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy == 0);
    BOOST_TEST(result.dz < 0);
    BOOST_TEST(result.dr < 0);
    BOOST_TEST(result.d2r == 0);

    const auto [params_2, m_2, fragment_2, rho_a_] = test_params(settings, 1e4);
    const auto result_2 = fcm::dfdt(fragment_2, params_2, *settings, 0);

    BOOST_TEST(result_2.dm < 0);
    BOOST_TEST(result_2.dx > 0);
    BOOST_TEST(result_2.dy == 0);
    BOOST_TEST(result_2.dz < 0);
    BOOST_TEST(result_2.dr > 0);
    BOOST_TEST(result_2.d2r == 0);
}

BOOST_AUTO_TEST_CASE(phi, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::chainReaction);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6, false, 0.5);
    const auto result = fcm::dfdt(fragment, params, *settings, 0);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy > 0);
    BOOST_TEST(result.dz < 0);
    BOOST_TEST(result.dr < 0);
    BOOST_TEST(result.d2r == 0);

    const auto [params_2, m_2, fragment_2, rho_a_] = test_params(settings, 1e6, false, -0.5);
    const auto result_2 = fcm::dfdt(fragment_2, params_2, *settings, 2000);

    BOOST_TEST(result_2.dm < 0);
    BOOST_TEST(result_2.dx > 0);
    BOOST_TEST(result_2.dy < 0);
    BOOST_TEST(result_2.dz < 0);
    BOOST_TEST(result_2.dr < 0);
    BOOST_TEST(result_2.d2r == 0);
}

BOOST_AUTO_TEST_CASE(negative_theta, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::chainReaction);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6, true, 0.5, -0.1);
    const auto result = fcm::dfdt(fragment, params, *settings, -1);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy > 0);
    BOOST_TEST(result.dz > 0);
    BOOST_TEST(result.dr < 0);
    BOOST_TEST(result.d2r == 0);
}

BOOST_AUTO_TEST_CASE(flat, * utf::tolerance(1e-8))
{
    const auto settings = make_settings(fcm::CloudDispersionModel::chainReaction,
                                        fcm::ODEsolver::AB2, true);
    const auto [params, m, fragment, rho_a] = test_params(settings, 1e6);
    const auto result = fcm::dfdt(fragment, params, *settings, 0);

    BOOST_TEST(result.dm < 0);
    BOOST_TEST(result.dtheta > 0);
    BOOST_TEST(result.dx > 0);
    BOOST_TEST(result.dy == 0);
    BOOST_TEST(result.dz < 0);
    BOOST_TEST(result.dr < 0);
    BOOST_TEST(result.d2r == 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(dEdz)

BOOST_AUTO_TEST_CASE(pancake, * utf::tolerance(0.3))
{
    const std::vector<fcm::CloudDispersionModel> cloud_models {
        fcm::CloudDispersionModel::pancake, fcm::CloudDispersionModel::debrisCloud,
        fcm::CloudDispersionModel::chainReaction
    };
    for (const auto cloud_model : cloud_models) {
        const auto settings = make_settings(cloud_model, fcm::ODEsolver::AB2, false, false);
        const auto [params, meteoroid, f, rho_a] = test_params(settings, 1e4, false);
        const double ground_height = 10e3;
        const double start_height = 70e3;
        const auto [dEdz, result] = fcm::solve_entry(meteoroid, start_height, ground_height,
                                                    rho_a, params, *settings, true);
        for (size_t i = 1; i < dEdz.size() - 1; i++) {
            if (dEdz[i-1] > 0 && dEdz[i] > 0 && dEdz[i+1] > 0) {
                BOOST_TEST(dEdz[i-1] / dEdz[i] == dEdz[i] / dEdz[i+1]);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(it_works, * utf::tolerance(1e-8))
{
    const auto settings = make_settings();
    auto [params, m, fragment, rho_a] = test_params(settings, 1e4);
    const double start_height = 100e3;
    const double ground_height = 10e3;
    fcm::dEdzInterpolator dEdz_vector(fragment, start_height, ground_height, settings->dh);
    const auto z0 = params.Rp + fragment.z();

    fcm::offset descend {-5, 0, 0, 0, 0, -1e3, 0, 0};
    fcm::offset ascend {+5, 0, 0, 0, 0, +2e3, 0, 0};

    fragment += descend;
    dEdz_vector.add_dedz(fragment);

    const auto values1 = dEdz_vector.values();
    const auto z_index_0 = dEdz_vector.z_index_0();
    const double E1 = 0.5 * m.mass() * std::square(m.velocity)
                      - m.mass() * params.g0 * std::square(params.Rp) / z0;
    const double E2 = 0.5 * m.mass() * std::square(m.velocity + descend.dv)
                      - m.mass() * params.g0 * std::square(params.Rp) / (z0 + descend.dz);
    const double dEdz1 = (E2 - E1) / descend.dz;

    const size_t max_index = std::floor(-descend.dz / settings->dh);
    BOOST_TEST(z_index_0 == std::floor((start_height - z0 + params.Rp) / settings->dh));
    BOOST_TEST(values1.size() == max_index + 1);
    BOOST_TEST(values1[0] == 0);
    BOOST_TEST(values1[max_index] == dEdz1);
    for (size_t i = 1; i < max_index; i++) {
        BOOST_TEST(values1[i] > 0);
        BOOST_TEST(values1[i] < dEdz1);
    }

    fragment += ascend;
    dEdz_vector.add_dedz(fragment);

    const auto values2 = dEdz_vector.values();
    const auto z_index_1 = dEdz_vector.z_index_0();
    const double E3 = 0.5 * m.mass() * std::square(m.velocity + descend.dv + ascend.dv)
                      - m.mass() * params.g0 * std::square(params.Rp) / (z0 + descend.dz + ascend.dz);
    const double dEdz2 = (E3 - E2) / ascend.dz;
    BOOST_TEST_REQUIRE(dEdz2 < dEdz1);
    BOOST_TEST(z_index_1 == std::ceil((start_height - z0 - descend.dz - ascend.dz + params.Rp) / settings->dh));
    BOOST_TEST(values2.size() = max_index + 1 + z_index_0 - z_index_1);
    BOOST_TEST(values2[0] == dEdz2);
    BOOST_TEST(values2[max_index + z_index_0 - z_index_1] == dEdz1);
    BOOST_TEST(values2[z_index_0 - z_index_1] == dEdz1 + (dEdz2 - dEdz1) * (-descend.dz) / ascend.dz);
    for (size_t i = 1; i < std::max(z_index_0 - z_index_1, max_index); i++) {
        if (i < z_index_0 - z_index_1) {
            BOOST_TEST(values2[i] > 0);
            BOOST_TEST(values2[i] > dEdz2);
        }
        if (i < max_index) {
            BOOST_TEST(values2[i + z_index_0 - z_index_1] > values1[i + z_index_0 - z_index_1]);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(atmospheric_entry)

BOOST_AUTO_TEST_CASE(normal_entry)
{
    const std::vector<fcm::CloudDispersionModel> cloud_models {
        fcm::CloudDispersionModel::pancake, fcm::CloudDispersionModel::debrisCloud,
        fcm::CloudDispersionModel::chainReaction
    };
    const std::vector<fcm::ODEsolver> solvers {fcm::ODEsolver::forwardEuler, fcm::ODEsolver::improvedEuler,
                                               fcm::ODEsolver::RK4, fcm::ODEsolver::AB2};
    const std::vector<bool> true_false {true, false};
    const std::vector<double> cloud_mass_fractions {0, 0.5, 1};
    const std::vector<double> strengths {5e3, 1e7};

    for (const double strength : strengths) {
    for (const bool fixed_timestep : true_false) {
    for (const bool flat : true_false) {
    for (const auto cloud_mass_frac : cloud_mass_fractions) {
    for (const auto cloud_model : cloud_models) {
    for (const auto solver : solvers) {
        const auto settings = make_settings(cloud_model, solver, flat, fixed_timestep);
        const auto [params, meteoroid, f, rho_a] = test_params(settings, strength, false, 0, 1, cloud_mass_frac);
        
        const double ground_height = 10e3;
        const double start_height = 70e3;
        const auto [dEdz, result] = fcm::solve_entry(meteoroid, start_height, ground_height,
                                                     rho_a, params, *settings, true);
        if (strength < 1e5) {
            if (cloud_mass_frac < 1) {
                BOOST_TEST(result.size() > 2);
            }
            else {
                BOOST_TEST(result.size() == 2);
            }
        } else {
            BOOST_TEST(!result.empty());
        }

        for (const auto& [info, ts] : result) {
            BOOST_TEST(ts.back()[0] > 1);
            BOOST_TEST(!info.escape);
            if (info.is_cloud) {
                if (ts.size() > 2 && cloud_model != fcm::CloudDispersionModel::chainReaction) {
                    BOOST_TEST(ts.back()[7] > ts.front()[7]);
                }
            } else {
                BOOST_TEST(ts.back()[7] <= ts.front()[7], tt::tolerance(1e-2));
            }

            if (!info.parent_ids.empty()) {
                BOOST_TEST(ts.front()[4] < start_height);
            }
            if (!info.daughter_ids.empty()) {
                BOOST_TEST(ts.back()[10] == info.strength, tt::tolerance(1e-8));
            }

            if (info.impact) {
                BOOST_TEST(ts.back()[4] == ground_height, tt::tolerance(1e-8));
            } else {
                BOOST_TEST(ts.back()[4] > ground_height);
            }
        }
    }}}}}}
}

BOOST_AUTO_TEST_CASE(does_not_finish_bug)
{
    const double big_strength = 1e4 * 1e3;
    const double small_strength = 1e4;
    const auto middle_strength = std::sqrt(big_strength * small_strength);
    const double density = 3e3;
    const double weibull_exp = 0.25;
    const double cloud_frac = 0.5;
    const std::vector<double> mass_fracs {0.5, 0.5};
    const double radius = 1;
    const double angle = 20.0 / 180.0 * M_PI;
    const double velocity = 12e3;

    std::list<fcm::StructuralGroup> groups {
        fcm::StructuralGroup(1.0/3, 9, small_strength, density, cloud_frac, weibull_exp, mass_fracs),
        fcm::StructuralGroup(1.0/3, 3, middle_strength, density, cloud_frac, weibull_exp, mass_fracs),
        fcm::StructuralGroup(1.0/3, 1, big_strength, density, cloud_frac, weibull_exp, mass_fracs)
    };

    const fcm::FCM_settings settings(fcm::CloudDispersionModel::debrisCloud, fcm::ODEsolver::AB2,
                                     1e-2, 1000, false, false, false, fcm::AblationModel::meteoroid_const_r);
    const fcm::FCM_crater_coeff crater_coeff(0.75, 1.5e3, 1e4, 0.15, 1, 1.1, 0.4, 1.0/3.0, 1.3);
    const fcm::FCM_params p(3.711, 3389.5e3, 1e-8, 1, 5e-4, 330e6, 1, 1, 0, 0.9, crater_coeff);

    std::vector<double> h (51);
    for (int i=0; i<h.size(); i++) h[i] = 2000*i;
    const std::vector<double> rho {
        0.015719434275203496, 0.013095743687311407, 0.010919849892539303, 0.0091274983680373536,
        0.0076973689930996177, 0.0065043355813630358, 0.0054636043368810018, 0.004575959784112617,
        0.0038139179978220442, 0.0031700026951084568, 0.0026271505725637689, 0.0021687120220533467,
        0.0017824630201062656, 0.0014596646552764232, 0.0011922805308827709, 0.00097132660923163151, 
        0.00078922380830628099, 0.00063946006129376543, 0.0005166379565995111, 0.00041664205762279136,
        0.00033434634445916395, 0.00026751427464996168, 0.00021289861995351538, 0.00016898955506652885,
        0.00013369934924692373, 0.00010550166254246331, 8.3029201061652646e-05, 6.5048924106220573e-05,
        5.0809883770948128e-05, 3.9602037382483893e-05, 3.0796167009546581e-05, 2.3849039771937491e-05,
        1.841261119107849e-05, 1.4193727151894847e-05, 1.0924554003355761e-05, 8.3692150399858069e-06,
        6.4076135163813281e-06, 4.8873327181735461e-06, 3.7287353563142335e-06, 2.8457467091401548e-06,
        2.1725676051872975e-06, 1.6591259157317281e-06, 1.2674146208182521e-06, 9.6848516873657509e-07,
        7.4028886231187771e-07, 5.6605789862275051e-07, 4.3295623543280449e-07, 3.312354190435926e-07,
        2.5350219750034169e-07, 1.9407093647912575e-07, 1.4861669922642247e-07
    };
    const fcm::AtmosphericDensity rho_a(std::move(h), rho);

    const fcm::Meteoroid impactor(density, velocity, radius, angle, small_strength, 0, std::move(groups));

    const auto result = fcm::solve_entry(impactor, 100e3, 0, rho_a, p, settings, false, 72129);
}

BOOST_AUTO_TEST_CASE(very_shallow_angle)
{
    const auto settings = make_settings();
    auto [params, meteoroid, f, rho_a] = test_params(settings, 1e5, false);

    const double entry_height = 70e3;
    meteoroid.angle = 0.02;
    meteoroid.velocity = 20e3;
    const auto [dEdz, result] = fcm::solve_entry(meteoroid, entry_height, 0, rho_a, params,
                                                 *settings, true);

    BOOST_TEST(result.size() == 1);
    const auto final_state = result.front().second.back();

    // print_result(result);

    BOOST_TEST(final_state[0] > 1);
    BOOST_TEST(final_state[4] >= entry_height);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(craters)

BOOST_AUTO_TEST_CASE(it_runs)
{
    const fcm::FragmentInfo info0 {10e3, 3e3, 0, std::vector<fcm::id_type>{},
                                   std::vector<fcm::id_type>{1, 2, 3}, false, false, false};
    const fcm::FragmentInfo info1 {30e3, 3e3, 1, std::vector<fcm::id_type>{0},
                                   std::vector<fcm::id_type>{}, false, true, false};
    const fcm::FragmentInfo info2 {33e3, 3e3, 2, std::vector<fcm::id_type>{0},
                                   std::vector<fcm::id_type>{}, false, true, false};
    const fcm::FragmentInfo info3 {10e3, 3e3, 3, std::vector<fcm::id_type>{0},
                                   std::vector<fcm::id_type>{}, true, true, false};
    
    const std::array<double, data_size> final_state0 {10, 20e3, 10e3, 1, 20e3, 20e3, 0, 1, 5e3, 1, 10e3};
    const std::array<double, data_size> final_state1 {13, 7e3, 2e3, 1.3, 0, 23e3, 50, 0.7, 5e3, 1, 27e3};
    const std::array<double, data_size> final_state2 {13, 7e3, 2e3, 1.3, 0, 23e3, -50, 0.7, 5e3, 1, 27e3};
    const std::array<double, data_size> final_state3 {15, 4e3, 200, 1.55, 0, 21e3, 0, 1.72, 185, 1, 12e3};

    const std::list<std::pair<fcm::FragmentInfo, std::list<std::array<double, data_size>>>> fragment_data {
        std::make_pair(info0, std::list<std::array<double, data_size>> {final_state0}),
        std::make_pair(info1, std::list<std::array<double, data_size>> {final_state1}),
        std::make_pair(info2, std::list<std::array<double, data_size>> {final_state2}),
        std::make_pair(info3, std::list<std::array<double, data_size>> {final_state3}),
    };

    const auto settings = make_settings(fcm::CloudDispersionModel::pancake);
    const auto [params, m, f, r] = test_params(settings, 1e6, false);

    const auto craters = fcm::calculate_craters(fragment_data, params, *settings);
    BOOST_TEST(craters.size() <= 3);

    const std::vector<fcm::id_type> valid_ids {1, 2, 3};
    for (const auto& crater : craters) {
        BOOST_TEST(crater.r > 0);
        BOOST_TEST(crater.x >= 21e3);
        for (const auto id : crater.fragment_ids) {
            BOOST_TEST((std::find(valid_ids.cbegin(), valid_ids.cend(), id) != valid_ids.cend()));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
