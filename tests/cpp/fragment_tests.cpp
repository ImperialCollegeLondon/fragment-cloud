#define BOOST_TEST_MODULE fragment class
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include "fragment.hpp"
#include "atmospheric_density.hpp"
#include "ode.hpp"
#include "parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace fcm;

struct TestParams {
    const double radius = 0.3;
    const double density = 3300;
    const double mass = sphere_m(density, radius);
    const double velocity = 200;
    const double theta = 0.5;
    const double z = 20e3;
    const double strength = 5e6;
    const double strength_scaler = 0.25;
    const double cloud_frac = 0.6;
    const id_type id = 3;
    const double t = 4;
    const double x = 40e3;
    const double y = 50;
    const double cos_phi = std::cos(-0.2);
    const double sin_phi = std::sin(-0.2);

    const std::shared_ptr<const FCM_settings> settings = std::make_shared<const FCM_settings>(
        CloudDispersionModel::debrisCloud, ODEsolver::AB2, 1e-2, 1000, true, false, false
    );
    const FCM_crater_coeff crater_coeff = FCM_crater_coeff(0.75, 1.5e3, 1e4, 0.15, 1, 1.1, 0.4,
                                                           0.33, 1.3);
    const std::shared_ptr<const FCM_params> parameters = std::make_shared<const FCM_params>(
        5, 6371e3, 2e-9, 0.5, 5e-4, 330e6, 0.2, 1.5, 1, 0.9, crater_coeff
    );
    std::shared_ptr<const AtmosphericDensity> rho_a;

    const offset test_delta = offset {1, 0.5, -0.1, 20, -30, -50, 0.1, 0.02};

    TestParams() {
        const std::vector<double> h {0, 10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3};
        std::vector<double> d(h.size());
        std::transform(h.cbegin(), h.cend(), d.begin(),
                       [](const double z){ return 1.2 * std::exp(-z/8e3); });
        this->rho_a = std::make_shared<const AtmosphericDensity>(std::move(h), d);
    }
};

auto test_fragment(std::list<SubFragment> subfragments=std::list<SubFragment>(),
                   double cloud_frac=-1) {
    const TestParams p;
    return Fragment (p.mass, p.velocity, p.radius, p.theta, p.z, p.strength,
                     cloud_frac < 0 ? p.cloud_frac : cloud_frac,
                     p.density, p.rho_a, p.parameters, p.settings, p.id, std::move(subfragments),
                     p.t, p.x, p.y, p.cos_phi, p.sin_phi);
}

void compare_states(const offset& state1, const offset& state2,
                    bool compare_d2r=true, const double tol=1e-8) {
    BOOST_TEST(state1.dv == state2.dv, tt::tolerance(tol));
    BOOST_TEST(state1.dm == state2.dm, tt::tolerance(tol));
    BOOST_TEST(state1.dtheta == state2.dtheta, tt::tolerance(tol));
    BOOST_TEST(state1.dx == state2.dx, tt::tolerance(tol));
    BOOST_TEST(state1.dy == state2.dy, tt::tolerance(tol));
    BOOST_TEST(state1.dz == state2.dz, tt::tolerance(tol));
    BOOST_TEST(state1.dr == state2.dr, tt::tolerance(tol));
    if (compare_d2r) {  
        BOOST_TEST(state1.d2r == state2.d2r, tt::tolerance(tol));
    }
}

BOOST_AUTO_TEST_SUITE(offset_struct)

BOOST_AUTO_TEST_CASE(scalar_multiplication, * utf::tolerance(1e-8))
{
    offset delta {1, 0.5, -0.1, 20, -30, -50, 0.1, 0.02};
    const double multiplier = 2;
    offset expected {2, 1, -0.2, 40, -60, -100, 0.2, 0.04};

    const auto product = delta * multiplier;
    
    BOOST_TEST_REQUIRE(product.dv == expected.dv);
    BOOST_TEST_REQUIRE(product.dm == expected.dm);
    BOOST_TEST_REQUIRE(product.dtheta == expected.dtheta);
    BOOST_TEST_REQUIRE(product.dx == expected.dx);
    BOOST_TEST_REQUIRE(product.dy == expected.dy);
    BOOST_TEST_REQUIRE(product.dz == expected.dz);
    BOOST_TEST_REQUIRE(product.dr == expected.dr);
    BOOST_TEST_REQUIRE(product.d2r == expected.d2r);
}

BOOST_AUTO_TEST_CASE(add_operator, * utf::tolerance(1e-8))
{
    offset delta1 {1, 0.5, -0.1, 20, -30, -50, 0.1, 0.02};
    const offset delta2 {0.2, 1, 0.2, -100, -30, 10, -0.01, 0.04};
    offset delta3 = delta1 * (-1);
    const offset sum = {1.2, 1.5, 0.1, -80, -60, -40, 0.09, 0.06};

    const auto sum1 = delta1 + delta2;
    compare_states(sum1, sum);

    delta1 += delta2;
    compare_states(delta1, sum);

    const auto delta4 = delta1 + std::move(delta3);
    compare_states(delta4, delta2);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(fragment_class)

BOOST_AUTO_TEST_CASE(constructor, * utf::tolerance(1e-8))
{
    const TestParams p;
    const auto frag = test_fragment();

    BOOST_TEST(frag.mass() == p.mass);
    BOOST_TEST(frag.velocity() == p.velocity);
    BOOST_TEST(frag.cos_theta() == std::cos(p.theta));
    BOOST_TEST(frag.sin_theta() == std::sin(p.theta));
    BOOST_TEST(frag.z() == p.z);
    BOOST_TEST(frag.radius() == p.radius);
    BOOST_TEST(frag.dr() == 0);
    BOOST_TEST(!frag.is_cloud());
    BOOST_TEST(frag.strength() == p.strength);
    BOOST_TEST(frag.density() == p.density);
    BOOST_TEST(frag.cos_phi() == p.cos_phi);
    BOOST_TEST(frag.sin_phi() == p.sin_phi);
    BOOST_TEST(frag.dt_prev() == p.settings->precision);
    BOOST_TEST(frag.dt_next() == p.settings->precision);
    BOOST_TEST(frag.dEdz() == 0);
    BOOST_TEST(frag.state().dv == p.velocity);
    BOOST_TEST(frag.state().dm == p.mass);
    BOOST_TEST(frag.state().dtheta == p.theta);
    BOOST_TEST(frag.state().dx == p.x);
    BOOST_TEST(frag.state().dy == p.y);
    BOOST_TEST(frag.state().dz == p.z);
    BOOST_TEST(frag.state().dr == p.radius);
    BOOST_TEST(frag.state().d2r == 0);
}

BOOST_AUTO_TEST_CASE(small_calculations, * utf::tolerance(1e-8))
{
    const TestParams p;
    const auto frag = test_fragment();
    const double rho_a = p.rho_a->calc(p.z);    

    BOOST_TEST(frag.scale_height() == p.rho_a->scale_height(p.z));
    BOOST_TEST(frag.air_density() == rho_a);
    BOOST_TEST(frag.rp() == ram_pressure(rho_a, p.velocity));
    BOOST_TEST(frag.E_kin() == kinetic_energy(p.mass, p.velocity));
    BOOST_TEST(frag.area() == M_PI * std::square(p.radius));
    BOOST_TEST(frag.crater_detectable());
}

BOOST_AUTO_TEST_CASE(save_df_dt_next, * utf::tolerance(1e-8))
{
    const TestParams p;
    auto frag = test_fragment();
    const double dt_next = 200;
    const double dt = 100;

    frag.save_df_prev(p.test_delta)
        .set_dt_next(dt_next)
        .advance_time(dt);

    BOOST_TEST(frag.dt_prev() == dt);
    BOOST_TEST(frag.dt_next() == dt_next);
    compare_states(frag.df_prev(), p.test_delta);
}

BOOST_AUTO_TEST_CASE(data_info, * utf::tolerance(1e-8))
{
    const TestParams p;
    const auto frag = test_fragment();

    const auto data = frag.data();
    const std::array<double, data_size> expected_data {
        p.t, p.mass, p.velocity, p.theta, p.z, p.x, p.y, p.radius, p.density, 0,
        ram_pressure(p.rho_a->calc(p.z), p.velocity)
    };
    for (int i = 0; i < data_size; i++) {
        BOOST_TEST(data[i] == expected_data[i]);
    }
    
    const auto info1 = frag.info(p.z + 1e3, p.z - 1e3);
    BOOST_TEST(info1.strength == p.strength);
    BOOST_TEST(info1.init_density == p.density);
    BOOST_TEST(info1.id == p.id);
    BOOST_TEST(info1.parent_ids.empty());
    BOOST_TEST(info1.daughter_ids.empty());
    BOOST_TEST(!info1.is_cloud);
    BOOST_TEST(!info1.impact);
    BOOST_TEST(!info1.escape);

    const auto info2 = frag.info(p.z, p.z-1e3);
    BOOST_TEST(!info2.impact);
    BOOST_TEST(info2.escape);

    const auto info3 = frag.info(p.z + 1e3, p.z);
    BOOST_TEST(info3.impact);
    BOOST_TEST(!info3.escape);
}

BOOST_AUTO_TEST_CASE(add_offset, * utf::tolerance(1e-8))
{
    const TestParams p;
    auto frag = test_fragment();

    const auto old_state = frag.state();

    auto delta_copy_1 = p.test_delta;
    const auto frag_sum_copy = frag + p.test_delta;
    const auto frag_sum_move = frag + std::move(delta_copy_1);
    
    auto frag_copy = frag;
    frag += p.test_delta;

    auto delta_copy_2 = p.test_delta;
    frag_copy += std::move(delta_copy_2);

    BOOST_TEST(frag.dEdz() > 0);
    compare_states(frag.state(), old_state + p.test_delta);
    compare_states(frag.delta_prev(), p.test_delta);

    BOOST_TEST(frag.dEdz() == frag_sum_copy.dEdz());
    compare_states(frag.state(), frag_sum_copy.state());

    BOOST_TEST(frag.dEdz() == frag_sum_move.dEdz());
    compare_states(frag.state(), frag_sum_move.state());

    BOOST_TEST(frag.dEdz() == frag_copy.dEdz());
    compare_states(frag.state(), frag_copy.state());
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE(breakup)

BOOST_AUTO_TEST_CASE(cloud_only, * utf::tolerance(1e-8))
{
    const TestParams p;
    const double z_start = 100e3;
    const double z_ground = 0;

    auto frag = test_fragment(std::list<SubFragment>(), 1);
    const auto state_before = frag.state();
    const auto frag_data_before = frag.data();
    const auto frag_info_before = frag.info(z_start, z_ground);

    const auto new_fragments = frag.break_apart();

    BOOST_TEST(new_fragments.size() == 1);
    const auto cloud = new_fragments.front();

    compare_states(frag.state(), cloud.state(), false);
    BOOST_TEST(cloud.is_cloud());
    BOOST_TEST(cloud.strength() == frag.strength());
    BOOST_TEST(cloud.density() == frag.density());
    BOOST_TEST(cloud.cos_phi() == frag.cos_phi());
    BOOST_TEST(cloud.sin_phi() == frag.sin_phi());

    BOOST_TEST(cloud.scale_height() == frag.scale_height());
    BOOST_TEST(cloud.air_density() == frag.air_density());
    BOOST_TEST(cloud.rp() == frag.rp());
    BOOST_TEST(cloud.E_kin() == frag.E_kin());
    BOOST_TEST(cloud.area() == frag.area());
    BOOST_TEST(cloud.crater_detectable());

    const auto frag_info = frag.info(z_start, z_ground);
    const auto frag_data = frag.data();

    BOOST_TEST(frag_info_before.strength == frag_info.strength);
    BOOST_TEST(frag_info_before.init_density == frag_info.init_density);
    BOOST_TEST(frag_info_before.id == frag_info.id);
    BOOST_TEST(frag_info_before.is_cloud == frag_info.is_cloud);
    BOOST_TEST(frag_info_before.impact == frag_info.impact);
    BOOST_TEST(frag_info_before.escape == frag_info.escape);

    const auto cloud_data = cloud.data();
    const auto cloud_info = cloud.info(z_start, z_ground);

    BOOST_TEST(frag_info_before.parent_ids.empty());
    BOOST_TEST(frag_info.parent_ids.empty());
    BOOST_TEST(frag_info_before.daughter_ids.empty());
    BOOST_TEST(frag_info.daughter_ids.size() == 1);
    BOOST_TEST(frag_info.daughter_ids.front() == cloud_info.id);
    BOOST_TEST(cloud_info.parent_ids.size() == 1);
    BOOST_TEST(cloud_info.parent_ids.front() == frag_info.id);

    BOOST_TEST(cloud_data[0] == frag_data[0]);
    BOOST_TEST(cloud_info.is_cloud);
    BOOST_TEST(cloud_info.impact == frag_info.impact);
    BOOST_TEST(cloud_info.escape == frag_info.escape);
}

BOOST_AUTO_TEST_CASE(one_fragment_only, * utf::tolerance(1e-8))
{
    const TestParams p;
    const double z_start = 100e3;
    const double z_ground = 0;

    const std::list<SubFragment> subfragments {{1, 10*p.strength, 2*p.density, p.cloud_frac,
                                                p.strength_scaler, std::vector<double> {0.4, 0.6}}};

    auto frag = test_fragment(subfragments, 0);
    const auto state_before = frag.state();
    const auto frag_data_before = frag.data();
    const auto frag_info_before = frag.info(z_start, z_ground);

    const auto new_fragments = frag.break_apart();

    BOOST_TEST(new_fragments.size() == 1);
    const auto subfrag = new_fragments.front();

    const auto subfrag_data = subfrag.data();
    const auto subfrag_info = subfrag.info(z_start, z_ground);

    BOOST_TEST(!subfrag.is_cloud());
    BOOST_TEST(subfrag.strength() == subfragments.front().strength);
    BOOST_TEST(subfrag.density() == subfragments.front().density);
    BOOST_TEST(subfrag.mass() == frag_data_before[1]);
    BOOST_TEST(subfrag.velocity() == frag_data_before[2]);
    BOOST_TEST(subfrag.cos_theta() == frag.cos_theta());
    BOOST_TEST(subfrag.sin_theta() == frag.sin_theta());

    BOOST_TEST(subfrag_data[0] == frag_data_before[0]);
    BOOST_TEST(subfrag.z() == frag.z());
    BOOST_TEST(subfrag_data[5] == frag_data_before[5]);
    BOOST_TEST(subfrag_data[6] == frag_data_before[6]);

    BOOST_TEST(subfrag.cos_phi() == frag.cos_phi());
    BOOST_TEST(subfrag.sin_phi() == frag.sin_phi());
    BOOST_TEST(subfrag.scale_height() == frag.scale_height());
    BOOST_TEST(subfrag.air_density() == frag.air_density());
    BOOST_TEST(subfrag.rp() == frag.rp());
    BOOST_TEST(subfrag.E_kin() == frag.E_kin());
    BOOST_TEST(subfrag.crater_detectable());

    const auto frag_info = frag.info(z_start, z_ground);

    BOOST_TEST(frag_info_before.strength == frag_info.strength);
    BOOST_TEST(frag_info_before.init_density == frag_info.init_density);
    BOOST_TEST(frag_info_before.id == frag_info.id);
    BOOST_TEST(frag_info_before.is_cloud == frag_info.is_cloud);
    BOOST_TEST(frag_info_before.impact == frag_info.impact);
    BOOST_TEST(frag_info_before.escape == frag_info.escape);

    BOOST_TEST(frag_info_before.parent_ids.empty());
    BOOST_TEST(frag_info.parent_ids.empty());
    BOOST_TEST(frag_info_before.daughter_ids.empty());
    BOOST_TEST(frag_info.daughter_ids.size() == 1);
    BOOST_TEST(frag_info.daughter_ids.front() == subfrag_info.id);
    BOOST_TEST(subfrag_info.parent_ids.size() == 1);
    BOOST_TEST(subfrag_info.parent_ids.front() == frag_info.id);

    BOOST_TEST(!subfrag_info.is_cloud);
    BOOST_TEST(subfrag_info.impact == frag_info.impact);
    BOOST_TEST(subfrag_info.escape == frag_info.escape);
}

BOOST_AUTO_TEST_CASE(fragment_and_cloud, * utf::tolerance(1e-8))
{
    const TestParams p;
    const double z_start = 100e3;
    const double z_ground = 0;

    const std::list<SubFragment> subfragments {{1, 10*p.strength, 2*p.density, p.cloud_frac,
                                                p.strength_scaler, std::vector<double> {0.4, 0.6}}};
    auto frag = test_fragment(subfragments);
    const auto state_before = frag.state();
    const auto frag_data_before = frag.data();
    const auto frag_info_before = frag.info(z_start, z_ground);

    const auto new_fragments = frag.break_apart();

    BOOST_TEST(new_fragments.size() == 2);
    const auto subfrag = new_fragments.front();
    const auto cloud = new_fragments.back();

    BOOST_TEST(!subfrag.is_cloud());
    BOOST_TEST(cloud.is_cloud());
    BOOST_TEST(subfrag.mass() == p.mass * (1 - p.cloud_frac));
    BOOST_TEST(cloud.mass() == p.mass * p.cloud_frac);

    const auto vx_subfrag = vx(subfrag.velocity(), subfrag.cos_theta(), subfrag.cos_phi());
    const auto vx_cloud = vx(cloud.velocity(), cloud.cos_theta(), cloud.cos_phi());
    const auto vx_frag = vx(p.velocity, std::cos(p.theta), p.cos_phi);

    const auto vy_subfrag = vy(subfrag.velocity(), subfrag.cos_theta(), subfrag.sin_phi());
    const auto vy_cloud = vy(cloud.velocity(), cloud.cos_theta(), cloud.sin_phi());
    const auto vy_frag = vy(p.velocity, std::cos(p.theta), p.sin_phi);

    const auto vz_subfrag = vz(subfrag.velocity(), subfrag.sin_theta());
    const auto vz_cloud = vz(cloud.velocity(), cloud.sin_theta());
    const auto vz_frag = vz(p.velocity, std::sin(p.theta));

     /* momentum conservation */
    BOOST_TEST(p.mass * vx_frag == subfrag.mass() * vx_subfrag + cloud.mass() * vx_cloud);
    BOOST_TEST(p.mass * vy_frag == subfrag.mass() * vy_subfrag + cloud.mass() * vy_cloud);
    BOOST_TEST(p.mass * vz_frag == subfrag.mass() * vz_subfrag + cloud.mass() * vz_cloud);

    /* transverse velocity */
    const auto vt = std::hypot(vx_subfrag - vx_cloud, vy_subfrag - vy_cloud, vz_subfrag - vz_cloud);
    BOOST_TEST(vt == V_T(p.parameters->frag_velocity_coeff, cloud.radius(), subfrag.radius(),
                         subfrag.air_density(), subfrag.density(), p.velocity));
}

BOOST_AUTO_TEST_CASE(multiple_fragments, * utf::tolerance(1e-8))
{
    const TestParams p;
    const double z_start = 100e3;
    const double z_ground = 0;

    const std::list<SubFragment> subfragments {
        {0.1, 10*p.strength, 2*p.density, p.cloud_frac, p.strength_scaler, std::vector<double> {0.4, 0.6}},
        {0.3, 5*p.strength, 1.2*p.density, p.cloud_frac, p.strength_scaler, std::vector<double> {0.4, 0.6}},
        {0.6, 20*p.strength, 2.5*p.density, p.cloud_frac, p.strength_scaler, std::vector<double> {0.4, 0.6}}
    };
    auto frag = test_fragment(subfragments, 0);
    const auto state_before = frag.state();
    const auto frag_data_before = frag.data();
    const auto frag_info_before = frag.info(z_start, z_ground);

    const auto new_fragments = frag.break_apart();

    BOOST_TEST(new_fragments.size() == 3);

    std::array<double, 3> momentum {0, 0, 0};
    std::list<double> mass_fractions;
    for (const auto& subfrag : new_fragments) {
        BOOST_TEST(!subfrag.is_cloud());
        mass_fractions.push_back(subfrag.mass() / p.mass);
        momentum[0] += subfrag.mass() * vx(subfrag.velocity(), subfrag.cos_theta(), subfrag.cos_phi());
        momentum[1] += subfrag.mass() * vy(subfrag.velocity(), subfrag.cos_theta(), subfrag.sin_phi());
        momentum[2] += subfrag.mass() * vz(subfrag.velocity(), subfrag.sin_theta());
    }

    for (const auto frac : mass_fractions) {
        BOOST_TEST((std::find_if(subfragments.cbegin(), subfragments.cend(),
                                [&](const auto& sf){ return std::abs(sf.mass_fraction - frac) < 1e-8; })
                    != subfragments.cend()));
    }

    /* momentum conservation */
    BOOST_TEST(p.mass * vx(p.velocity, std::cos(p.theta), p.cos_phi) == momentum[0]);
    BOOST_TEST(p.mass * vy(p.velocity, std::cos(p.theta), p.sin_phi) == momentum[1]);
    BOOST_TEST(p.mass * vz(p.velocity, std::sin(p.theta)) == momentum[2]);
}

BOOST_AUTO_TEST_CASE(tiny_fragment, * utf::tolerance(1e-8))
{
    const TestParams p;
    const double z_start = 100e3;
    const double z_ground = 0;
    const std::list<SubFragment> subfragments {
        {1e-7, 10*p.strength, 2*p.density, p.cloud_frac, p.strength_scaler, std::vector<double> {0.4, 0.6}},
        {1-1e-7, 5*p.strength, 1.2*p.density, p.cloud_frac, p.strength_scaler, std::vector<double> {0.4, 0.6}}
    };
    auto frag = test_fragment(subfragments, 1e-7);
    const auto new_fragments = frag.break_apart();
    BOOST_TEST_REQUIRE(new_fragments.size() == 1);
}

BOOST_AUTO_TEST_SUITE_END()