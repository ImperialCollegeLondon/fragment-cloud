#include "fcm.hpp"

#include "atmospheric_density.hpp"
#include "fragment.hpp"
#include "ode.hpp"
#include "parameters.hpp"
#include "solvers.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

#include <iostream>
#include <iomanip>

using namespace fcm;

// TODO: Avoid ifs/switch in here somehow?
// TODO: Declare noexcept, since cloud cannot be anything else, so no exception should be thrown?
offset fcm::dfdt(const Fragment& fragment, const FCM_params& params, const FCM_settings& settings) {
    const auto A = fragment.area();
    const auto cos_theta = fragment.cos_theta();
    const auto sin_theta = fragment.sin_theta();

    // if
    const auto grav_acc = settings.flat_planet ? g_flat(params.g0) : g(params.g0, fragment.z(), params.Rp);

    offset result;
    result.dv = dvdt(params.drag_coeff, fragment.rp(), A, fragment.mass(), grav_acc, sin_theta);
    result.dm = dmdt(params.ablation_coeff, fragment.rp(), A, fragment.velocity());
    result.dz = dzdt(fragment.velocity(), sin_theta);

    if (settings.flat_planet) {
        result.dtheta = dthetadt_flat(grav_acc, cos_theta, fragment.velocity(),
                                      params.lift_coeff, fragment.rp(), A, fragment.mass());
        result.dx = dxdt_flat(fragment.velocity(), cos_theta, fragment.cos_phi());
        result.dy = dydt_flat(fragment.velocity(), cos_theta, fragment.sin_phi());
    }
    else {
        result.dtheta = dthetadt(grav_acc, cos_theta, fragment.velocity(), params.lift_coeff,
                                 fragment.rp(), A, fragment.mass(), params.Rp, fragment.z());
        result.dx = dxdt(fragment.velocity(), cos_theta, fragment.cos_phi(), params.Rp, fragment.z());
        result.dy = dydt(fragment.velocity(), cos_theta, fragment.sin_phi(), params.Rp, fragment.z());
    }

    switch(settings.cloud_dispersion_model) {
        case CloudDispersionModel::pancake: {
            result.d2r = d2rdt2_pancake(fragment.rp(), fragment.strength(), params.cloud_disp_coeff,
                                        fragment.density(), fragment.radius());
            result.dr = fragment.dr();
        }
        break;

        case CloudDispersionModel::debrisCloud: {
            result.d2r = 0;
            result.dr = drdt_debriscloud(fragment.rp(), fragment.strength(), params.cloud_disp_coeff,
                                         fragment.air_density(), fragment.density(),
                                         fragment.velocity());
        }
        break;
        
        case CloudDispersionModel::chainReaction: {
            result.d2r = 0;
            result.dr = drdt_chainreaction(fragment.rp(), fragment.strength(), fragment.radius(),
                                           fragment.mass(), result.dm, params.cloud_disp_coeff,
                                           fragment.density());
        }
        break;

        default:
            throw std::invalid_argument("Invalid cloud dispersion model");
    }

    return result;
}

class dEdz_Fragment {
public:
    using container_t = std::deque<double>;
    using index_t = container_t::size_type;

    dEdz_Fragment(const Fragment& fragment, const double z_start, const double z_ground,
                  const double dh) : z_start_(z_start), dh_(dh) {
        assert(z_ground < z_start);

        this->z_index_0_ = this->z_index_ = std::floor((z_start - fragment.z()) / dh);
        this->z_index_max_ = std::floor((z_start - z_ground) / dh) + 1;
        this->dEdz_prev_ = 0;
    }

    inline auto values() const noexcept { return this->values_; }
    inline auto z_index_0() const noexcept { return this->z_index_0_ - this->negative_extension_; }

    void add_dedz(const Fragment& fragment) {
        const auto tmp = (this->z_start_ - fragment.z()) / this->dh_;
        const index_t z_index_new = std::max(fragment.sin_theta() > 0 ? std::floor(tmp)
                                                                      : std::ceil(tmp), 0.0);
        if (z_index_new > this->z_index_) {
            for (auto i = this->z_index_ + 1; i <= std::min(z_index_new, this->z_index_max_); i++) {
                const auto z_target = this->z_start_ - i * this->dh_;
                const auto dEdz = fragment.dEdz() - (fragment.dEdz() - this->dEdz_prev_)
                                    * (fragment.z() - z_target) / fragment.delta_prev().dz;
                if (i - this->z_index_0_ >= this->values_.size() - this->negative_extension_) {
                    this->values_.push_back(dEdz);
                } else {
                    this->values_[i - this->z_index_0_ + this->negative_extension_] += dEdz;
                }
            }
        }
        else if (z_index_new < this->z_index_) {
            for (auto i = long(this->z_index_) - 1; i >= z_index_new && i >= 0; i--) {
                const auto z_target = this->z_start_ - i * this->dh_;
                const auto dEdz = fragment.dEdz() - (fragment.dEdz() - this->dEdz_prev_)
                    * (fragment.z() - z_target) / fragment.delta_prev().dz;
                if (i - this->z_index_0_ >= this->values_.size() - this->negative_extension_) {
                    if (this->z_index_0_ == this->negative_extension_) break;
                    this->values_.push_front(dEdz);
                    this->negative_extension_++;
                } else {
                    this->values_[i - this->z_index_0_ + this->negative_extension_] += dEdz;
                }
            }
        }
        this->z_index_ = z_index_new;
        this->dEdz_prev_ = fragment.dEdz();
    }

private:
    double z_start_, dh_, dEdz_prev_;
    index_t z_index_, z_index_0_, z_index_max_;
    index_t negative_extension_ = 0;
    container_t values_ {};
};

auto _solve_fragment(Fragment&& fragment, const double z_start, const double z_ground,
                     const bool calculate_dEdz, const FCM_params& params, const FCM_settings& settings,
                     const std::function<offset(const Fragment&)>& df,
                     const std::function<Fragment(Fragment&&)>& step) {

    std::list<std::array<double, data_size>> timeseries;
    if (settings.record_data) {
        timeseries.push_back(fragment.data());
    }
    const auto E_kin_start = fragment.E_kin();

    // First timestep with RK4 solver
    const auto df_prev = df(fragment);
    // TODO: maybe base this on velocity / strength as well?
    const auto dt = settings.precision * (settings.fixed_timestep ? 1.0 : 1e-4);
    fragment = fcm::RK4(std::move(fragment), df, dt);
    fragment.advance_time(dt).save_df_prev(std::move(df_prev)).set_dt_next(dt);

    unsigned int iter = 1;
    bool fragmentation_happened = false;
    dEdz_Fragment dEdz_vector(fragment, z_start, z_ground, settings.dh);

    while (iter < settings.max_iterations) {
        assert(fragment.velocity() >= 0);
        assert(fragment.mass() > 0);
        assert(-M_PI <= fragment.state().dtheta && fragment.state().dtheta <= M_PI);
        assert(fragment.radius() > 0);

        // fill dEdz vector
        if (calculate_dEdz) {
            dEdz_vector.add_dedz(fragment);
        }

        // check if impact
        if (fragment.z() < z_ground) {
            const auto dt = ((z_ground - fragment.z()) / fragment.delta_prev().dz);
            fragment += fragment.delta_prev() * dt;
            fragment.advance_time(dt);
            break;
        }

        // check if escape
        if (fragment.z() >= z_start) break;

        // check if too small to produce visible crater
        if (!fragment.crater_detectable()) break;

        // check if cloud has depleted all kinetic energy => stop simulation
        if (fragment.is_cloud() && fragment.E_kin() < settings.cloud_stopping_criterion * E_kin_start) {
            break;
        }

        // check if fragment strength is exceeded => break up
        // TODO: make this a function or a class for testing
        if (!fragment.is_cloud() && fragment.rp() > fragment.strength()) {
            fragment.backtrack_strength();
            fragmentation_happened = true;
            break;
        }

        if (settings.record_data) {
            timeseries.push_back(fragment.data());
        }

        fragment = step(std::move(fragment));
        iter++;
    }

    if (iter >= settings.max_iterations) {
        throw std::runtime_error("Max iterations exceeded");
    }

    timeseries.push_back(fragment.data());
    const auto info = fragment.info(z_start, z_ground);

    std::list<Fragment> daughter_fragments;
    if (fragmentation_happened) {
        daughter_fragments = fragment.break_apart();
    }

    return std::make_tuple(info, std::make_pair(dEdz_vector.z_index_0(), dEdz_vector.values()),
                           timeseries, daughter_fragments);
}

auto _adaptive_timestep(const double tolerance, const offset& delta_scale, const offset& delta,
                        const double dt, const offset& delta_prev, const double dt_prev,
                        const double length_scale, const double angle_scale=1) {
    
    const auto error = delta + delta_prev * (-dt/dt_prev);
    double relative_error = 0.0;
    relative_error += std::square(error.dv / delta_scale.dv);
    relative_error += std::square(error.dm / delta_scale.dm);
    relative_error += std::square(error.dtheta / angle_scale);
    relative_error += std::square(error.dr / delta_scale.dr);
    relative_error += std::square(error.dx / length_scale);
    relative_error += std::square(error.dy / length_scale);
    relative_error += std::square(error.dz / length_scale);
    relative_error = std::sqrt(relative_error);

    const auto dt_next = dt * (1 + 0.1 * (std::log2(1 + tolerance / relative_error) - 1));

    return std::make_pair(dt_next, relative_error / tolerance);
}

std::pair<
    std::vector<double>,
    std::list<std::pair<
        FragmentInfo,
        std::list<std::array<double, data_size>>
    >>
> fcm::solve_entry(const Meteoroid& impactor, const double z_start, const double z_ground,
                   const AtmosphericDensity& rho_a, const FCM_params& params, 
                   const FCM_settings& settings, const bool calculate_dEdz, const id_type seed) {

    if (z_start <= 0) throw std::invalid_argument("z_start must be > 0");
    if (z_start <= z_ground) throw std::invalid_argument("z_start must be > z_ground");
    if (z_ground <= -params.Rp) throw std::invalid_argument("z_ground must be > -Rp");

    const std::function<offset(const Fragment&)> df = [&](const Fragment& fragment){
        return fcm::dfdt(fragment, params, settings);
    };

    std::function<Fragment(Fragment&&)> step;
    switch (settings.ode_solver) {
        case ODEsolver::forwardEuler: {
            if (settings.fixed_timestep) {
                step = [&](Fragment&& frag){
                    const auto dt = frag.dt_prev();
                    auto result = fcm::forward_euler(std::move(frag), df, dt);
                    result.advance_time(dt);
                    return result;
                };
            } else {
                step = [&](Fragment&& frag){
                    auto dt = frag.dt_next();
                    while (true) {
                        auto result = fcm::forward_euler(frag, df, dt);
                        const auto [dt_next, error_factor] = _adaptive_timestep(
                            settings.precision, frag.state(), result.delta_prev(), dt,
                            frag.delta_prev(), frag.dt_prev(), 100*params.min_crater_radius
                        );
                        if (error_factor < 2) {
                            result.advance_time(dt).set_dt_next(dt_next);
                            return result;
                        }
                        dt = std::sqrt(dt_next * dt);
                    }
                };
            }
            break;
        }
        case ODEsolver::improvedEuler: {
            if (settings.fixed_timestep) {
                step = [&](Fragment&& frag) {
                    const auto dt = frag.dt_prev();
                    auto result = fcm::improved_euler(std::move(frag), df, dt);
                    result.advance_time(dt);
                    return result;
                };
            } else {
                step = [&](Fragment&& frag){
                    auto dt = frag.dt_next();
                    while (true) {
                        auto result = fcm::improved_euler(frag, df, dt);
                        const auto [dt_next, error_factor] = _adaptive_timestep(
                            settings.precision, frag.state(), result.delta_prev(), dt,
                            frag.delta_prev(), frag.dt_prev(), 100*params.min_crater_radius
                        );
                        if (error_factor < 2) {
                            result.advance_time(dt).set_dt_next(dt_next);
                            return result;
                        }
                        dt = std::sqrt(dt_next * dt);
                    }
                };
            }
            break;
        }
        case ODEsolver::RK4: {
            if (settings.fixed_timestep) {
                step = [&](Fragment&& frag) {
                    const auto dt = frag.dt_prev();
                    auto result = fcm::RK4(std::move(frag), df, dt);
                    result.advance_time(dt);
                    return result;
                };
            } else {
                step = [&](Fragment&& frag){
                    auto dt = frag.dt_next();
                    while (true) {
                        auto result = fcm::RK4(frag, df, dt);
                        const auto [dt_next, error_factor] = _adaptive_timestep(
                            settings.precision, frag.state(), result.delta_prev(), dt,
                            frag.delta_prev(), frag.dt_prev(), 100*params.min_crater_radius
                        );
                        if (error_factor < 2) {
                            result.advance_time(dt).set_dt_next(dt_next);
                            return result;
                        }
                        dt = std::sqrt(dt_next * dt);
                    }
                };
            }
            break;
        }
        case ODEsolver::AB2: {
            if (settings.fixed_timestep) {
                step = [&](Fragment&& frag) {
                    const auto dt = frag.dt_prev();
                    const auto df_prev = frag.df_prev();
                    auto [result, df_current] = fcm::AB2(std::move(frag), std::move(df_prev), df, dt, dt);
                    result.save_df_prev(std::move(df_current)).advance_time(dt);
                    return result;
                };
            } else {
                step = [&](Fragment&& frag){
                    auto dt = frag.dt_next();
                    while (true) {
                        auto [result, df_current] = fcm::AB2(frag, frag.df_prev(), df, dt, frag.dt_prev());
                        const auto [dt_next, error_factor] = _adaptive_timestep(
                            settings.precision, frag.state(), result.delta_prev(), dt,
                            frag.delta_prev(), frag.dt_prev(), 100*params.min_crater_radius
                        );
                        if (error_factor < 2) {
                            result.advance_time(dt).set_dt_next(dt_next)
                                .save_df_prev(std::move(df_current));
                            return result;
                        }
                        dt = std::sqrt(dt_next * dt);
                    }
                };
            }
            break;
        }
        default:
            throw std::invalid_argument("Invalid ODE solver");
    }

    std::list<SubFragment> impactor_structure;
    for (const auto& group : impactor.structural_groups) {
        SubFragment s {group.mass_fraction / group.pieces, group.strength, group.density,
                       group.cloud_mass_frac, group.strength_scaler, group.fragment_mass_fractions};
        for (unsigned int i=0; i<group.pieces; i++) {
            impactor_structure.push_back(s);
        }
    }

    std::list<std::pair<FragmentInfo, std::list<std::array<double, data_size>>>> solutions;
    std::stack<Fragment> fragments;
    std::vector<double> dEdz;
    if (calculate_dEdz) {
        dEdz.resize(std::ceil((z_start - z_ground) / settings.dh + 1), 0);
    }
    // Fragment ctor
    fragments.push(Fragment(impactor.mass(), impactor.velocity, impactor.radius, impactor.angle,
                            z_start, impactor.strength, impactor.cloud_mass_frac,
                            impactor.density, std::make_shared<const AtmosphericDensity>(rho_a),
                            std::make_shared<const FCM_params>(params),
                            std::make_shared<const FCM_settings>(settings),
                            seed, std::move(impactor_structure)));

    while (!fragments.empty()) {
        auto [info, dEdz_fragment, timeseries, daughter_fragments] = _solve_fragment(
            std::move(fragments.top()), z_start, z_ground, calculate_dEdz, params, settings, df, step
        );
        fragments.pop();
        for (auto& fragment : daughter_fragments) {
            fragments.push(std::move(fragment));
        }
        solutions.emplace_back(std::move(info), std::move(timeseries));
        if (calculate_dEdz) {
            assert(dEdz_fragment.first >= 0);
            assert(dEdz_fragment.first + dEdz_fragment.second.size() <= dEdz.size());
            for (size_t i = 0; i < dEdz_fragment.second.size(); i++) {
                dEdz[i + dEdz_fragment.first] += dEdz_fragment.second[i];
            }
        }
    }

    return std::make_pair(dEdz, solutions);
}

inline auto _dist(const Crater& c1, const Crater& c2) {
    return std::hypot(c1.x - c2.x, c1.y - c2.y);
}

constexpr auto _mean_coordinate(double c1, double w1, double c2, double w2) {
    return (c1 * w1 + c2 * w2) / (w1 + w2);
}

std::list<Crater> fcm::calculate_craters(
    const std::list<std::pair<FragmentInfo, std::list<std::array<double, data_size>>>>& fragments,
    const FCM_params& params, const FCM_settings& settings
) {
    std::list<Crater> craters;
    for (const auto& [info, ts] : fragments) {
        if (info.impact) {
            const auto final_state = ts.back();
            const auto radius = Holsapple_crater_radius(
                params.ground_strength, params.ground_density, final_state[2]*std::sin(final_state[3]),
                settings.flat_planet ? g_flat(params.g0) : g(params.g0, final_state[4], params.Rp),
                final_state[7], final_state[8], params.K1, params.nu, params.mu, params.K2,
                params.Kr, final_state[1], params.rim_factor
            );
            if (final_state[7] > M_SQRT1_2*radius) {
                if (info.is_cloud) {
                    // TODO: std::cout or std::cerr ?
                    // std::cerr << "Warning: Crater radius calculated from debris cloud impact is "
                    //           << "small compared to debris cloud radius. Ratio = "
                    //           << std::setprecision(3) << final_state[7] / radius << std::endl;
                }
                else {
                    // throw std::runtime_error("Bug: Crater radius is very small compared to the fragment size");
                    std::cerr << "Warning: Crater radius calculated from fragment impact is small "
                              << "compared to fragment radius. Ratio = "
                              << std::setprecision(3) << final_state[7] / radius << std::endl;
                }
            }
            craters.push_back(Crater {final_state[5], final_state[6], radius,
                                      std::list<id_type> {info.id} });
        }
    }
    if (craters.empty()) {
        return craters;
    }
    bool there_has_been_a_change;
    do {
        there_has_been_a_change = false;
        craters.sort([](const auto& c1, const auto& c2){ return c1.x < c2.x; });
        const auto r_max = std::max_element(
            craters.cbegin(), craters.cend(),
            [](const auto& c1, const auto& c2){ return c1.r < c2.r; }
        )->r;

        for (auto it = craters.begin(); it != craters.cend(); it++) {
            auto it2 = it;
            while (++it2 != craters.end() && it2->x - it->x < r_max) {
                if (_dist(*it, *it2) < std::max(it2->r, it->r)) {
                    const auto r1cubed = std::cube(it->r);
                    const auto r2cubed = std::cube(it2->r);
                    it->x = _mean_coordinate(it->x, r1cubed, it2->x, r2cubed);
                    it->y = _mean_coordinate(it->y, r1cubed, it2->y, r2cubed);
                    it->r = std::cbrt(r1cubed + r2cubed);
                    it->fragment_ids.splice(it->fragment_ids.cend(), it2->fragment_ids);

                    craters.erase(it2);
                    it2 = it;
                    there_has_been_a_change = true;
                }
            }
        }
    } while (there_has_been_a_change);
    
    auto it = craters.begin();
    while (it != craters.cend()) {
        if (it->r < params.min_crater_radius) {
            it = craters.erase(it);
        } else {
            it++;
        }
    }

    return craters;
}