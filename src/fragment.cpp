#include "fragment.hpp"
#include "ode.hpp"
#include "atmospheric_density.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stack>
#include <stdexcept>
#include <utility>

using namespace fcm;

offset offset::operator*(const double dt) const {
	offset product;

    // TODO: vectorize with AVX?
	product.dv = this->dv * dt;
    product.dm = this->dm * dt;
    product.dtheta = this->dtheta * dt;
    product.dx = this->dx * dt;
    product.dy = this->dy * dt;
    product.dz = this->dz * dt;
    product.d2r = this->d2r * dt;
    product.dr = this->dr * dt;

	return product;
}

void offset::operator+=(const offset& other) {
    // TODO: vectorize with AVX?
	this->dv += other.dv;
    this->dm += other.dm;
    this->dtheta += other.dtheta;
    this->dx += other.dx;
    this->dy += other.dy;
    this->dz += other.dz;
    this->d2r += other.d2r;
    this->dr += other.dr;
}

// Fragment ctor
Fragment::Fragment(double mass, double velocity, double radius, double theta, double z,
                   double strength, double cloud_mass_frac, double init_density,
                   std::shared_ptr<const AtmosphericDensity> rho_a,
                   std::shared_ptr<const FCM_params> params,
                   std::shared_ptr<const FCM_settings> settings, id_type fragment_id,
                   std::list<SubFragment>&& inner_structure,
                   double t, double x, double y, double cos_phi, double sin_phi
) : state_{velocity, mass, theta, x, y, z, radius, 0}, t_(t), dt_(settings->precision), dt_next_(dt_),
    strength_(strength), cos_phi_(cos_phi), sin_phi_(sin_phi), cloud_mass_frac_(cloud_mass_frac), 
    init_density_(init_density), params_(std::move(params)), settings_(std::move(settings)),
    rho_a_(std::move(rho_a)), fragment_id_(fragment_id)
{
    assert(mass > 0);
    assert(velocity >= 0);
    assert(radius >= 0);
    assert(-M_PI <= theta && theta <= M_PI);
    assert(strength > 0);
    assert(0 <= cloud_mass_frac <= 1);
    assert(-1 <= cos_phi && cos_phi <= 1);
    assert(-1 <= sin_phi && sin_phi <= 1);
    assert(abs(std::hypot(cos_phi, sin_phi) - 1) < 1e-10);
    assert(init_density > 0);
    // assert(is_cloud || abs(1 - init_density / sphere_rho(mass, radius)) < 1e-10);

    inner_structure.sort([](const auto& a, const auto& b){
        return a.mass_fraction > b.mass_fraction;
    });
    if (!inner_structure.empty()) {
        assert(1e-10 > abs(1 - std::accumulate(
            inner_structure.cbegin(), inner_structure.cend(), 0.0,
            [](const double s, const auto& g){ return g.mass_fraction + s; }
        )));
    }
    this->inner_structure_ = std::stack<SubFragment, std::list<SubFragment>>(std::move(inner_structure));
    this->rng_.seed(fragment_id);
}

Fragment Fragment::split_subfragment(const SubFragment& subfragment,
                                     const double total_subfragments_mass) {
    const auto subfragment_mass = subfragment.mass_fraction * total_subfragments_mass;
    const auto subfragment_radius = sphere_r(subfragment_mass, subfragment.density);
    assert(subfragment_mass / this->mass() - 1 < 1e-10);

    const double remaining_mass_fraction = std::max(1 - subfragment_mass/this->mass(), 0.0);

    v_after new_velocities;
    if (remaining_mass_fraction > 1e-10) {
        this->state_.dr *= std::cbrt(remaining_mass_fraction);
        this->state_.d2r *= std::cbrt(remaining_mass_fraction);        
        this->dEdz_ *= remaining_mass_fraction;
        this->state_.dm -= subfragment_mass;

        const auto v_t = V_T(
            this->params_->frag_velocity_coeff, std::max(this->radius(), subfragment_radius),
            std::min(this->radius(), subfragment_radius), this->air_density(), subfragment.density,
            this->velocity()
        );
        new_velocities = apply_perpendicular_velocity(
            v_t, this->velocity(), this->cos_theta(), this->sin_theta(), this->cos_phi_,
            this->sin_phi_, std::min(subfragment_mass/this->mass(), this->mass()/subfragment_mass),
            this->rng_()
        );
        this->state_.dv = new_velocities.fragment_v;
        this->state_.dtheta = new_velocities.fragment_theta;
        this->cos_phi_ = new_velocities.fragment_cos_phi;
        this->sin_phi_ = new_velocities.fragment_sin_phi;
    } else {
        new_velocities = v_after {this->velocity(), this->state_.dtheta, this->cos_phi_, this->sin_phi_,
                                  0, this->state_.dtheta, this->cos_phi_, this->sin_phi_};
    }

    std::list<SubFragment> new_inner_structure;
    assert(std::is_sorted(subfragment.fragment_mass_fractions.cbegin(),
                          subfragment.fragment_mass_fractions.cend()));
    double factor = 1;
    double frac_sum = 0;
    for (size_t i = 0; i < subfragment.fragment_mass_fractions.size(); i++) {
        const auto frac = subfragment.fragment_mass_fractions[i];
        const auto frac_random = std::uniform_real_distribution((1 - this->params_->fragment_mass_disp) * frac,
                                                                frac)(this->rng_);
        const auto frac_final = i < subfragment.fragment_mass_fractions.size() - 1
                                ? frac_random * factor : 1 - frac_sum;

        // TODO: Base mean on initial bulk strength and weight? Or on initial subgroup strength and weight?
        //       Or like here on parent weight and strength? Could easily compound to super strong fragments like this...
        const auto mean_strength = fragment_mean_strength(frac_final, subfragment.strength,
                                                          subfragment.strength_scaler);
        const auto strength_random = mean_strength * std::pow(10, 
            std::normal_distribution(0.0, this->params_->strengh_scaling_disp)(this->rng_)
        );

        new_inner_structure.push_back(
            SubFragment {frac_final, std::min(strength_random, this->params_->max_strength),
                         subfragment.density, subfragment.cloud_mass_frac,
                         subfragment.strength_scaler, subfragment.fragment_mass_fractions}
        );
        frac_sum += frac_final;
        factor *= (1 - subfragment.fragment_mass_fractions[i]) / (1 - frac_random);
    }
    assert(abs(frac_sum - 1) < 1e-10);

    // Fragment ctor
    Fragment new_fragment(
        subfragment_mass, new_velocities.subfragment_v, subfragment_radius,
        new_velocities.subfragment_theta, this->z(), subfragment.strength,
        subfragment.cloud_mass_frac, subfragment.density, this->rho_a_, this->params_,
        this->settings_, this->rng_(), std::move(new_inner_structure), this->t_, this->state_.dx,
        this->state_.dy, new_velocities.subfragment_cos_phi, new_velocities.subfragment_sin_phi
    );
    new_fragment.dEdz_ = this->dEdz_ * subfragment_mass / this->mass();

    return new_fragment;
}

std::list<Fragment> Fragment::break_apart() {
    std::list<Fragment> subfragments;
    if (this->cloud_mass_frac_ < 1) {
        assert(!this->inner_structure_.empty());
        const double total_subfragments_mass = this->mass() * (1 - this->cloud_mass_frac_);
        while (!this->inner_structure_.empty()) {
            auto subfragment = this->split_subfragment(this->inner_structure_.top(),
                                                             total_subfragments_mass);
            this->inner_structure_.pop();

            if (subfragment.crater_detectable()) {
                if (subfragment.rp() > subfragment.strength()) {
                    subfragments.splice(subfragments.cend(), subfragment.break_apart());
                } else {
                    subfragments.push_back(std::move(subfragment));
                }
            }
        }
    }
    if (this->cloud_mass_frac_ > 0) {
        assert(this->inner_structure_.empty());
        assert(this->mass() > 1e-5);
        // Fragment ctor
        Fragment cloud(this->mass(), this->velocity(), this->radius(), this->state_.dtheta, this->z(),
                       this->strength(), this->cloud_mass_frac_, this->init_density_, this->rho_a_,
                       this->params_, this->settings_, this->rng_(), std::list<SubFragment>(),
                       this->t_, this->state_.dx, this->state_.dy, this->cos_phi_, this->sin_phi_);

        cloud.dEdz_ = this->dEdz_;
        cloud.is_cloud_ = true;
        if (cloud.crater_detectable()) {
            subfragments.push_back(std::move(cloud));
        }
    }
    for (auto& subfragment : subfragments) {
        this->daughter_fragments_.push_back(subfragment.fragment_id_);
        subfragment.parent_fragments_ = this->parent_fragments_;
        subfragment.parent_fragments_.push_back(this->fragment_id_);
    }

    return subfragments;
}

inline auto _relative_error(const Fragment& frag) {
    return abs(1.0 - frag.rp() / frag.strength());
}

void Fragment::backtrack_strength(double tolerance, unsigned short max_iterations) {
    unsigned short count = 0;
    auto relative_error = _relative_error(*this);
    while (relative_error > tolerance) {
        const auto divisor = 2.0 * this->delta_prev().dv / (this->velocity() * this->dt_prev())
                            + this->velocity() * this->sin_theta() / this->scale_height();
        const auto dt = (this->strength() / this->rp() - 1) / divisor;
        *this += this->delta_prev() * (dt / this->dt_prev());
        this->advance_time(dt);

        const auto new_relative_error = _relative_error(*this);
        if (new_relative_error > relative_error) {
            // std::cerr << "Warning: Back-tracking to get fragment.rp() close to "
            //             << "fragment.strength() is diverging. Stopped after " << count
            //             << " iterations. Relative error = " << std::setprecision(3)
            //             << 100 * abs(1 - this->rp() / this->strength()) << "%" << std::endl;
            break;
        }
        relative_error = new_relative_error;
        count++;
        if (count > max_iterations) {
            std::cerr << "Warning: Back-tracking to get fragment.rp() close to"
                        << "fragment.strength() did not converge. Stopped after 10 iterations."
                        << " Relative error = " << std::setprecision(3)
                        << 100 * abs(1 - this->rp() / this->strength()) << "%" << std::endl;
            break;
        }
    }
}
