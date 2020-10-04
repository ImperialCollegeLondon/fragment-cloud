#ifndef FRAGMENT_HPP
#define FRAGMENT_HPP

#include "ode.hpp"
#include "parameters.hpp"
#include "atmospheric_density.hpp"

#include <array>
#include <cmath>
#include <list>
#include <memory>
#include <random>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#define data_size 11

namespace fcm {

/**
 * @brief Return type of the ODE function
 */
struct offset {
    double dv, dm, dtheta, dx, dy, dz, dr, d2r;

    // TODO: Is by reference or by value faster for offset struct?
    void operator+=(const offset& other);
    offset operator*(const double dt) const;

    /**
     * @brief add operator
     * 
     * @tparam T offset
     */
    template <class T>
    offset operator+(T&& other) const {
        offset sum = std::forward<T>(other);
        sum += *this;
	    return sum;
    }
};

/**
 * @brief Info about a subfragment (to be generated at next breakup)
 */
struct SubFragment {
    /**
     * @brief subfragment mass as fraction of total mass of all new fragments at next breakup
     */
    double mass_fraction;

    /**
     * @brief aerodynamic strength, in [Pa]
     */
    double strength;

    /**
     * @brief density in [kg/m^3]
     */
    double density;

    /**
     * @brief Fraction of subfragment class that forms a debris cloud when
     * this subfragment breaks up
     */
    double cloud_mass_frac;

    /**
     * @brief Exponent for Weibull equation to determine strength of new subfragments when
     * this subfragment breaks up
     */
    double strength_scaler;

    /**
     * @brief When this subfragment breaks up, list of new subfragments to be generated, with
     * their mass fractions
     */
    std::vector<double> fragment_mass_fractions {};
};

using id_type = std::mt19937::result_type;

/**
 * @brief Relevant info about a fragment (for debugging)
 */
struct FragmentInfo {
    /**
     * @brief aerodynamic strength, in [Pa] 
     */
    double strength;

    /**
     * @brief density when fragment was generated, in [kg/m^3]
     */
    double init_density;
    
    /**
     * @brief fragment ID
     */
    id_type id;
    
    /**
     * @brief parent IDs (bulk -> fragment 1 -> fragment 1.1 -> ... -> parent of this one);
     */
    std::vector<id_type> parent_ids;
    
    /**
     * @brief if breakup happened, list of IDs of all new fragments generated in this breakup
     */
    std::vector<id_type> daughter_ids;
    
    /**
     * @brief whether this fragment is treated as a debris cloud (radius expands)
     */
    bool is_cloud; 
    
    /**
     * @brief whether final state of fragment is at ground level (-> impact)
     * or out of the atmosphere (-> escape)
     */
    bool impact, escape;
};

/**
 * @brief 
 */
class Fragment {
private:
    // variable
    offset state_;
    double t_, dt_, dt_next_;
    std::stack<SubFragment, std::list<SubFragment>> inner_structure_;

    double dEdz_ = 0;
    offset delta_prev_ {};
    offset df_prev_ {};
    std::vector<id_type> daughter_fragments_ {};
    std::mt19937 rng_ {};

    // const
    double strength_, cos_phi_, sin_phi_, cloud_mass_frac_, init_density_;
    std::shared_ptr<const FCM_params> params_;
    std::shared_ptr<const FCM_settings> settings_;
    std::shared_ptr<const AtmosphericDensity> rho_a_;
    id_type fragment_id_;

    std::vector<id_type> parent_fragments_ {};
    bool is_cloud_ = false;

    /**
     * @brief fragment density, assuming spherical cloud (if it is a cloud)
     */
    constexpr auto cloud_density() const noexcept {
        return this->is_cloud_ ? sphere_rho(this->state_.dm, this->state_.dr) : this->init_density_;
    }

    /**
     * @brief Kinetic + gravitational energy of fragment
     */
    auto E() const noexcept {
        if (!this->settings_->flat_planet) {
            return energy(this->state_.dm, this->state_.dv, this->state_.dz, this->params_->g0,
                          this->params_->Rp);
        }
        return energy_flat(this->state_.dm, this->state_.dv, this->state_.dz, this->params_->g0);
    }

    /**
     * @brief Break off subfragment from *this.
     * 
     * Calculates velocity and trajectory of new subfragment by applying a transverse velocity V_T
     * perpendicular to the current trajectory. V_T is calculated after Passey and Melosh (1980)
     * [https://www.sciencedirect.com/science/article/pii/001910358090072X]
     * Direction of VT in this 2D plane is chosen randomly.
     * Velocity, mass etc. of *this is adjusted according to momentum conservation.
     * 
     * @param subfragment: features of subfragment to break off
     * @param total_subfragments_mass: total mass of all subfragments 
     * @return Fragment: new subfragment, to be simulated independently 
     */
    Fragment split_subfragment(const SubFragment& subfragment, const double total_subfragments_mass);

public:
    /**
     * @brief Construct a new Fragment object
     * 
     * @param mass: fragment mass, in [kg]
     * @param velocity: fragment velocity, in [m/s] 
     * @param radius: fragment radius, in [m] (perpendicular to trajectory if not spherical) 
     * @param theta: trajectory angle w.r.t. horizon; 0 = horizontal, pi/2 = vertical downwards
     * @param z: start elevation above sea level, in [m] 
     * @param strength: fragment aerodynamic strength, in [Pa] 
     * @param cloud_mass_frac: on breakup, fraction of fragment mass that forms a debris cloud
     * @param init_density: fragment density, in [kg/m^3]
     * @param rho_a: atmospheric density interpolator class 
     * @param params 
     * @param settings
     * @param fragment_id
     * @param inner_structure: list of subfragments that will be generated on breakup
     * @param t: start time, if not 0 
     * @param x: start x-position, if not 0 
     * @param y: start y-position, if not 0
     * @param cos_phi: cos(phi), where phi is the angle in the xy-plane
     * @param sin_phi: sin(phi), where phi is the angle in the xy-plane 
     */
    Fragment(double mass, double velocity, double radius, double theta, double z, double strength,
             double cloud_mass_frac, double density, std::shared_ptr<const AtmosphericDensity> rho_a,
             std::shared_ptr<const FCM_params> params, std::shared_ptr<const FCM_settings> settings,
             id_type fragment_id=0, std::list<SubFragment>&& inner_structure=std::list<SubFragment>(),
             double t=0, double x=0, double y=0, double cos_phi=1, double sin_phi=0);
    /**
     * @brief fragment mass, in [kg] 
     */
    constexpr auto mass() const noexcept { return this->state_.dm; }
    
    /**
     * @brief fragment velocity, in [m/s]
     */
    constexpr auto velocity() const noexcept { return this->state_.dv; }
    
    /**
     * @brief cos(theta), where theta is the trajectory angle w.r.t. horizon;
     * 0 = horizontal, pi/2 = vertical downwards
     */
    inline auto cos_theta() const noexcept { return std::cos(this->state_.dtheta); }
    
    /**
     * @brief sin(theta), where theta is the trajectory angle w.r.t. horizon;
     * 0 = horizontal, pi/2 = vertical downwards
     */
    inline auto sin_theta() const noexcept { return std::sin(this->state_.dtheta); }
    
    /**
     * @brief elevation above sea level, in [m]
     */
    constexpr auto z() const noexcept { return this->state_.dz; }
    
    /**
     * @brief fragment radius perpendicular to trajectory, in [m]
     * 
     * When fragment ablates, the pancake and debris cloud model assume that it just deforms into an
     * ellipsoid, but area relevant for drag (perpendicular) to trajectory, stays constant
     */
    constexpr auto radius() const noexcept { return this->state_.dr; }
    
    /**
     * @brief change in radius, in [m/s]
     */
    constexpr auto dr() const noexcept { return this->state_.d2r; }
    
    /**
     * @brief Whether this fragment is a debris cloud
     */
    constexpr auto is_cloud() const noexcept { return this->is_cloud_; }
    
    /**
     * @brief aerodynamic strength, in [Pa]
     */
    constexpr auto strength() const noexcept { return this->strength_; }

    /**
     * @brief density of the meteoroid material, in [kg/m^3]
     */
    constexpr auto density() const noexcept { return this->init_density_; }
    
    /**
     * @brief cos(phi), where phi is the trajectory angle in the xy-plane
     */
    constexpr auto cos_phi() const noexcept { return this->cos_phi_; }
    
    /**
     * @brief cos(phi), where phi is the trajectory angle in the xy-plane
     */
    constexpr auto sin_phi() const noexcept { return this->sin_phi_; }
    
    /**
     * @brief delta t (time step size) used in previous iteration, in [s]
     */
    constexpr auto dt_prev() const noexcept { return this->dt_; }
    
    /**
     * @brief delta t (time step size) to use for next iteration, in [s]
     */
    constexpr auto dt_next() const noexcept { return this->dt_next_; }

    /**
     * @brief Energy change (kinetic + gravitational) per unit height, in [J/m]
     */
    constexpr auto dEdz() const noexcept { return this->dEdz_; }
    
    /**
     * @brief current state of the fragment
     */
    constexpr auto& state() const noexcept { return this->state_; }
    
    /**
     * @brief change to state in previous iteration (for adaptive time step)
     */
    inline const auto& delta_prev() const noexcept { return this->delta_prev_; }
    
    /**
     * @brief ODE evaluation of previous state (for multi-step schemes)
     */
    inline const auto& df_prev() const noexcept { return this->df_prev_; }

    /**
     * @brief scale height H of air density function at current elevation z, in [m].
     * 
     * So at this height, the air density is best approximated by rho(z) = rho(z1) * exp(-(z-z1) / H)
     */
    inline auto scale_height() const noexcept { return this->rho_a_->scale_height(this->z()); }
    
    /**
     * @brief air density density at current elevation z, in [kg/m^3]
     */
    inline auto air_density() const noexcept { return this->rho_a_->calc(this->z()); }
    
    /**
     * @brief ram pressure, in [Pa]
     */
    inline auto rp() const noexcept { return ram_pressure(this->air_density(), this->velocity()); }
    
    /**
     * @brief kinetic energy, in [J]
     */
    constexpr auto E_kin() const noexcept { return kinetic_energy(this->mass(), this->velocity()); }
    
    /**
     * @brief projected area perpendicual to trajectory, in [m^2] (relevant for drag)
     */
    constexpr auto area() const noexcept { return M_PI * std::square(this->radius()); }

    /**
     * @brief data point for time series to be returned to python
     */
    inline auto data() const noexcept {
        return std::array<double, data_size> {
            this->t_, this->mass(), this->velocity(), this->state_.dtheta, this->z(),
            this->state_.dx, this->state_.dy, this->radius(), this->cloud_density(), this->dEdz(),
            this->rp()
        };
    }

    /**
     * @brief relevant information about final state of this fragment
     * 
     * @param z_start: height of meteoroid above sea level at simulation start, in [m]
     * @param z_ground: height of ground above sea level, in [m] 
     */
    inline auto info(const double z_start, const double z_ground) const noexcept {
        return FragmentInfo {this->strength(), this->density(), this->fragment_id_,
                             this->parent_fragments_, this->daughter_fragments_, this->is_cloud(),
                             this->z() - z_ground < 1e-5, this->z() >= z_start};
    };

    /**
     * @brief Whether the fragment will produce an impact crater large enough to be detected.
     * 
     * Calculates an upper bound for the crater size and compares it to minimum detectable size parameter,
     * using ... from Daubar et al. (2020)
     * [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JE006382]
     */
    inline auto crater_detectable() const noexcept {
        // TODO: use v2_ground_upper_bound() ? Or use some kind of terminal velocity?
        return this->params_->min_crater_radius < radius_in_regolith(
            this->radius(), this->params_->g0, std::square(this->velocity()), this->cloud_density(),
            this->params_->ground_density, this->params_->rim_factor
        );
    }
    
    /**
     * @brief Change fragment state by offset delta. Calculates dEdz based on this.
     * 
     * @tparam T: offset struct
     * @param delta: changes to state
     */
    template<class T>
    void operator+=(T&& delta) {
        const auto old_energy = this->E();
        this->state_ += delta;
        const auto new_energy = this->E();
        this->dEdz_ = (new_energy - old_energy) / delta.dz;

        this->delta_prev_ = std::forward<T>(delta);
    }

    /**
     * @brief Create copy of this Fragment with state modified by offset delta
     * 
     * @tparam T: offset struct 
     * @param delta: changes to state
     * @return Fragment: copy of this fragment with change to state applied
     */
    template<class T>
    Fragment operator+(T&& delta) const {
        auto sum = *this;
        sum += std::forward<T>(delta);
        return sum;
    }

    /**
     * @brief define the time step size to be used in the next iteration
     */
    inline auto& set_dt_next(const double dt) noexcept { this->dt_next_ = dt; return *this; };

    /**
     * @brief increment time (usually after one iteration is complete)
     */
    inline auto& advance_time(const double dt) noexcept {
        this->t_ += dt; 
        this->dt_ = dt;
        return *this;
    };

    /**
     * @brief save ODE evaluation of state before current iteration (for multistep schemes)
     * 
     * @tparam T: offset struct
     * @param df: return values of ODEs
     */
    template<class T>
    inline auto& save_df_prev(T&& df) noexcept {
        this->df_prev_ = std::forward<T>(df);
        return *this;
    };

    /**
     * @brief Splits off all subfragments in the :this->inner_structure_: stack, and/or generates a
     * debris cloud (according to this->cloud_mass_frac_).
     * 
     * @return std::list<Fragment> list of all new subfragments (to be solved independently)
     */
    std::list<Fragment> break_apart();

    /**
     * @brief Take back part of the previous time step in order to have ram_pressure = strength
     * up to a relative tolerance.
     * 
     * prints warning to std::cerr if relative error increases after any iteration or it relative
     * error larger than :param: tolearnce after :param: max_iterations
     * 
     * @param tolerance : max relative tolerance
     * @param max_iterations : max number of iterations before abort
     * @TODO unit test
     */
    void backtrack_strength(double tolerance=1e-10, unsigned short max_iterations=10);
};

} // namespace fcm

#endif // !FRAGMENT_HPP
