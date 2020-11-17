#ifndef FCM_HPP
#define FCM_HPP

#include "atmospheric_density.hpp"
#include "fragment.hpp"
#include "parameters.hpp"

#include <array>
#include <cmath>
#include <list>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

namespace fcm {

/**
 * @brief Info about a structural group within a meteoroid in the Wheeler et al. (2018) FCM model.
 * On first break up, the meteoroid splits up into a debris cloud and these structural pieces.
 */
struct StructuralGroup : public SubFragment {
    using pieces_t = unsigned short int;
    /**
     * @brief number of identical pieces in the group
     */
    pieces_t pieces;

    /**
     * @brief Construct a new StructuralGroup object
     * 
     * @param mass_fraction: total group mass as fraction of the total meteoroid mass
     * @param pieces: number of identical pieces in the group
     * @param strength: aerodynamic strength in [Pa]
     * @param density: mass density in [kg/m^3]
     * @param cloud_mass_frac: fraction of mass that produces a debris cloud on breakup
     * @param strength_scaler: scaling factor in Weibull equation determining the aerodynamic
     *                         strength of debris pieces after further breakup events
     * @param fragment_mass_fractions: number and mass fractions of debris pieces that are formed
     *                                 after further breakup events
     */
    StructuralGroup(const double mass_fraction, const pieces_t pieces, const double strength,
                    const double density, const double cloud_mass_frac, const double strength_scaler,
                    const std::vector<double>& fragment_mass_fractions) :
        SubFragment {mass_fraction, strength, density, cloud_mass_frac, strength_scaler,
                     fragment_mass_fractions}, pieces(pieces)
    {
        if (mass_fraction <= 0 || mass_fraction > 1) {
            throw std::invalid_argument("mass_fraction must be in the half-open interval (0, 1]");
        }
        if (pieces == 0) throw std::invalid_argument("pieces must be > 0");
        if (strength <= 0) throw std::invalid_argument("strength must be > 0");
        if (density <= 0) throw std::invalid_argument("density must be > 0");
        if (cloud_mass_frac < 0 || cloud_mass_frac > 1) {
            throw std::invalid_argument("cloud_mass_frac must be in the interval [0, 1]");
        }
        if (cloud_mass_frac < 1) {
            if (strength_scaler <= 0) throw std::invalid_argument("strength_scaler must be > 0");
            if (fragment_mass_fractions.empty()) {
                throw std::invalid_argument("fragment_mass_fractions must be provided if cloud_mass_frac < 1");
            }
            if (!std::is_sorted(fragment_mass_fractions.cbegin(), fragment_mass_fractions.cend())){
                throw std::invalid_argument("fragment_mass_fractions must be sorted in ascending order");
            }
            if (std::abs(std::accumulate(fragment_mass_fractions.cbegin(), fragment_mass_fractions.cend(), 1) - 1) > 1e-10) {
                throw std::invalid_argument("sum of fragment_mass_fractions must equal 1");
            }
        }
    }
};

struct Meteoroid {
    /**
     * @brief average density in [kg/m^3]
     */
    double density;

    /**
     * @brief velocity at the start of the simulation in [m/s]
     */
    double velocity;

    /**
     * @brief radius in [m]
     */
    double radius;

    /**
     * @brief trajectory angle w.r.t. ground. Straight down = pi/2, parallel to ground = 0
     */
    double angle;

    /**
     * @brief aerodynamic strength in [Pa]
     */
    double strength;

    /**
     * @brief fraction of mass that produces a debris cloud on breakup
     */
    double cloud_mass_frac;

    /**
     * @brief Inner structure as defined in the Wheeler et. al (2018) model
     */
    std::list<StructuralGroup> structural_groups;

    /**
     * @brief Construct a new Meteoroid object
     * 
     * @param density: average density in [kg/m^3]
     * @param velocity: velocity at the start of the simulation in [m/s]
     * @param radius: meteoroid radius in [m]
     * @param angle: trajectory angle w.r.t. ground. Straight down = pi/2, parallel to ground = 0
     * @param strength: aerodynamic strength of meteoroid in [Pa]
     * @param cloud_mass_frac: fraction of mass that produces a debris cloud on breakup
     * @param structural_groups: Inner structure as defined in the Wheeler et. al (2018) model
     */
    Meteoroid(double density, double velocity, double radius, double angle, double strength,
              double cloud_mass_frac,
              std::list<StructuralGroup>&& structural_groups=std::list<StructuralGroup>())
        : density(density), velocity(velocity), radius(radius), angle(angle), strength(strength),
          cloud_mass_frac(cloud_mass_frac), structural_groups(std::move(structural_groups))
    {
        if (density <= 0) throw std::invalid_argument("density must be > 0");
        if (velocity < 0) throw std::invalid_argument("velocity must be >= 0");
        if (radius <= 0) throw std::invalid_argument("radius must be > 0");
        if (angle <= 0 || angle > M_PI_2) {
            throw std::invalid_argument("angle must be in half-open interval (0, pi/2]");
        }
        if (strength <= 0) throw std::invalid_argument("strength must be > 0");
        if (cloud_mass_frac < 0 || cloud_mass_frac > 1) {
            throw std::invalid_argument("frag_velocity_coeff must be in the interval [0, 1]");
        }
        if (!structural_groups.empty() && std::abs(
            std::accumulate(structural_groups.cbegin(), structural_groups.cend(), 0,
                            [](const double s, const auto& g){ return g.mass_fraction + s; }) - 1
        ) > 1e-10) {
            throw std::invalid_argument("sum of mass_fractions must equal 1");
        }
    }
    
    /**
     * @brief Calculate mass of the meteoroid based on provided density and radius
     * 
     * @return constexpr double 
     */
    constexpr double mass() const noexcept { return sphere_m(this->density, this->radius); };
};

/**
 * @brief Calculate the differential equations for a fragment or a debris cloud with a given state
 *        within the atmosphere. See README.md for the equations.
 * 
 * @param fragment: meteoroid fragment
 * @param params: simulation parameters
 * @param settings: simulation settings
 * @return offset: vector of d(state)/dt values
 */
offset dfdt(const Fragment& fragment, const FCM_params& params, const FCM_settings& settings);

/**
 * @brief Simulates atmospheric entry including break up. Returns time series data for all
 *        meteoroid fragments, as well as the sum of dE/dz(z) of all fragments.
 * 
 * @param impactor: Meteoroid as defined in the Wheeler et al. (2018) publication
 * @param z_start: simulation start altitude above MOLA_0 in [m]
 * @param z_ground: ground altitiude above MOLA_0 in [m]
 * @param rho_a: atmospheric density
 * @param params: simulation parameters
 * @param settings: simulation settings
 * @param calculate_dEdz: whether to calculate dE/dz
 * @param seed: seed for the random number generator
 * @return std::pair with 
 *         first = dE/dz values, evenly spaced according to settings.dh
 *         second = list of data about the fragments; std::pair with
 *                  first = FragmentInfo object
 *                  second = list of time series data (if length 1 then final state)
 */
std::pair<
    std::vector<double>,
    std::list<std::pair<
        FragmentInfo,
        std::list<std::array<double, data_size>>
    >>
> solve_entry(const Meteoroid& impactor, const double z_start, const double z_ground,
              const AtmosphericDensity& rho_a, const FCM_params& params, 
              const FCM_settings& settings, const bool calculate_dEdz=false, const id_type seed=0);

/**
 * @brief Crater info with (x,y) coordinates and radius r.
 */
struct Crater {
    /**
     * @brief downrange coordinate in [m]
     */
    double x;

    /**
     * @brief orthogonal coordinate in [m]
     */
    double y;

    /**
     * @brief rim-to-rim crater radius in [m]
     */
    double r;

    /**
     * @brief List with IDs of meteoroid fragments that formed this crater (for debugging)
     */
    std::list<id_type> fragment_ids;
};

/**
 * @brief Based on the final velocity, impact angle and mass of all meteoroid fragments, 
 *        calculates the spatial coordinates and sizes of all impact craters.
 * 
 * @param fragments: list of time series data of meteoroid fragments;
 *                   second output of solve_entry() function
 * @param params: simulation parameters
 * @param settings: simulation settings
 * @return std::list<Crater> List of impact craters
 */
std::list<Crater> calculate_craters(
    const std::list<std::pair<FragmentInfo, std::list<std::array<double, data_size>>>>& fragments,
    const FCM_params& params, const FCM_settings& settings
);

} // namespace fcm

#endif // !FCM_HPP
