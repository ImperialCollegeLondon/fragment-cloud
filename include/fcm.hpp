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

struct StructuralGroup : public SubFragment {
    using pieces_t = unsigned short int;
    pieces_t pieces;

    StructuralGroup(const double mass_fraction, const pieces_t pieces, const double strength,
                    const double density, const double cloud_mass_frac, const double strength_scaler,
                    const std::vector<double>& fragment_mass_fractions) :
        SubFragment {mass_fraction, strength, density, cloud_mass_frac, strength_scaler,
                     fragment_mass_fractions}, pieces(pieces)
    {
        if (mass_fraction < 0 || mass_fraction > 1) {
            throw std::invalid_argument("mass_fraction must be in the interval [0, 1]");
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
    double density, velocity, radius, angle, strength, cloud_mass_frac;
    std::list<StructuralGroup> structural_groups;

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
    
    constexpr double mass() const noexcept { return sphere_m(this->density, this->radius); };
};

offset dfdt(const Fragment& fragment, const FCM_params& params, const FCM_settings& settings);

std::pair<
    std::vector<double>,
    std::list<std::pair<
        FragmentInfo,
        std::list<std::array<double, data_size>>
    >>
> solve_entry(const Meteoroid& impactor, const double z_start, const double z_ground,
              const AtmosphericDensity& rho_a, const FCM_params& params, 
              const FCM_settings& settings, const bool calculate_dEdz=false, const id_type seed=0);

struct Crater {
    double x, y, r;
    std::list<id_type> fragment_ids;

    constexpr auto r3() const noexcept { return r*r*r; };
};

std::list<Crater> calculate_craters(
    const std::list<std::pair<FragmentInfo, std::list<std::array<double, data_size>>>>& fragments,
    const FCM_params& params, const FCM_settings& settings
);

} // namespace fcm

#endif // !FCM_HPP
