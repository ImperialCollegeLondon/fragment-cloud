#ifndef ATMOSPHERIC_DENSITY_HPP
#define ATMOSPHERIC_DENSITY_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace fcm {

/**
 * @brief Interpolates exponentially between values given in two vectors
 */
class AtmosphericDensity {
public:
    /**
     * @brief Construct a new AtmosphericDensity object
     * 
     * @param height : height above sea level, in [m], in ascending order
     * @param density : air density at these heights, in [kg/m^3]
     */
    template<class T, class U>
    AtmosphericDensity(T&& height, const U& density) {
        assert(height.size() == density.size());
        assert(std::is_sorted(height.cbegin(), height.cend()));
        assert(std::is_sorted(density.crbegin(), density.crend()));
        assert(density.back() > 0);

        this->height_ = std::forward<T>(height);
        this->log_rho_ = std::vector<double>(density.size());

        std::transform(density.cbegin(), density.cend(), this->log_rho_.begin(),
                       [](const double rho){ return std::log(rho); });
    }

    /**
     * @brief calculate air density at height h, in [kg/m^3]
     * 
     * Extrapolates exponentially based on scale height at the edge of the height vector provided
     * 
     * @param h : height above sea level, in [m]
     */
    double calc(double h) const noexcept;

    /**
     * @brief calculate scale height H at height h
     * 
     * Value H for which, around height h, the atmoshperic density is best approximated
     * by rho(h) = rho_0 * exp(-h / H) for some value rho_0
     * 
     * @param h : height above sea level, in [m]
     * @return double : scale height, in [m]
     */
    double scale_height(double h) const noexcept;

private:
    std::vector<double> height_, log_rho_;
};

}

#endif // !ATMOSPHERIC_DENSITY_HPP
