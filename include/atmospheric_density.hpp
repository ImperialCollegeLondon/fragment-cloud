#ifndef ATMOSPHERIC_DENSITY_HPP
#define ATMOSPHERIC_DENSITY_HPP

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
    AtmosphericDensity(const std::vector<double>& height, const std::vector<double>& density);
    AtmosphericDensity(std::vector<double>&& height, const std::vector<double>& density);

    /**
     * @brief calculate air density at height h, in [m/s]
     * 
     * Extrapolates exponentially based on scale height at the edge of the height vector provided
     * 
     * @param h : height above sea level, in [m]
     */
    double calc(const double h) const noexcept;

    /**
     * @brief calculate scale height H at height h
     * 
     * Value H for which, at height h, the atmoshperic density is best approximated
     * by rho(h) = rho_0 * exp(-h / H) for some value rho_0
     * 
     * @param h : height above sea level, in [m]
     * @return double : scale height, in [m]
     */
    double scale_height(const double h) const noexcept;

private:
    std::vector<double> height_, log_rho_;
};

}

#endif // !ATMOSPHERIC_DENSITY_HPP
