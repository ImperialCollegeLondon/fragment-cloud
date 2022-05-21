#include "atmospheric_density.hpp"

using namespace fcm;

inline auto _h_index(const std::vector<double>& sorted_vector, const double h) {
    return std::max(1L, std::distance(
        sorted_vector.cbegin(), std::lower_bound(sorted_vector.cbegin(), sorted_vector.cend(), h)
    ));
};

double AtmosphericDensity::calc(const double h) const noexcept {
    const auto h_index = std::min(long(this->height_.size() - 1), _h_index(this->height_, h));
    if (std::abs(h - this->height_[h_index]) < 1e-1) {
        return std::exp(this->log_rho_[h_index]);
    }
    const double t = (h - this->height_[h_index - 1]) /
                     (this->height_[h_index] - this->height_[h_index - 1]);

    // TODO: once C++20 spec is finalized, use std::lerp()
    const double log_rho = this->log_rho_[h_index - 1]
                           + t * (this->log_rho_[h_index] - this->log_rho_[h_index - 1]);

    return std::exp(log_rho);
}

double AtmosphericDensity::scale_height(const double h) const noexcept {
    const auto h_index = std::min(long(this->height_.size() - 1), _h_index(this->height_, h));

    return (this->height_[h_index] - this->height_[h_index - 1])
            / (this->log_rho_[h_index - 1] - this->log_rho_[h_index]);
}