#include "atmospheric_density.hpp"

#include <algorithm>
#include <cassert>
#include <vector>
#include <cmath>

using namespace fcm;

AtmosphericDensity::AtmosphericDensity(const std::vector<double>& height,
                                       const std::vector<double>& density) {
    assert(height.size() == density.size());
    assert(std::is_sorted(height.cbegin(), height.cend()));
    assert(std::is_sorted(density.crbegin(), density.crend()));
    assert(density.back() > 0);

    this->height_ = height;
    this->log_rho_ = std::vector<double>(density.size());

    std::transform(density.cbegin(), density.cend(), this->log_rho_.begin(),
                   [](const double rho){ return std::log(rho); });
}

AtmosphericDensity::AtmosphericDensity(std::vector<double>&& height,
                                       const std::vector<double>& density) {
    assert(height.size() == density.size());
    assert(std::is_sorted(height.cbegin(), height.cend()));
    assert(std::is_sorted(density.crbegin(), density.crend()));
    assert(density.back() > 0);

    this->height_ = std::move(height);
    this->log_rho_ = std::vector<double>(density.size());

    std::transform(density.cbegin(), density.cend(), this->log_rho_.begin(),
                   [](const double rho){ return std::log(rho); });
}

inline auto _h_index(const std::vector<double>& sorted_vector, const double h) {
    return std::max(1L, std::distance(
        sorted_vector.cbegin(), std::lower_bound(sorted_vector.cbegin(), sorted_vector.cend(), h)
    ));
};

double AtmosphericDensity::calc(const double h) const noexcept {
    const auto h_index = std::min(long(this->height_.size() - 1), _h_index(this->height_, h));
    // const auto h_index = _h_index(this->height_, h);
    // if (h_index == this->height_.size()) {
    //     return 0;
    // }
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