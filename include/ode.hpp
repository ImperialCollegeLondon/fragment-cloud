#ifndef ODE_HPP
#define ODE_HPP 

#include <cmath>

constexpr double M_4_PI_3 = 4*M_PI/3;

namespace std {

template<class T>
constexpr T square(T a) noexcept {
    return a*a;
}

template<class T>
constexpr T cube(T a) noexcept {
    return a*a*a;
}

} // namespace std

namespace fcm {

/* **********************
    Conversion Functions
   ********************** */

/**
 * @brief velocity in x-direction, in [m/s]
 * 
 * not projected, i.e. not scaled, onto planet surface
 * 
 * @param v : meteoroid velocity in [m/s]
 * @param cos_theta : cos(theta)
 *                    where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param cos_phi : cos(phi)
 *                  where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory
 */
constexpr auto vx(double v, double cos_theta, double cos_phi) noexcept {
    return v * cos_theta * cos_phi;
}

/**
 * @brief velocity in y-direction, in [m/s]
 * 
 * not projected, i.e. not scaled, onto planet surface
 * 
 * @param v : meteoroid velocity in [m/s]
 * @param cos_theta : cos(theta)
 *                    where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param sin_phi : sin(phi)
 *                  where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory
 */
constexpr auto vy(double v, double cos_theta, double sin_phi) noexcept {
    return v * cos_theta * sin_phi;
}

/**
 * @brief velocity in z-direction, in [m/s]
 * 
 * negative = downwards, positive = upwards
 * 
 * @param v : meteoroid velocity in [m/s]
 * @param sin_theta : sin(theta)
 *                    where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 */
constexpr auto vz(double v, double sin_theta) noexcept {
    return -v * sin_theta;
}

/**
 * @brief calculate meteoroid velocity, in [m/s], from vertical and horizontal components
 * 
 * @param vh : velocity component in xy-plane, in [m/s]
 * @param vz : velocity component in z-direction, in [m/s]
 */
inline auto v(double vh, double vz) noexcept {
    return std::hypot(vh, vz);
}

/**
 * @brief calculate trajectory angle w.r.t. horizon, in [radians], from vertical and horizontal components
 * 
 * @param vh : velocity component in xy-plane, in [m/s]
 * @param vz : velocity component in z-direction, in [m/s]
 */
constexpr auto theta(double vh, double vz) noexcept {
    if (vh > 0.1) return std::atan2(-vz, vh);
    return M_PI;
}

/**
 * @brief cos(phi)
 * 
 * phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory
 * 
 * @param vx : velocity component in x-direction, in [m/s]
 * @param vh : velocity component in xy-plane, in [m/s]
 */
constexpr auto cos_phi(double vx, double vh) noexcept {
    return vx / vh;
}

/**
 * @brief sin(phi)
 * 
 * phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory
 * 
 * @param vy : velocity component in y-direction, in [m/s]
 * @param vh : velocity component in xy-plane, in [m/s]
 */
constexpr auto sin_phi(double vy, double vh) noexcept {
    return vy / vh;
}

/**
 * @brief gravitational acceleration at height z, in [m/s^2]
 * 
 * @param g0 : gravitational acceleration at z = 0, in [m/s^2]
 * @param z : z-coordinate of meteoroid, in [m]
 * @param Rp : radius of planetary body, in [m]
 */
constexpr auto g(double g0, double z, double Rp) noexcept {
    return g0 / std::square(1. + z/Rp);
}

/**
 * @brief gravitational acceleration in flat planet approximation
 * 
 * @param g0 : gravitational acceleration at z = 0, in [m/s^2]
 */
constexpr auto g_flat(double g0) noexcept {
    return g0;
}

/**
 * @brief ram pressure on meteoroid surface at the front
 * 
 * @param rho_a : air density, in [kg/m^3]
 * @param v : meteoroid velocity, in [m/s]
 */
constexpr auto ram_pressure(double rho_a, double v) noexcept {
    return rho_a * std::square(v);
}

/**
 * @brief compute radius of spherical object, in [m], given its mass and density
 * 
 * @param m : mass, in [kg]
 * @param rho : density, in [kg/m^3]
 */
inline auto sphere_r(double m, double rho) noexcept {
    return std::cbrt(m / (M_4_PI_3 * rho));
}

/**
 * @brief compute density of spherical object, in [kg/m^3], given its mass and radius
 * 
 * @param m : mass, in [kg]
 * @param r : radius, in [m]
 */
constexpr auto sphere_rho(double m, double r) noexcept {
    return m / (M_4_PI_3 * std::cube(r));
}

/**
 * @brief compute mass of spherical object, in [kg], given its density and radius
 * 
 * @param rho : density, in [kg/m^3]
 * @param r : radius, in [m]
 */
constexpr auto sphere_m(double rho, double r) noexcept {
    return M_4_PI_3 * rho * std::cube(r);
}

/* *********
    Breakup
   ********* */

/**
 * @brief transverse velocity V_T in separate fragments model
 * 
 * Source: Passey and Melosh (1980), p.224
 * [https://www.sciencedirect.com/science/article/pii/001910358090072X]
 * 
 * @param C : coefficient quantifying when fragment wakes separate (distance > C * radius_of_larger_fragment)
 * @param R_m : meteoroid radius, in [m], after fragmentation
 * @param R_f : fragment radius, in [m]
 * @param rho_a : air density, in [kg/m^3]
 * @param rho_f : fragment density, in [kg/m^3]
 * @param v : meteoroid velocity, in [m/s], before fragmentation
 */
inline auto V_T(double C, double R_m, double R_f, double rho_a, double rho_f, double v) noexcept {
    return v * std::sqrt(1.5*C * R_m/R_f * rho_a / rho_f);
}

/**
 * @brief Return type from breakup_velocities() function
 */
struct v_after {
    /**
     * @brief velocity, theta, cos(phi), sin(phi) of new subfragment
     */
    double subfragment_v, subfragment_theta, subfragment_cos_phi, subfragment_sin_phi;

    /**
     * @brief velocity, theta, cos(phi), sin(phi) of fragment after separation of new subfragment
     */
    double fragment_v, fragment_theta, fragment_cos_phi, fragment_sin_phi;
};

/**
 * @brief Calculates velocities and angles of fragment and its new subfragment
 *        s.t. total momentumis conserved and the velocity difference between the two
 *        is perpendicular to the current velocity and equals :param: v_t.
 *
 * @param v_t : transverse velocity, in [m/s]
 * @param v : fragment velocity, in [m/s], before fragmentation
 * @param cos_theta : cos(theta), where theta is the fragment's angle w.r.t horizon (0 = horizontal, pi/2 = vertical downwards)
 * @param sin_theta : sin(theta)
 * @param cos_phi : cos(phi), where phi is the trajectory angle in the xy-plane (0 = pre-entry angle)
 * @param sin_phi : sin(phi)
 * @param subfragment_mass_fraction : subfragment_mass / fragment_mass, both after fragmentation
 * @param rand : random integer
 */
inline auto apply_perpendicular_velocity(const double v_t, const double v, const double cos_theta,
                                         const double sin_theta, const double cos_phi,
                                         const double sin_phi, const double subfragment_mass_fraction,
                                         const long long rand) noexcept {
    const auto cos_alpha = std::cos(rand);
    const auto sin_alpha = std::sqrt(1 - std::square(cos_alpha)) * ((rand % 2) ? 1 : -1);

    auto vx_m = vx(v, cos_theta, cos_phi);
    auto vy_m = vy(v, cos_theta, sin_phi);
    auto vz_m = vz(v, sin_theta);

    const auto V_T_x = v_t * (sin_theta * cos_phi * cos_alpha + sin_phi * sin_alpha);
    const auto V_T_y = v_t * (sin_theta * sin_phi * cos_alpha - cos_phi * sin_alpha);
    const auto V_T_z = v_t * cos_theta * cos_alpha;

    const auto subfrag_fac = 1.0 / (subfragment_mass_fraction + 1);
    const auto frag_fac = 1.0 / (1.0 / subfragment_mass_fraction + 1);

    const auto vx_f = vx_m + V_T_x * subfrag_fac;
    const auto vy_f = vy_m + V_T_y * subfrag_fac;
    const auto vz_f = vz_m + V_T_z * subfrag_fac;
    const auto vh_f = std::hypot(vx_f, vy_f);
    
    vx_m -= V_T_x * frag_fac;
    vy_m -= V_T_y * frag_fac;
    vz_m -= V_T_z * frag_fac;
    const auto vh_m = std::hypot(vx_m, vy_m);

    return v_after {
        fcm::v(vh_f, vz_f), fcm::theta(vh_f, vz_f), fcm::cos_phi(vx_f, vh_f), fcm::sin_phi(vy_f, vh_f),
        fcm::v(vh_m, vz_m), fcm::theta(vh_m, vz_m), fcm::cos_phi(vx_m, vh_m), fcm::sin_phi(vy_m, vh_m),
    };
}

/**
 * @brief Weibull scaling law for strength of meteoroids with radius
 * 
 * Source: E.g. Svetsov et al. (1995), p.125
 * [https://www.sciencedirect.com/science/article/pii/S0019103585711165]
 * 
 * @param mass_frac : fragment mass / meteoroid mass before fragmentation
 * @param sigma_0 : meteoroid strength, in [Pa], before fragmentation
 * @param alpha : scaling exponent
 * @return fragment strength, in [Pa]
 */
inline auto fragment_mean_strength(double mass_frac, double sigma_0, double alpha) noexcept {
    return sigma_0 * std::pow(mass_frac, -alpha);
}

/* ********************************
    Gravitational + Kinetic Energy
   ******************************** */

/**
 * @brief kinetic energy of an object, in [J]
 * 
 * @param m : mass, in [kg]
 * @param v : velocity, in [m/s]
 */
constexpr auto kinetic_energy(double m, double v) noexcept {
    return 0.5 * m * std::square(v);
}

/**
 * @brief Kinetic + gravitational energy, in [J]
 *        0 at z=infinity and v=0
 * 
 * @param m : mass, in [kg]
 * @param v : velocity, in [m/s]
 * @param z : height above sea level, in [m]
 * @param g0 : gravitational acceleration at sea level, in [m/s^2]
 * @param Rp : radius of the planet, in [m]
 */
constexpr auto energy(double m, double v, double z, double g0, double Rp) noexcept {
    return kinetic_energy(m, v) - m * g0 * std::square(Rp) / (Rp + z);
}

/**
 * @brief Kinetic + gravitational energy in flat planet approximation, in [J]
 *        0 at z=0 and v=0
 * 
 * @param m : mass, in [kg]
 * @param v : velocity, in [m/s]
 * @param z : height above sea level, in [m]
 * @param g0 : gravitational acceleration at sea level, in [m/s^2]
 */
constexpr auto energy_flat(double m, double v, double z, double g0) noexcept {
    return kinetic_energy(m, v) + m * g0 * z;
}

/**
 * @brief Upper bound for squared meteoroid velocity at height h_ground, in [(m/s)^2],
 *        given its current velocity and height
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param g0 : gravitatinal acceleration at sea level, in [m/s^2]
 * @param z : height above sea level, in [m]
 * @param h_ground : ground height above sea level, in [m]
 */
constexpr auto v2_ground_upper_bound(double v, double g0, double z, double h_ground) noexcept {
    return std::square(v) + 2*g0 * (z - h_ground);
};


/* **********************
    Impact Crater Radius
   ********************** */

/**
 * @brief impact crater radius, in [m], for cohesionless, sand-like regolith
 * 
 * Source: Daubar et al. (2020), p.?, eq.S1
 * [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JE006382]
 * 
 * @param r : impactor radius, in [m]
 * @param g0 : gravitational acceleration, in [m/s^2], at impact height
 * @param vz2 : squared vertical component of impactor velocity, in [(m/s)^2]
 * @param rho_m : impactor density, in [kg/m^3]
 * @param rho_g : regolith density, in [kg/m^3]
 * @param rf : rim-to-rim factor
 */
inline auto radius_in_regolith(double r, double g0, double vz2, double rho_m,
                               double rho_g, double rf) noexcept {
    return 1.03 * rf * std::pow(r, 0.83) * std::pow(vz2 / g0, 0.17) * std::pow(rho_m / rho_g, 0.33);
}

/**
 * @brief impact crater radius, in [m], for weakly cohesive, porous, granular soil
 * 
 * Source: Daubar et al. (2020), p.?, eq.S2
 * [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JE006382]
 * 
 * @param r : impactor radius, in [m]
 * @param Y : cohesive strength of soil, in [Pa]
 * @param vz2 : squared vertical component of impactor velocity, in [(m/s)^2]
 * @param rho_m : impactor density, in [kg/m^3]
 * @param rho_g : soil density, in [kg/m^3]
 * @param rf : rim-to-rim factor
 */
inline auto radius_in_granular_soil(double r, double Y, double vz2, double rho_m,
                                    double rho_g, double rf) noexcept {
    return 1.03 * rf * r * std::pow(rho_g * vz2 / Y, 0.205) * std::pow(rho_m / rho_g, 0.4);
}

/**
 * @brief impact crater radius, in [m]
 *        [citation needed: Holsapple 1993? -> Ask Gareth]
 * 
 * @param Y : cohesive strength of soil, in [Pa]
 * @param rho_terrain : soil density, in [kg/m^3]
 * @param U : vertical component of impactor velocity, in [m/s]
 * @param g : gravitational acceleration, in [m/s^2], at impact height
 * @param R : impactor radius, in [m]
 * @param rho_impactor : impactor density, in [kg/m^3]
 * @param K_1, nu, mu, K_2, K_r : coefficients in Holsapple equations
 * @param m : impactor mass, in [kg]
 * @param rf : rim-to-rim factor
 */
inline auto Holsapple_crater_radius(double Y, double rho_terrain, double U, double g, double R,
                                    double rho_impactor, double K_1, double nu, double mu,
                                    double K_2, double K_r, double m, double rf) {
    const auto pi_2 = g*R / (U*U);
    const auto pi_3 = Y / (rho_terrain * U*U);
    const auto pi_4 = rho_terrain / rho_impactor;
    const auto exp1 = (6*nu - 2 - mu) / (3*mu);
    const auto exp2 = (6*nu - 2) / (3*mu);
    const auto exp3 = (2 + mu) / 2;
    const auto exp4 = -3*mu / (2 + mu);
    const auto pi_V = K_1 * std::pow(pi_2 * std::pow(pi_4, exp1)
                                     + K_2 * std::pow(pi_3 * std::pow(pi_4, exp2), exp3),
                                     exp4);
    const auto V = pi_V * m / rho_terrain;

    return rf * K_r * std::cbrt(V);
};


/* ***************
    ODE Functions
   *************** */

/**
 * @brief acceleration (drag + gravity), in [m/s^2]
 * 
 * @param C_D : drag coefficient
 * @param rp : ram pressure, in [Pa]
 * @param A : meteoroid area perpendicular to tranjectory, in [m^2]
 * @param m : meteoroid mass, in [kg]
 * @param g : gravitational acceleration, in [m/s^2]
 * @param sin_theta : sin(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 */
constexpr auto dvdt(double C_D, double rp, double A, double m, double g, double sin_theta) noexcept {
    return -C_D * rp * A / (2.*m) + g * sin_theta;
}

/**
 * @brief mass loss due to ablation, in [kg/s]
 * 
 * @param C_ab : ablaction coefficient
 * @param rp : ram pressure, in [Pa]
 * @param A : meteoroid area perpendicular to trajectory, in [m^2]
 * @param v : meteoroid velocity, in [m/s]
 */
constexpr auto dmdt(double C_ab, double rp, double A, double v) noexcept {
    return -C_ab * A * v * rp / 2.;
}

/**
 * @brief change in vertical trajectory angle, in [1/s]
 *        (lift + gravity + planet curvature)
 * 
 * @param g : gravitational acceleration, in [m/s^2]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param v : meteoroid velocity, in [m/s]
 * @param C_L : lift coefficent
 * @param rp : ram pressure, in [Pa]
 * @param A : area of meteoroid perpendicular to tranjectory, in [m^2]
 * @param m : meteoroid mass, in [kg]
 * @param Rp : planet radius, in [m]
 * @param z : height above sea level, in [m]
 */
constexpr auto dthetadt(double g, double cos_theta, double v, double C_L, double rp, double A,
                        double m, double Rp, double z) noexcept {
    return cos_theta * (g/v - v / (Rp + z)) - C_L * rp * A / (2.0 * m * v);
}

/**
 * @brief change in vertical trajectory angle, in [1/s]
 *        (lift + gravity + planet curvature), flat planet approximation
 * 
 * @param g : gravitational acceleration, in [m/s^2]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param v : meteoroid velocity, in [m/s]
 * @param C_L : lift coefficent
 * @param rp : ram pressure, in [Pa]
 * @param A : area of meteoroid perpendicular to tranjectory, in [m^2]
 * @param m : meteoroid mass, in [kg]
 */
constexpr auto dthetadt_flat(double g, double cos_theta, double v, double C_L, double rp, double A,
                             double m) noexcept {
    return cos_theta * g/v - C_L * rp * A / (2.0 * m * v);
}

/**
 * @brief change in z-coordinate, in [m/s]
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param sin_theta : sin(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 */
constexpr auto dzdt(double v, double sin_theta) noexcept {
    return vz(v, sin_theta);
}

/**
 * @brief change in x-coordinate, in [m/s]
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param cos_phi : cos(phi), where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory.
 * @param Rp : planet radius, in [m]
 * @param z : height above sea level, in [m]
 */
constexpr auto dxdt(double v, double cos_theta, double cos_phi, double Rp, double z) noexcept {
    return vx(v, cos_theta, cos_phi) / (1. + z/Rp);
}

/**
 * @brief change in x-coordinate, in [m/s], flat planet approximation.
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param cos_phi : cos(phi), where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory.
 */
constexpr auto dxdt_flat(double v, double cos_theta, double cos_phi) noexcept {
    return vx(v, cos_theta, cos_phi);
}

/**
 * @brief change in y-coordinate, in [m/s], flat planet approximation.
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param sin_phi : sin(phi), where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory.
 */
constexpr auto dydt(double v, double cos_theta, double sin_phi, double Rp, double z) noexcept {
    return vy(v, cos_theta, sin_phi) / (1. + z/Rp);
}

/**
 * @brief change in y-coordinate, in [m/s], flat planet approximation.
 * 
 * @param v : meteoroid velocity, in [m/s]
 * @param cos_theta : cos(theta), where theta is the trajectory angle w.r.t. horizon.
 *                    pi/2 = vertical, downwards, 0 = horizontal
 * @param sin_phi : sin(phi), where phi is the trajectory angle in the xy-plane w.r.t. initial tranjectory.
 */
constexpr auto dydt_flat(double v, double cos_theta, double sin_phi) noexcept {
    return vy(v, cos_theta, sin_phi);
}

/**
 * @brief Second derivative of radius in Pancake model, in [m/s^2]
 * 
 * Source: Chyba et al, 1993 [https://doi.org/10.1038/361040a0]
 * 
 * @param rp : ram pressure, in [Pa]
 * @param strength : meteoroid aerodynamic strength, in [Pa]
 * @param C_D : coefficient
 * @param rho_m : initial meteoroid density, in [kg/m^3]
 * @param r : debris cloud radius, in [m]
 */
constexpr auto d2rdt2_pancake(double rp, double strength, double C_D, double rho_m,
                              double r) noexcept {
    if (rp < strength) return 0.0;
    return C_D * rp / (2. * r * rho_m);
}

/**
 * @brief Derivative of bolide radius in Debris Cloud model, in [m/s]
 * 
 * Source: Hills and Goda, 1993 [https://ui.adsabs.harvard.edu/abs/1993AJ....105.1114H]
 * 
 * @param rp : ram pressure, in [Pa]
 * @param strength : meteoroid aerodynamic strength, in [Pa]
 * @param alpha : coefficient
 * @param rho_a : air density, in [kg/m^3]
 * @param rho_m : initial meteoroid density, in [kg/m^3]
 * @param v : debris cloud velocity, in [m/s]
 */
constexpr auto drdt_debriscloud(double rp, double strength, double alpha, double rho_a,
                                double rho_m, double v) noexcept {
    if (rp < strength) return 0.0;
    return v * std::sqrt(3.5 * alpha * rho_a/rho_m);
}

/**
 * @brief Derivative of radius of solid fragment undergoing ablation in [m/s]
 * 
 * Source: Avramenko et al, 2014 [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2013JD021028]
 * Fragments are approximated as having a perfectly spherical shape. Applying this rate of change to
 * the fragment's radius will make sure that it keeps its spherical shape when losing mass due to ablation.
 * 
 * @param r : meteoroid radius, in [m]
 * @param m : meteoroid mass, in [kg]
 * @param dmdt : ablation mass loss, in [kg/s]
 */
constexpr auto drdt_ablation(double r, double m, double dmdt) noexcept {
    return r/m * dmdt/3;
}

/**
 * @brief Derivative of radius in Chain Reaction model, in [m/s]
 * 
 * Source: Avramenko et al, 2014 [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2013JD021028]
 * 
 * @param rp : ram pressure, in [Pa]
 * @param strength : meteoroid aerodynamic strength, in [Pa]
 * @param r : meteoroid radius, in [m]
 * @param m : meteoroid mass, in [kg]
 * @param dmdt : ablation mass loss, in [kg/s]
 * @param C_fr : coefficient
 * @param rho_m : initial meteoroid density, in [kg/m^3]
 */
constexpr auto drdt_chainreaction(double rp, double strength, double r, double m, double dmdt,
                                  double C_fr, double rho_m) noexcept {
    auto drdt = drdt_ablation(r, m, dmdt);
    if (rp > strength) {
        drdt += C_fr * std::sqrt(rp - strength) * r /
                  (2. * std::cbrt(m) * std::cbrt(std::sqrt(rho_m)));
    }
    return drdt;
}
  
} // namespace fcm

#endif // !ODE_HPP
