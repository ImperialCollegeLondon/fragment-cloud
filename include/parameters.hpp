#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <stdexcept>

namespace fcm {

/**
 * @brief Theoretical model used to describe discrete fragmentation
 */
enum class FragmentationModel {
    /**
     * @brief By Passey and Melosh, 1980 [https://www.sciencedirect.com/science/article/pii/001910358090072X]
     * 
     * Assumes all fragments fly in separate wakes after breakup.
     */
    independentWake,

    /**
     * @brief By [citation needed]
     * @not_implemented
     */
    collectiveWake,

    /**
     * @brief By [citation needed]
     * @not_implemented
     */
    nonCollectiveWake
};

/**
 * @brief Theoretical model used to describe debris clouds
 */
enum class CloudDispersionModel {
    /**
     * @brief By Chyba et al, 1993 [https://doi.org/10.1038/361040a0]
     */
    pancake,
    
    /**
     * @brief By Hills and Goda, 1993 [https://ui.adsabs.harvard.edu/abs/1993AJ....105.1114H]
     */
    debrisCloud,
    
    /**
     * @brief By Avramenko et al, 2014 [https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2013JD021028]
     */
    chainReaction
};

/**
 * @brief Theoretical model describing the ablation process
 */
enum class AblationModel {
    /**
     * @brief Model for stoney or iron meteoroids, or rubble piles.
     * 
     * Source: E.g. Passey and Melosh (1980)
     * [https://www.sciencedirect.com/science/article/pii/001910358090072X]
     */
    meteoroid,
    
    /**
     * @brief Model for comets.
     * Source: Crawford, 1996 [https://doi.org/10.1017/S0252921100115490]
     * 
     * @not_implemented
     */
    comet
};

/**
 * @brief Finite difference solver for ODEs
 */
enum class ODEsolver {
    /**
     * @brief Forward Euler scheme
     */
    forwardEuler,
    
    /**
     * @brief Improved Euler scheme (a.k.a. explicit RK2 scheme)
     */
    improvedEuler,
    
    /**
     * @brief 4th order explicit Runge-Kutta scheme
     */
    RK4,
    
    /**
     * @brief Second order Adams-Bashforth scheme (an explicit linear multistep scheme)
     */
    AB2
};

struct FCM_settings {
    FragmentationModel fragmentation_model;
    CloudDispersionModel cloud_dispersion_model;
    ODEsolver ode_solver;
    AblationModel ablation_model;

    /**
     * @brief Use the flat-planet approximation
     */
    bool flat_planet;

    /**
     * @brief Disable the ODE solver's variable step size
     */
    bool fixed_timestep;

    /**
     * @brief Relative error target for variable step size OR time step in [s] (if fixed_timestep)
     */
    double precision;
    
    /**
     * @brief Height step size, in [m], for the dEdz(z) curve
     */
    double dh;

    /**
     * @brief Keep and return all time series data
     */
    bool record_data;

    /**
     * @brief Threshold for fraction of initial kinetic energy of debris clouds
     * Stops simulation if the fraction drops below this threshold.
     */
    double cloud_stopping_criterion;

    /**
     * @brief Maximum number of iterations for ODE solver.
     * 
     * Simulation throws error if this is exceeded.
     */
    unsigned int max_iterations;

    FCM_settings(CloudDispersionModel cloud_model, ODEsolver solver, bool flat_planet,
                 bool fixed_timestep, double precision, double dh, bool record_data,
                 double cloud_stopping_criterion=1e-4, unsigned int max_iterations=1e6,
                 FragmentationModel frag_model=FragmentationModel::independentWake,
                 AblationModel ablation_model=AblationModel::meteoroid) :
        fragmentation_model(frag_model), cloud_dispersion_model(cloud_model), ode_solver(solver),
        ablation_model(ablation_model), flat_planet(flat_planet), fixed_timestep(fixed_timestep),
        precision(precision), dh(dh), record_data(record_data),
        cloud_stopping_criterion(cloud_stopping_criterion), max_iterations(max_iterations)
    {
        if (precision > 1 || 1e-7 > precision) {
            throw std::invalid_argument("precision must be in interval [1e-7, 1]");
        }
        if (max_iterations < 100) {
            throw std::invalid_argument("max_iterations must be >= 100");
        }
        if (cloud_stopping_criterion > 0.2 || 0 > cloud_stopping_criterion) {
            throw std::invalid_argument("cloud_stopping_criterion must be in interval [0, 0.2]");
        }
        if (dh <= 0) throw std::invalid_argument("dh must be > 0");
    }
};

struct FCM_crater_coeff {
    /**
     * @brief Minimum detectable crater radius, in [m]
     */
    double min_crater_radius;

    /**
     * @brief Density of the ground material, in [kg/m^3]
     */
    double ground_density;

    /**
     * @brief Strength of the ground material, in [Pa]
     */
    double ground_strength;

    /**
     * @brief K_1, K_2, K_r, mu, nu coefficients in Holsapple crater radius
     */
    double K1, K2, Kr, mu, nu;

    /**
     * @brief scaling factor for calculating rim-to-rim impact crater diameter
     */
    double rim_factor;

    FCM_crater_coeff(double min_crater_radius, double ground_density, double ground_strength,
                     double K1, double K2, double Kr, double mu, double nu, double rim_factor)
        : min_crater_radius(min_crater_radius), ground_density(ground_density),
          ground_strength(ground_strength), K1(K1), K2(K2), Kr(Kr), mu(mu), nu(nu),
          rim_factor(rim_factor)
    {
        if (min_crater_radius < 0) throw std::invalid_argument("min_crater_radius must be >= 0");
        if (ground_density <= 0) throw std::invalid_argument("ground_density must be > 0");
        if (ground_strength < 0) throw std::invalid_argument("ground_strength must be >= 0");
        if (K1 <= 0) throw std::invalid_argument("K1 must be > 0");
        if (K2 < 0) throw std::invalid_argument("K2 must be >= 0");
        if (Kr <= 0) throw std::invalid_argument("Kr must be > 0");
        if (mu <= 0) throw std::invalid_argument("mu must be > 0");
        if (nu < 0) throw std::invalid_argument("nu must be >= 0");
        if (rim_factor < 1) throw std::invalid_argument("rim_factor must be > 1");
    }
};

struct FCM_params {
    /**
     * @brief Gravitational acceleration at height 0, in [m/s^2]
     * examples: g0 = 9.81 (Earth); g0 = 3.711 (Mars); g0 = 8.87 (Venus)
     */
    double g0;

    /**
     * @brief Average planet radius, in [m]
     * examples: Rp = 6371e3 (Earth); Rp = 3389.5e3 (Mars); Rp = 6051.8e3 (Venus)
     */
    double Rp;

    /**
     * @brief Ablation coefficient, in [kg/J]
     * describes meteoroid mass loss in the atmoshere due to friction
     */
    double ablation_coeff;

    /**
     * @brief Dimensionless drag coefficient between meteoroid material and atmosphere
     */
    double drag_coeff;
    
    /**
     * @brief Dimensionless lift coefficient of meteoroid in the atmosphere
     */
    double lift_coeff;

    /**
     * @brief Maximum aerodynamic strength for a fragment, in [Pa]
     */
    double max_strength;

    /**
     * @brief Dimensionless bow shock interaction coefficient
     * Used to calculate velocity of fragments after fragmentation in independent wake model
     */
    double frag_velocity_coeff;

    /**
     * @brief Dimensionless cloud dispersion coefficient
     * Used to calculate dr/dt when aerodynamic strength is exceeded
     */
    double cloud_disp_coeff;

    /**
     * @brief Width of normal distribution from which fragment strengths are drawn after fragmentation
     */
    double strengh_scaling_disp;

    /**
     * @brief How much mass fractions of fragments are varied randomly.
     */
    double fragment_mass_disp;

    /**
     * @brief Minimum detectable crater radius, in [m]
     */
    double min_crater_radius;

    /**
     * @brief Density of the ground material, in [kg/m^3]
     */
    double ground_density;

    /**
     * @brief Strength of the ground material, in [Pa]
     */
    double ground_strength;

    /**
     * @brief K_1, K_2, K_r, mu, nu coefficients in Holsapple crater radius
     */
    double K1, K2, Kr, mu, nu;

    /**
     * @brief scaling factor for calculating rim-to-rim impact crater diameter
     */
    double rim_factor;


    FCM_params(double g0, double Rp, double ablation_coeff, double drag_coeff, double lift_coeff,
               double max_strength, double frag_velocity_coeff, double cloud_disp_coeff,
               double strengh_scaling_disp, double fragment_mass_disp,
               const FCM_crater_coeff& crater_coeff)
        : g0(g0), Rp(Rp), ablation_coeff(ablation_coeff), drag_coeff(drag_coeff),
        lift_coeff(lift_coeff), max_strength(max_strength), frag_velocity_coeff(frag_velocity_coeff),
        cloud_disp_coeff(cloud_disp_coeff), strengh_scaling_disp(strengh_scaling_disp),
        fragment_mass_disp(fragment_mass_disp), min_crater_radius(crater_coeff.min_crater_radius),
        ground_density(crater_coeff.ground_density), ground_strength(crater_coeff.ground_strength),
        K1(crater_coeff.K1), K2(crater_coeff.K2), Kr(crater_coeff.Kr), mu(crater_coeff.mu),
        nu(crater_coeff.nu), rim_factor(crater_coeff.rim_factor)
    {
        if (g0 <= 0) throw std::invalid_argument("g0 must be > 0");
        if (Rp <= 0) throw std::invalid_argument("Rp must be > 0");
        if (ablation_coeff < 0) throw std::invalid_argument("ablation_coeff must be >= 0");
        if (drag_coeff < 0) throw std::invalid_argument("drag_coeff must be >= 0");
        if (lift_coeff < 0) throw std::invalid_argument("lift_coeff must be >= 0");
        if (max_strength <= 0) throw std::invalid_argument("max_strength must be > 0");
        if (frag_velocity_coeff <= 0) throw std::invalid_argument("frag_velocity_coeff must be > 0");
        if (cloud_disp_coeff <= 0) throw std::invalid_argument("cloud_disp_coeff must be > 0");
        if (strengh_scaling_disp < 0) throw std::invalid_argument("strengh_scaling_disp must be >= 0");
        if (0 > fragment_mass_disp || fragment_mass_disp >= 1)
            throw std::invalid_argument("fragment_mass_disp must be in half-open interval [0, 1)");
    };
};

} // namespace fcm

#endif // !PARAMETERS_HPP
