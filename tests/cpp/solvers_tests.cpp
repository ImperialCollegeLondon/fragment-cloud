#define BOOST_TEST_MODULE ODE solvers
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

#include "solvers.hpp"

#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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

using namespace std;

template<class StateT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t(const function<NoRefStateT(StateT&&, const TimeT, const TimeT)>& step,
    const NoRefStateT& y0, const TimeT t0, const TimeT t_end, const TimeT dt) {
    auto t = t0;
    auto y = y0;
    do {
        y = step(forward<StateT>(y), t, dt);
        t += dt;
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        y = step(forward<StateT>(y), t, t_end - t);
    }

    return y;
}

template<class StateT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t(const function<NoRefStateT(StateT&&, const TimeT)>& step,
                const NoRefStateT& y0, const TimeT t0, const TimeT t_end, const TimeT dt) {
    auto t = t0;
    auto y = y0;
    do {
        y = step(forward<StateT>(y), dt);
        t += dt;
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        y = step(forward<StateT>(y), t_end - t);
    }

    return y;
}

template<class StateT, class DeltaT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t(const function<tuple<NoRefStateT, DeltaT>(StateT&&, const DeltaT&, const TimeT, const TimeT, const TimeT)>& step,
                DeltaT df_prev, const NoRefStateT& y1, const TimeT t1, const TimeT t_end, const TimeT dt) {
    auto t = t1;
    auto y = y1;
    do {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t, dt, dt);
        y = move(y_next);
        df_prev = move(df_next);
        t += dt;
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t, t_end - t, dt);
        y = move(y_next);
    }
    
    return y;
}

template<class StateT, class DeltaT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t(const function<tuple<NoRefStateT, DeltaT>(StateT&&, const DeltaT&, const TimeT, const TimeT)>& step,
           DeltaT df_prev, const NoRefStateT& y1, const TimeT t1, const TimeT t_end, const TimeT dt) {
    auto t = t1;
    auto y = y1;
    do {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, dt, dt);
        y = move(y_next);
        df_prev = move(df_next);
        t += dt;
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t_end - t, dt);
        y = move(y_next);
    }
    
    return y;
}

template<class StateT, class DeltaT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t_variable(
    const function<tuple<NoRefStateT, DeltaT>(StateT&&, const DeltaT&, const TimeT, const TimeT, const TimeT)>& step,
    NoRefStateT df_prev, const NoRefStateT& y1, const TimeT t1, const TimeT t_end, const TimeT dt
) {
    auto t = t1;
    auto y = y1;
    unsigned int counter = 0;
    do {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t, dt*(1 + counter%2), dt*(2-counter%2));
        y = move(y_next);
        df_prev = move(df_next);
        t += dt * (1 + counter++ % 2);
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t, t_end - t, dt);
        y = move(y_next);
    }
    
    return y;
}

template<class StateT, class DeltaT, class TimeT, class NoRefStateT=remove_reference_t<StateT>>
NoRefStateT y_t_variable(
    const function<tuple<NoRefStateT, DeltaT>(StateT&&, const DeltaT&, const TimeT, const TimeT)>& step,
    NoRefStateT df_prev, const NoRefStateT& y1, const TimeT t1, const TimeT t_end, const TimeT dt
) {
    auto t = t1;
    auto y = y1;
    unsigned int counter = 0;
    do {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, dt*(1 + counter%2), dt*(2-counter%2));
        y = move(y_next);
        df_prev = move(df_next);
        t += dt * (1 + counter++ % 2);
    } while (t < t_end);

    if (std::abs(1 - t/t_end) > 1e-10) {
        auto [y_next, df_next] = step(forward<StateT>(y), df_prev, t_end - t, dt);
        y = move(y_next);
    }
    
    return y;
}

function<double(const double, const double&)> test_ode() {
    const auto ode_func = [](const double t, const double& y){
        return y + cube(t);
    };

    return ode_func;
}

function<double(const double)> sol_function() {
    const auto sol_func = [](const double t){
        return 7*exp(t) - cube(t) - 3*square(t) - 6*t - 6;
    };

    return sol_func;
}


auto test_values() {
    const double t0 = 0;
    const double t_max = 3;
    const double y0 = 1;

    const auto ode_sol = sol_function();
    const auto y_solution = ode_sol(t_max);

    return make_tuple(t0, t_max, y0, y_solution);
}

BOOST_AUTO_TEST_SUITE(time_dependent_1D_ode)

BOOST_AUTO_TEST_CASE(forward_euler)
{
    const auto ode = test_ode();
    const auto [t0, t_max, y0, y_solution] = test_values();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001};

    vector<double> errors;

    double t, y, y_move;
    for (const auto dt : dts) {
        const function<double(const double&, const double, const double)> solver =
            [&](const double& y, const double t, const double dt) {
                return fcm::forward_euler(y, ode, t, dt);
            };
        const function<double(double&&, const double, const double)> solver_move =
            [&](double&& y, const double t, const double dt) {
                return fcm::forward_euler(move(y), ode, t, dt);
            };
        y = y_t<const double&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y == y_move, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be first order accurate
        BOOST_TEST(err_log_diff == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(improved_euler)
{
    const auto ode = test_ode();
    const auto [t0, t_max, y0, y_solution] = test_values();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005};

    vector<double> errors;

    double t, y, y_move;
    for (const auto dt : dts) {
        const function<double(const double&, const double, const double)> solver =
            [&](const double& y, const double t, const double dt) {
                return fcm::improved_euler(y, ode, t, dt);
            };
        const function<double(double&&, const double, const double)> solver_move =
            [&](double&& y, const double t, const double dt) {
                return fcm::improved_euler(move(y), ode, t, dt);
            };
        y = y_t<const double&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y == y_move, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be second order accurate
        BOOST_TEST(err_log_diff/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(RK4)
{
    const auto ode = test_ode();
    const auto [t0, t_max, y0, y_solution] = test_values();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005};

    vector<double> errors;

    double t, y, y_move;
    for (const auto dt : dts) {
        const function<double(const double&, const double, const double)> solver =
            [&](const double& y, const double t, const double dt) {
                return fcm::RK4(y, ode, t, dt);
            };
        const function<double(double&&, const double, const double)> solver_move =
            [&](double&& y, const double t, const double dt) {
                return fcm::RK4(move(y), ode, t, dt);
            };
        y = y_t<const double&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y == y_move, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be forth order accurate
        BOOST_TEST(err_log_diff/4 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(AB2)
{
    const auto ode = test_ode();
    const auto sol = sol_function();
    const auto [t0, t_max, y0, y_solution] = test_values();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005};

    vector<double> errors, errors_variable;

    double t, y, y_move, y_variable, y_move_variable;
    for (const auto dt : dts) {
        const function<tuple<double, double>(const double&, const double&, const double, const double, const double)> solver =
            [&](const double& y, const double& df_prev, const double t, const double dt, const double dt_prev) {
                return fcm::AB2(y, df_prev, ode, t, dt, dt_prev);
            };
        const function<tuple<double, double>(double&&, const double&, const double, const double, const double)> solver_move =
            [&](double&& y, const double& df_prev, const double t, const double dt, const double dt_prev) {
                return fcm::AB2(move(y), df_prev, ode, t, dt, dt_prev);
            };

        const auto df_prev = ode(t0, y0);
        const auto y1 = sol(t0 + dt);
        y = y_t<const double&>(solver, df_prev, y1, t0 + dt, t_max, dt);
        y_move = y_t(solver_move, df_prev, y1, t0 + dt, t_max, dt);

        BOOST_TEST(y == y_move, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));

        y_variable = y_t_variable<const double&>(solver, df_prev, y1, t0 + dt, t_max, dt);
        y_move_variable = y_t_variable(solver_move, df_prev, y1, t0 + dt, t_max, dt);

        BOOST_TEST(y_variable == y_move_variable, tt::tolerance(1e-10));
        errors_variable.push_back(std::abs(y_variable - y_solution));
    }

    double err_log_diff, err_log_diff_variable;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());
        err_log_diff_variable = log(errors_variable[i]) - log(errors_variable.back());

        // should be second order accurate
        BOOST_TEST(err_log_diff/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
        BOOST_TEST(err_log_diff_variable/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_SUITE_END()

struct State {
    double x, y;

    constexpr void operator+=(const State& other) noexcept {
        this->x += other.x;
        this->y += other.y;
    }

    template<class T>
    constexpr remove_reference_t<T> operator+(T&& other) const noexcept {
        auto sum = forward<T>(other);
        sum += *this;
        return sum;
    }

    constexpr State operator*(const double dt) const noexcept {
        return State {this->x * dt, this->y * dt};
    }
    inline double operator-(const State& other) const noexcept {
        return hypot(this->x - other.x, this->y - other.y);
    }
};

function<State(const State&)> test_ode_2D() {
    const auto ode_func = [](const State& pos){ return State {-pos.y, pos.x}; };

    return ode_func;
}

function<State(const double)> sol_function_2D() {
    const auto sol_func = [](const double t){ return State {cos(t), sin(t)}; };
    return sol_func;
}


auto test_values_2D() {
    const auto ode_sol = sol_function_2D();

    const double t0 = 0;
    const auto y0 = ode_sol(t0);
    const double t_max = 3;
    const auto y_solution = ode_sol(t_max);

    return make_tuple(t0, t_max, y0, y_solution);
}

BOOST_AUTO_TEST_SUITE(time_independent_2D_ode)

BOOST_AUTO_TEST_CASE(forward_euler)
{
    const auto ode = test_ode_2D();
    const auto [t0, t_max, y0, y_solution] = test_values_2D();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001};

    vector<double> errors;
    double t;
    State y, y_move;
    for (const auto dt : dts) {
        const function<State(const State&, const double)> solver =
            [&](const State& y, const double dt) { return fcm::forward_euler(y, ode, dt); };
        const function<State(State&&, const double)> solver_move =
            [&](State&& y, const double dt) { return fcm::forward_euler(move(y), ode, dt); };
        
        y = y_t<const State&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y.x == y_move.x, tt::tolerance(1e-10));
        BOOST_TEST(y.y == y_move.y, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be first order accurate
        BOOST_TEST(err_log_diff == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(improved_euler)
{
    const auto ode = test_ode_2D();
    const auto [t0, t_max, y0, y_solution] = test_values_2D();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005};

    vector<double> errors;
    double t;
    State y, y_move;
    for (const auto dt : dts) {
        const function<State(const State&, const double)> solver =
            [&](const State& y, const double dt) { return fcm::improved_euler(y, ode, dt); };
        const function<State(State&&, const double)> solver_move =
            [&](State&& y, const double dt) { return fcm::improved_euler(move(y), ode, dt); };
        
        y = y_t<const State&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y.x == y_move.x, tt::tolerance(1e-10));
        BOOST_TEST(y.y == y_move.y, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be second order accurate
        BOOST_TEST(err_log_diff/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(RK4)
{
    const auto ode = test_ode_2D();
    const auto [t0, t_max, y0, y_solution] = test_values_2D();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005};

    vector<double> errors;
    double t;
    State y, y_move;
    for (const auto dt : dts) {
        const function<State(const State&, const double)> solver =
            [&](const State& y, const double dt) { return fcm::RK4(y, ode, dt); };
        const function<State(State&&, const double)> solver_move =
            [&](State&& y, const double dt) { return fcm::RK4(move(y), ode, dt); };
        
        y = y_t<const State&>(solver, y0, t0, t_max, dt);
        y_move = y_t(solver_move, y0, t0, t_max, dt);

        BOOST_TEST(y.x == y_move.x, tt::tolerance(1e-10));
        BOOST_TEST(y.y == y_move.y, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));
    }

    double err_log_diff;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());

        // should be forth order accurate
        BOOST_TEST(err_log_diff/4 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_CASE(AB2)
{
    const auto ode = test_ode_2D();
    const auto sol = sol_function_2D();
    const auto [t0, t_max, y0, y_solution] = test_values_2D();
    const vector<double> dts {0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005};

    vector<double> errors, errors_variable;
    double t;
    State y, y_move, y_variable, y_move_variable;
    for (const auto dt : dts) {
        const function<tuple<State, State>(const State&, const State&, const double, const double)> solver =
            [&](const State& y, const State& df_prev, const double dt, const double dt_prev) {
                return fcm::AB2(y, df_prev, ode, dt, dt_prev);
            };
        const function<tuple<State, State>(State&&, const State&, const double, const double)> solver_move =
            [&](State&& y, const State& df_prev, const double dt, const double dt_prev) {
                return fcm::AB2(move(y), df_prev, ode, dt, dt_prev);
            };

        const auto df_prev = ode(y0);
        const auto y1 = sol(t0 + dt);
        y = y_t<const State&>(solver, df_prev, y1, t0 + dt, t_max, dt);
        y_move = y_t(solver_move, df_prev, y1, t0 + dt, t_max, dt);

        BOOST_TEST(y.x == y_move.x, tt::tolerance(1e-10));
        BOOST_TEST(y.y == y_move.y, tt::tolerance(1e-10));
        errors.push_back(std::abs(y - y_solution));

        y_variable = y_t_variable<const State&>(solver, df_prev, y1, t0 + dt, t_max, dt);
        y_move_variable = y_t_variable(solver_move, df_prev, y1, t0 + dt, t_max, dt);

        BOOST_TEST(y_variable.x == y_move_variable.x, tt::tolerance(1e-10));
        BOOST_TEST(y_variable.y == y_move_variable.y, tt::tolerance(1e-10));
        errors_variable.push_back(std::abs(y_variable - y_solution));
    }

    double err_log_diff, err_log_diff_variable;
    for (int i = 0; i < errors.size(); i++) {
        err_log_diff = log(errors[i]) - log(errors.back());
        err_log_diff_variable = log(errors_variable[i]) - log(errors_variable.back());

        // should be second order accurate
        BOOST_TEST(err_log_diff/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
        BOOST_TEST(err_log_diff_variable/2 == log(dts[i]) - log(dts.back()), tt::tolerance(5e-2));
    }
}

BOOST_AUTO_TEST_SUITE_END()