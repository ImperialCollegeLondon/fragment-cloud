#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcm {

/**
 * @brief (first order, explicit) forward Euler step for time-independent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT forward_euler(StateT&& current_state,
                          const std::function<DeltaT(const NoRefStateT&)>& df, const TimeT dt) {
    auto next_state = std::forward<StateT>(current_state);
    next_state += df(next_state) * dt;
    
    return next_state;
}

/**
 * @brief (first order, explicit) forward Euler step for time-dependent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param t: current time
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT forward_euler(StateT&& current_state,
                          const std::function<DeltaT(const TimeT, const NoRefStateT&)>& df,
                          const TimeT t, const TimeT dt) {
    auto next_state = std::forward<StateT>(current_state);
    next_state += df(t, next_state) * dt;
    
    return next_state;
}

/**
 * @brief (second order, explicit) improved Euler step for time-independent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT improved_euler(StateT&& current_state,
                           const std::function<DeltaT(const NoRefStateT&)>& df, const TimeT dt) {
    auto next_state = std::forward<StateT>(current_state);
    const auto k = df(next_state) * dt;

    next_state += (k + df(next_state + k) * dt) * 0.5;
    
    return next_state;
}

/**
 * @brief (second order, explicit) improved Euler step for time-dependent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param t: current time
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT improved_euler(StateT&& current_state,
                           const std::function<DeltaT(const TimeT, const NoRefStateT&)>& df,
                           const TimeT t, const TimeT dt) {
    auto next_state = std::forward<StateT>(current_state);
    const auto k = df(t, next_state) * dt;

    next_state += (k + df(t + dt, next_state + k) * dt) * 0.5;
    
    return next_state;
}

/**
 * @brief (forth order, explicit) Runge-Kutta 4 step for time-independent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT RK4(StateT&& current_state,
                const std::function<DeltaT(const TimeT, const NoRefStateT&)>& df,
                const TimeT t, const TimeT dt) {
    auto next_state = std::forward<StateT>(current_state);

    const auto k1 = df(t, next_state) * dt;
    const auto k2 = df(t + dt*0.5, next_state + k1*0.5) * dt;
    const auto k3 = df(t + dt*0.5, next_state + k2*0.5) * dt;
    const auto k4 = df(t + dt, next_state + k3) * dt;

    next_state += (k1 + k2*2 + k3*2 + k4) * (1./6.);
    
    return next_state;
}

/**
 * @brief (forth order, explicit) Runge-Kutta 4 step for time-dependent ODE function
 * 
 * @tparam StateT
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df: ODE function
 * @param t: current time
 * @param dt: time step
 * @return NoRefStateT: next state
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
NoRefStateT RK4(StateT&& current_state, const std::function<DeltaT(const NoRefStateT&)>& df,
                const TimeT dt)  {
    auto next_state = std::forward<StateT>(current_state);

    const auto k1 = df(next_state) * dt;
    const auto k2 = df(next_state + k1*0.5) * dt;
    const auto k3 = df(next_state + k2*0.5) * dt;
    const auto k4 = df(next_state + k3) * dt;

    next_state += (k1 + k2*2 + k3*2 + k4) * (1./6.);
    
    return next_state;
}

/**
 * @brief (second order, explicit, multistep) variable stepsize
 *        Adams-Bashforth 2 scheme for time-independent ODE function
 * 
 * @tparam StateT 
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df_prev: df(previous_state) 
 * @param df: ODE function
 * @param dt: time step
 * @param dt_prev: previous time step
 * @return std::tuple<NoRefStateT, DeltaT> [next state, df(t, current_state)]
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
std::tuple<NoRefStateT, DeltaT> AB2(StateT&& current_state, const DeltaT& df_prev,
                                    const std::function<DeltaT(const NoRefStateT&)>& df, 
                                    const TimeT dt, const TimeT dt_prev) {

    auto next_state = std::forward<StateT>(current_state);
    const auto omega = dt / (2*dt_prev);
    const auto df_current = df(next_state);

    next_state += (df_current * (1+omega) + df_prev * (-omega)) * dt;

    return std::make_tuple(next_state, df_current);
}

/**
 * @brief (second order, explicit, multistep) variable stepsize
 *        Adams-Bashforth 2 scheme for time-dependent ODE function
 * 
 * @tparam StateT 
 * @tparam DeltaT 
 * @tparam TimeT 
 * @tparam NoRefStateT=std::remove_reference_t<StateT> 
 * @param current_state 
 * @param df_prev: df(t - dt_prev, previous_state) 
 * @param df: ODE function
 * @param t: current time
 * @param dt: time step
 * @param dt_prev: previous time step
 * @return std::tuple<NoRefStateT, DeltaT> [next state, df(t, current_state)]
 */
template<class StateT, class DeltaT, class TimeT, class NoRefStateT=std::remove_reference_t<StateT>>
std::tuple<NoRefStateT, DeltaT> AB2(StateT&& current_state, const DeltaT& df_prev,
                                    const std::function<DeltaT(const TimeT, const NoRefStateT&)>& df,
                                    const TimeT t, const TimeT dt, const TimeT dt_prev) {
    auto next_state = std::forward<StateT>(current_state);
    const auto omega = dt / (2*dt_prev);
    const auto df_current = df(t, next_state);

    next_state += (df_current * (1+omega) + df_prev * (-omega)) * dt;

    return std::make_tuple(next_state, df_current);
}

} // namespace fcm

#endif // ! SOLVERS_HPP