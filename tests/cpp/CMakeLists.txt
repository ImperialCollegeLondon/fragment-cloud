add_executable(ODE_tests ode_tests.cpp)
target_link_libraries(ODE_tests Boost::unit_test_framework)
add_test(NAME ODE_tests COMMAND ODE_tests)

add_executable(rho_a_tests atmoshperic_density_tests.cpp)
target_link_libraries(rho_a_tests rho_a Boost::unit_test_framework)
add_test(NAME atmoshperic_density_tests COMMAND rho_a_tests)

add_executable(fragment_tests fragment_tests.cpp)
target_link_libraries(fragment_tests fragment rho_a Boost::unit_test_framework)
add_test(NAME fragment_tests COMMAND fragment_tests)

add_executable(FCM_tests fcm_tests.cpp)
target_link_libraries(FCM_tests FCM fragment rho_a Boost::unit_test_framework)
add_test(NAME FCM_tests COMMAND FCM_tests)

add_executable(solvers_tests solvers_tests.cpp)
target_link_libraries(solvers_tests Boost::unit_test_framework)
add_test(NAME solvers_tests COMMAND solvers_tests)

add_executable(Python_API_tests python_api_tests.cpp)
target_link_libraries(Python_API_tests python_api)
link_python(Python_API_tests)
add_test(NAME Python_API_tests COMMAND Python_API_tests)
