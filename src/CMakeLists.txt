add_library(rho_a SHARED atmospheric_density.cpp)

add_library(fragment SHARED fragment.cpp)
target_link_libraries(fragment rho_a)

add_library(FCM SHARED fcm.cpp)
target_link_libraries(FCM rho_a fragment)

add_library(python_api python_api.cpp)
target_link_libraries(python_api FCM rho_a fragment)
link_python(python_api)