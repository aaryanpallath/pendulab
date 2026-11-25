cmake_minimum_required(VERSION 3.20)
project(pendulab CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_library(pendulab INTERFACE)
target_include_directories(pendulab INTERFACE include)

add_executable(dpsim tools/dpsim.cpp)
target_link_libraries(dpsim PRIVATE pendulab)

enable_testing()
add_executable(test_sim tests/test_sim.cpp)
target_link_libraries(test_sim PRIVATE pendulab)
add_test(NAME test_sim COMMAND test_sim)
