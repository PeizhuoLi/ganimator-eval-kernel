cmake_minimum_required(VERSION 3.4...3.18)
project(ganimator_eval_kernel)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(pybind11)
pybind11_add_module(ganimator_eval_kernel src/main.cpp)

target_compile_definitions(ganimator_eval_kernel
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
