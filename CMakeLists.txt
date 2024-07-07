cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")
project(qap)

file(GLOB gpu_source_files "src/*.cu")
file(GLOB header_files "include/*.hpp")

enable_language("CUDA")
add_executable(${CMAKE_PROJECT_NAME} ${gpu_source_files} ${header_files})
message(STATUS "USE_CUDA: ${USE_CUDA}")
add_definitions(-DUSE_CUDA)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
