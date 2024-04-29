cmake_minimum_required(VERSION 3.13)
project(QAP
LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(
        include
)

set(SOURCE_FILES
        src/solution.cpp
        src/population.cpp
        src/bee_algorithm.cpp
        )

add_executable(qap ${SOURCE_FILES} main.cpp)