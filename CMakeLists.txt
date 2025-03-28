cmake_minimum_required(VERSION 3.24)
project(Heat_Equation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-Wall -O3 -fopenmp)
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++20 -fopenmp")

include_directories("${PROJECT_SOURCE_DIR}/include")
file(GLOB CPP_SOURCES "source/*.cpp")

# add_executable(Heat_Equation ${CPP_SOURCES} main.cpp)

add_library(library STATIC ${CPP_SOURCES})

link_directories(${CMAKE_SOURCE_DIR}/build)

add_library(heat_equation MODULE ${CMAKE_SOURCE_DIR}/heat_equation.pyx)

target_link_libraries(heat_equation library)
target_include_directories(heat_equation PRIVATE ${CMAKE_SOURCE_DIR}/include)
