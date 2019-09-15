cmake_minimum_required(VERSION 3.14)
project(projectionDistortionCorrection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(projectionDistortionCorrection main.cpp EnergyFunctions.cpp MeshUtils.cpp MeshUtils.h EnergyFunctions.h)

target_link_libraries(projectionDistortionCorrection ${OpenCV_LIBS})
target_link_libraries(projectionDistortionCorrection ${CERES_LIBRARIES})

