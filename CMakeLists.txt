cmake_minimum_required( VERSION 2.8 )
project (libslam)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3")

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
include_directories( ${PROJECT_SOURCE_DIR}/include )

find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## executable
add_executable(pose_2d2d src/pose_2d2d.cpp)
target_link_libraries(pose_2d2d ${OpenCV_LIBRARIES})






