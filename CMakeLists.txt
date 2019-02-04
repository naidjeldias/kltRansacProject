cmake_minimum_required(VERSION 3.8)
project(KLTRansac)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp eightpoint.hpp eightPoint.cpp)
add_executable(KLTRansac ${SOURCE_FILES})

include_directories($(OpenCV_INCLUDE_DIRS))
target_link_libraries(KLTRansac ${OpenCV_LIBS})