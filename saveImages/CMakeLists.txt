cmake_minimum_required(VERSION 2.8)

project( SaveMoving )
find_package( OpenCV REQUIRED)
add_executable( SaveMoving saveMovingObject.cpp )
target_link_libraries( SaveMoving ${OpenCV_LIBS} )
