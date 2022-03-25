cmake_minimum_required(VERSION 3.0)
project( opencv_labs )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( lab3_1 lab3/src/task1.cpp )
add_executable( lab3_2 lab3/src/task2.cpp )
add_executable( lab3_3 lab3/src/task3.cpp )
add_executable( lab3_4 lab3/src/task4.cpp )


target_link_libraries( lab3_1 ${OpenCV_LIBS} )
target_link_libraries( lab3_2 ${OpenCV_LIBS} )
target_link_libraries( lab3_3 ${OpenCV_LIBS} )
target_link_libraries( lab3_4 ${OpenCV_LIBS} )
