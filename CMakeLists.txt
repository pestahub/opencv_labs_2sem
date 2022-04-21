cmake_minimum_required(VERSION 3.0)
project( opencv_labs )
set(OpenCV_DIR /home/pesta/opencv3.4.1/build)
find_package(OpenCV REQUIRED PATHS "/home/pesta/opencv3.4.1/build")



add_executable( lab3_1 lab3/src/task1.cpp )
add_executable( lab3_2 lab3/src/task2.cpp )
add_executable( lab3_3 lab3/src/task3.cpp )
add_executable( lab3_4 lab3/src/task4.cpp )

add_executable( lab5 lab5/src/main.cpp )

target_link_libraries( lab3_1 ${OpenCV_LIBS} )
target_link_libraries( lab3_2 ${OpenCV_LIBS} )
target_link_libraries( lab3_3 ${OpenCV_LIBS} )
target_link_libraries( lab3_4 ${OpenCV_LIBS} )

target_link_libraries( lab5 ${OpenCV_LIBS} )
