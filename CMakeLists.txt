cmake_minimum_required(VERSION 3.8)
project(moveit_analysis)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
    scripts/view_metrics.py
    nodes/scene_monitor.py
    DESTINATION lib/${PROJECT_NAME})

ament_package()

