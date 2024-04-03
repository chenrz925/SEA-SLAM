cmake_minimum_required(VERSION 3.0.2)
project(slam)

SET(CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Rlease")
endif()
## no warning
add_definitions(-w)
## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# set optimization level to -O1  if you want to debug using gdb
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -pthread -lpthread")# -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -pthread -lpthread")# -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  tf
  dynamic_reconfigure
  message_generation
  semantic_cnn
  detect_3d_cuboid
  line_lbd
)
# set(${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4/)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Glog REQUIRED)

# set(BOOST_INCLUDE_DIR )
# include(../Thirdparty/thirdparty.cmake)

set(DBoW2 ${PROJECT_SOURCE_DIR}/../3rdparty/DBoW2)
set(G2O ${PROJECT_SOURCE_DIR}/../3rdparty/g2o)
message(WARNING ${DBoW2})
message(WARNING ${G2O})
# file(GLOB_RECURSE BOOST_LIBS "${BOOST}/lib/*.so")
# file(GLOB_RECURSE OPENCV_LIBS "${OPENCV}/lib/*.so")
# file(GLOB_RECURSE CERES_LIBS "${CERES}/lib/*.so")
# file(GLOB_RECURSE GLOG_LIBS "${GLOG}/lib/*.so")
# file(GLOB_RECURSE PANGOLIN_LIBS "${PANGOLIN}/lib/*.so")
# file(GLOB_RECURSE GLEW_LIBS "${GLEW}/*.so")
# file(GLOB_RECURSE OPENGL_LIBS "${OPENGL}/*.so")
## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES SLAM
 CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs  cv_bridge image_transport tf dynamic_reconfigure message_generation semantic_cnn
 detect_3d_cuboid
 line_lbd
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
## added========================================================================

# Munkres
set(MUNKRES_INCLUDE ${PROJECT_SOURCE_DIR}/lib/munkres-cpp/src)
set(MUNKRES_LIB ${PROJECT_SOURCE_DIR}/lib/munkres-cpp/libmunkres.so)
# list(APPEND CMAKE_LINK_LIBS ${MUNKRES_LIB})
include_directories(${MUNKRES_INCLUDE})

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  # ${Glog_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  # ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/CameraModels
  ${DBoW2}
  ${G2O}
  # ${EIGEN}/include
  # ${OPENCV}/include
  # ${PANGOLIN}/include
  # ${GLOG}/include

  # ${GTK3_INCLUDE_DIRS}
)
aux_source_directory(src DIR_LIB_SRCS)
# Declare a C++ library
add_library(${PROJECT_NAME}
    ${DIR_LIB_SRCS}
    ${PROJECT_SOURCE_DIR}/src/CameraModels/Pinhole.cpp
    ${PROJECT_SOURCE_DIR}/src/CameraModels/KannalaBrandt8.cpp
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/CameraModels/GeometricCamera.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/CameraModels/Pinhole.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/CameraModels/KannalaBrandt8.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/ImgProcessor.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/MapObject.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/ObjectDetection.h
    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/WeightCalcu.h
    ${PROJECT_SOURCE_DIR}/src/ImgProcessor.cpp
    ${PROJECT_SOURCE_DIR}/src/Cluster.cc
    ${PROJECT_SOURCE_DIR}/src/MapObject.cc
    ${PROJECT_SOURCE_DIR}/src/ObjectDetection.cc
    ${PROJECT_SOURCE_DIR}/src/Map_Add.cc
    ${PROJECT_SOURCE_DIR}/src/Tracking_Add.cc
    ${PROJECT_SOURCE_DIR}/src/Frame_Add.cc
    ${PROJECT_SOURCE_DIR}/src/Atlas_Add.cc
    ${PROJECT_SOURCE_DIR}/src/WeightCalcu.cpp

  )

  target_link_libraries(
    ${PROJECT_NAME}
    ${OPENCV_LIB_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${Pangolin_LIBRARIES}
    # ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/ #libDBoW2.so
    # ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/ #libg2o.so
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    # ${Glog_LIB_DIRS}
    ${MUNKRES_LIB}
    # ${GTK3_LIBRARY_DIRS}
    -lboost_serialization
    -lcrypto
    # libpango_windowing.so
    # ${BOOST_LIBS}
    # ${OPENCV_LIBS}
    # ${CERES_LIBS}
    ${PROJECT_SOURCE_DIR}/../3rdparty/g2o/lib/libg2o.so
    ${PROJECT_SOURCE_DIR}/../3rdparty/DBoW2/lib/libDBoW2.so
    # ${GLOG_LIBS}
    # ${PANGOLIN_LIBS}
    # ${GLEW_LIBS}
    # ${OPENGL_LIBS}
  )

# LINK_directories(/usr/local/lib)


add_executable(Dynamic_RGBD demo/ros_tum_rgbd_node.cpp)
add_executable(Dynamic_Mono_Inertial demo/ros_mono_inertial.cpp)
add_executable(Dynamic_Mono demo/ros_mono.cc)
add_executable(Dynamic_Stereo_Inertial demo/ros_stereo_inertial.cpp)
add_executable(Dynamic_Stereo demo/ros_stereo.cc)
add_executable(Dynamic_Stereo_Inertial_kitti demo/stereo_inertial_kitti.cc)
add_executable(Dynamic_IMU_Stereo_euroc demo/stereo_inertial_euroc.cc)
add_executable(Dynamic_Stereo_kitti demo/stereo_kitti.cpp)

add_dependencies(Dynamic_RGBD ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_RGBD   ${PROJECT_NAME} )

add_dependencies(Dynamic_Mono_Inertial ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Mono_Inertial   ${PROJECT_NAME} )
add_dependencies(Dynamic_Mono  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Mono   ${PROJECT_NAME} )
add_dependencies(Dynamic_Stereo_Inertial ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Stereo_Inertial   ${PROJECT_NAME} )
add_dependencies(Dynamic_Stereo ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Stereo   ${PROJECT_NAME} )
add_dependencies(Dynamic_Stereo_Inertial_kitti ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Stereo_Inertial_kitti   ${PROJECT_NAME} )
add_dependencies(Dynamic_IMU_Stereo_euroc ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_IMU_Stereo_euroc   ${PROJECT_NAME} )
add_dependencies(Dynamic_Stereo_kitti ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Dynamic_Stereo_kitti   ${PROJECT_NAME} )
## added========================================================================
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(Dynamic_RGBD src/ros_tum_rgbd_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(Dynamic_RGBD ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(Dynamic_RGBD   ${catkin_LIBRARIES} )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_SLAM.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
