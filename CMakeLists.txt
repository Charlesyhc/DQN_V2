project(adnn)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3")



#caffe
include_directories(/work/caffe-master/build/install/include)
link_directories(/work/caffe-master/build/install/lib)

#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#lapack
include_directories(/usr/local/opt/lapack/include)
link_directories(/usr/local/opt/lapack/lib)

#openblas
include_directories(/usr/local/opt/openblas/include)
link_directories(/usr/local/opt/openblas/lib)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Boost
set(Boost_USE_STATIC_LIBS OFF)  # only find static libs
set(Boost_USE_DEBUG_LIBS OFF) # ignore debug libs and
set(Boost_USE_RELEASE_LIBS ON)  # only find release libs
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS system program_options thread timer chrono REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()
message(STATUS ${Boost_LIBRARIES})



#include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

#file copy for prototxt
file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/prototxt/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/project )

set(ADNN_LINK_LIBS
  caffe
  caffeproto
  protobuf
  glog
  gflags
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(project)

