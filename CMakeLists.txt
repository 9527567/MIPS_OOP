cmake_minimum_required(VERSION 3.10)
#设置项目名称
project(MIPS_OOP VERSION 0.1)
#添加子模块pybind11
add_subdirectory(include/pybind11)
set(CMAKE_CXX_STANDARD 17)

if(UNIX)
    message("current platform: Linux ")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2")
elseif(WIN32)
    message("current platform: Windows ")
else()
    message("current platform: unknown")
endif()
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)
include_directories(src)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)
if(Boost_FOUND)
    message ("boost found")
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()
aux_source_directory(src SOURCE)

#add_executable(example ${SOURCE})
pybind11_add_module(example
        ${SOURCE}
        )
target_link_libraries(example
        PRIVATE
        Boost::container
        )
if(OpenMP_CXX_FOUND)
    target_link_libraries(example PUBLIC OpenMP::OpenMP_CXX)
endif()
