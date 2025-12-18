cmake_minimum_required(VERSION 3.10)
project(MultiscaleDeformableAttnPlugin CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find TensorRT
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib
    PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(WARNING "TensorRT not found. Please set TENSORRT_ROOT to your TensorRT installation directory.")
    message(WARNING "Example: cmake -DTENSORRT_ROOT=/path/to/TensorRT ..")
else()
    message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
endif()

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)

# Plugin source files
set(PLUGIN_SOURCES
    MultiscaleDeformableAttnPlugin.cpp
    MultiscaleDeformableAttnKernel.cu
)

# Build plugin as shared library
add_library(MultiscaleDeformableAttnPlugin SHARED ${PLUGIN_SOURCES})

target_link_libraries(MultiscaleDeformableAttnPlugin
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(MultiscaleDeformableAttnPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS MultiscaleDeformableAttnPlugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES MultiscaleDeformableAttnPlugin.h
    DESTINATION include
)

# Example executable (optional)
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES AND TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    add_executable(plugin_example example.cpp)
    target_link_libraries(plugin_example
        MultiscaleDeformableAttnPlugin
        ${TENSORRT_LIBRARY}
        ${CUDA_LIBRARIES}
    )
    set_target_properties(plugin_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
