cmake_minimum_required(VERSION 3.20)
project(ImagePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)   # Turing, Ampere, Ada - edit for your GPU
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(ipp_core STATIC src/Bmp.cpp src/Kernels.cu src/Pipeline.cpp)
target_include_directories(ipp_core PUBLIC include)
target_link_libraries(ipp_core PUBLIC CUDA::cudart_static Threads::Threads)
target_compile_options(ipp_core PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /permissive->
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(ImagePipeline src/main.cpp)
target_link_libraries(ImagePipeline PRIVATE ipp_core)

add_executable(ImagePipelineTests tests/Tests.cpp)
target_link_libraries(ImagePipelineTests PRIVATE ipp_core)
enable_testing()
add_test(NAME unit_tests COMMAND ImagePipelineTests)

if(MSVC)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ImagePipeline)
endif()
