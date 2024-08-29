cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(slice_and_batch_trt_plugin LANGUAGES CXX CUDA)
#set(TRT_ROOT "/home/humble/shared/libraries/TensorRT-10.0.1.6")
set(TRT_ROOT "/home/humble/shared/libraries/TensorRT-10.1.0.27")
set(TRT_INCLUDES "${TRT_ROOT}/include;${TRT_ROOT}/samples/common;${TRT_ROOT}/samples")
set(TRT_LIBS "${TRT_ROOT}/lib/aarch64-linux-gnu")


set(CMAKE_CUDA_ARCHITECTURES "87;")

add_library(slice_and_batch_lib SHARED
  src/sliceAndBatchKernel.cu  src/sliceAndBatchKernel.h  src/sliceAndBatchPlugin.cpp src/sliceAndBatchPlugin.h
  ${TRT_ROOT}/samples/common/logger.cpp 
)

# Request that slice_and_batch_lib be built with -std=c++14
# As this is a public compile feature anything that links to slice_and_batch_lib
# will also build with -std=c++14
target_compile_features(slice_and_batch_lib PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the slice_and_batch
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( slice_and_batch_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(slice_and_batch_lib PUBLIC  ${TRT_INCLUDES} /usr/local/cuda/include)
target_link_libraries(slice_and_batch_lib PUBLIC 
  "${TRT_LIBS}/libnvinfer.so"
  "${TRT_LIBS}/libnvinfer_plugin.so"
  "${TRT_LIBS}/libnvonnxparser.so"
)

#include dirs:
# /home/a249s197/work/libraries/TensorRT/samples/common
# /home/a249s197/work/libraries/TensorRT-10.0.1.6/include

add_executable(slice_and_batch_test src/main.cpp)
set_target_properties(slice_and_batch_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(slice_and_batch_test PUBLIC  ${TRT_INCLUDES})
target_link_libraries(slice_and_batch_test PRIVATE slice_and_batch_lib)
