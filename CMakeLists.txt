# CMake version related
cmake_minimum_required(VERSION 2.8.9)
if (${CMAKE_VERSION} VERSION_GREATER 3.0)
  cmake_policy(SET CMP0046 OLD)
endif()

# Set project
project(dnnmark)

# Set path of sub-directory with other cmake file
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Locate CUDA package 
find_package(CUDA REQUIRED)

# Detect GPU device
if(CUDA_FOUND)
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR} 
    ${CMAKE_SOURCE_DIR}/tools/has_cuda_gpu.c
    CMAKE_FLAGS 
      -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
      -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)

  # Display number of GPUs found
  message("${RUN_OUTPUT_VAR}")

  # COMPILE_RESULT_VAR is TRUE when compile succeeds
  # RUN_RESULT_VAR is zero when a GPU is found
  if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
      set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
  else()
      set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
  endif()
endif()

option (double-test "Make data type double" OFF)

# Buld NICE library only with CUDA
if (CUDA_FOUND)

  # Print messge to indicate whether GPU device exists
  if (CUDA_HAVE_GPU)
    message(STATUS "GPU device exists")
  else()
    message(WARNING "GPU device does not exist")
  endif()

  #include the built-in findCUDA cmake scripts
  include(FindCUDA)
  
  # Stop if CUDNN_ROOT is not defined
  if (NOT DEFINED CUDNN_ROOT)
    message(FATAL_ERROR "Path to CuDNN is not defined")
  endif()

  if (EXISTS ${CUDNN_ROOT})
    message(STATUS "CuDNN library root path: " ${CUDNN_ROOT})
    include_directories(${CUDNN_ROOT}/include)
  else()
    message(FATAL_ERROR "Path to CuDNN library not exists")
  endif()

  # Add deinition of C++11 stardard
  add_definitions(-std=c++11)

  # Enable double data type
  if (double-test)
    add_definitions(-DDOUBLE_TEST)
  endif()
  
  # Set path of DNNMark include files
  set(DNNMARK_INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
  include_directories(${DNNMARK_INCLUDES}) 
  
  # Set source files
  file(GLOB_RECURSE DNNMARK_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} core/src/*.cc)
  message(STATUS "DNNMark Source files: " ${DNNMARK_SOURCES})

  # Set NVCC flags  
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_50,code=sm_50)

  # Find cuda libraries
  find_library(CUDNN_LIBRARY cudnn ${CUDNN_ROOT}/lib64)
  find_library(CUDA_BLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDA_RAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  # Find glog library
  find_library(GLOG_LIBRARY glog)

  # Add NICE library together with CUDA
  include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(${PROJECT_NAME} SHARED ${DNNMARK_SOURCES})
  add_dependencies(${PROJECT_NAME} ${CUDNN_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_BLAS_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_RAND_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_LIBRARIES})
  add_dependencies(${PROJECT_NAME} ${GLOG_LIBRARY})
  target_link_libraries(${PROJECT_NAME} 
                        ${CUDNN_LIBRARY}
                        ${CUDA_BLAS_LIBRARY}
                        ${CUDA_RAND_LIBRARY} 
                        ${CUDA_LIBRARIES}
                        ${GLOG_LIBRARY}
                        m)

else()

  message(FATAL_ERROR "CUDA not exists")

endif()

# Benchmarking
option (enable-benchmarking "Compile all benchmarks of DNNMark" ON)

if (enable-benchmarking)
  # Include the cmake file of gflag
  find_library(GFLAGS_LIBRARY gflags)

  # Invoke cmake files of each benchmark
  add_subdirectory(benchmarks/test_fwd_conv)
  #add_subdirectory(benchmarks/test_bwd_conv)
  add_subdirectory(benchmarks/test_fwd_pool)
  #add_subdirectory(benchmarks/test_bwd_pool)
  add_subdirectory(benchmarks/test_fwd_lrn)
  #add_subdirectory(benchmarks/test_bwd_lrn)
  #add_subdirectory(benchmarks/test_fwd_activation)
  #add_subdirectory(benchmarks/test_bwd_activation)
  #add_subdirectory(benchmarks/test_fwd_fully_connected)
  #add_subdirectory(benchmarks/test_bwd_fully_connected)
  #add_subdirectory(benchmarks/test_fwd_softmax)
  #add_subdirectory(benchmarks/test_bwd_softmax)

endif()

