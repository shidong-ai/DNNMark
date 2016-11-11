// The MIT License (MIT)
// 
// Copyright (c) 2016 Northeastern University
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CORE_INCLUDE_DNNMARK_H_
#define CORE_INCLUDE_DNNMARK_H_

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <cudnn.h>

#include "dnn_param.h"


namespace dnnmark {

#define CUDA_CALL(x) \
do {\
  cudaError_t ret = x;\
  if(ret != cudaSuccess) {\
    std::cout << "CUDA Error at " << __FILE__ << __LINE__ << std::endl;\
    std::cout << cudaGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CURAND_CALL(x) \
do {\
  if((x) != CURAND_STATUS_SUCCESS) {\
    std::cout << "CURAND Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CUDNN_CALL(x) \
do {\
  cudnnStatus_t ret = x;\
  if(ret != CUDNN_STATUS_SUCCESS) {\
    std::cout << "CUDNN Error at " << __FILE__ << __LINE__;\
    std::cout << cudnnGetErrorString(ret) << std::endl;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CUBLAS_CALL(x) \
do {\
  if((x) != CUBLAS_STATUS_SUCCESS) {\
    std::cout << "CUDNN Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#define CONFIG_CHECK(x) \
do {\
  if ((x) != 0) {\
    std::cout << "Parse configuration Error at " << __FILE__ << __LINE__;\
    exit(EXIT_FAILURE);\
  }\
} while(0)\

#ifdef DOUBLE_TEST
#define TestType double
#else
#define TestType float
#endif

// Code courtesy of Caffe
template <typename T>
class DataType;
template class DataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template class DataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

// Benchmark running mode
// None: the benchmark haven't been setup
// Standalone: only one layer will be benchmarked
// Composed: multiple layers will be benchmarked, maybe a whole network
enum RunMode {
  NONE = 0,
  STANDALONE,
  COMPOSED
};

// Layer type
enum LayerType {
  DATA = 0,
  CONVOLUTION,
  POOLING,
  ACTIVIATION,
  LRN,
  FC,
  SOFTMAX
};


template <typename T>
class DNNMark {
 private:
  RunMode run_mode_;
  std::map<int, Layer<T> *> layers_map_;
  std::map<int, LayerType> layer_type_map_;
  std::vector<Layer<T> *> composed_model_;
  int num_layers_;

  // Memory manager
  
 public:
  DNNMark();
  int ParseAllConfig(const std::string &config_file);
  int ParseDNNMarkConfig(const std::string &config_file);
  int ParseDataConfig(const std::string &config_file);
  int ParseConvolutionConfig(const std::string &config_file);
  int Initialize();
  
};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNNMARK_H_
