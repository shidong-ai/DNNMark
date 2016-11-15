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

#ifndef CORE_INCLUDE_DNN_PARAM_H_
#define CORE_INCLUDE_DNN_PARAM_H_

#include <iostream>
#include <cudnn.h>

namespace dnnmark {

struct DataDim {
  int n_;
  int c_;
  int h_;
  int w_;

  DataDim()
  : n_(0), c_(0), h_(0), w_(0) {}
};

inline std::ostream &operator<<(std::ostream &os, const DataDim &data_dim) {
  os << std::endl;
  os << "[Data Dim] N: " << data_dim.n_ << std::endl;
  os << "[Data Dim] C: " << data_dim.c_ << std::endl;
  os << "[Data Dim] H: " << data_dim.h_ << std::endl;
  os << "[Data Dim] W: " << data_dim.w_ << std::endl;
  return os;
}

struct ConvolutionParam {
  cudnnConvolutionMode_t mode_;
  int output_num_;
  int pad_h_;
  int pad_w_;
  int stride_u_;
  int stride_v_;
  int upscale_x_;
  int upscale_y_;
  int kernel_size_h_;
  int kernel_size_w_;
  cudnnConvolutionFwdPreference_t conv_fwd_pref_;
  cudnnConvolutionBwdFilterPreference_t conv_bwd_filter_pref_;
  cudnnConvolutionBwdDataPreference_t conv_bwd_data_pref_;
  ConvolutionParam()
  : mode_(CUDNN_CROSS_CORRELATION), output_num_(32),
    pad_h_(2), pad_w_(2),
    stride_u_(1), stride_v_(1),
    upscale_x_(1), upscale_y_(1),
    kernel_size_h_(5), kernel_size_w_(5),
    conv_fwd_pref_(CUDNN_CONVOLUTION_FWD_PREFER_FASTEST),
    conv_bwd_filter_pref_(CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST),
    conv_bwd_data_pref_(CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST) {}
  
};

inline std::ostream &operator<<(std::ostream &os,
                         const ConvolutionParam &conv_param) {
  os << std::endl;
  os << "[Convolution Param] Output Num: "
     << conv_param.output_num_ << std::endl;
  os << "[Convolution Param] Pad H: "
     << conv_param.pad_h_ << std::endl;
  os << "[Convolution Param] Pad W: "
     << conv_param.pad_w_ << std::endl;
  os << "[Convolution Param] Stride U: "
     << conv_param.stride_u_ << std::endl;
  os << "[Convolution Param] Stride V: "
     << conv_param.stride_v_ << std::endl;
  os << "[Convolution Param] Kernel Size H: "
     << conv_param.kernel_size_h_ << std::endl;
  os << "[Convolution Param] Kernel Size W: "
     << conv_param.kernel_size_w_ << std::endl; 

  return os;
}

struct PoolingParam {
  cudnnPoolingMode_t mode_;
  int pad_h_;
  int pad_w_;
  int stride_h_;
  int stride_w_;
  int kernel_size_h_;
  int kernel_size_w_;
  PoolingParam()
  : mode_(CUDNN_POOLING_MAX),
    pad_h_(0), pad_w_(0),
    stride_h_(2), stride_w_(2),
    kernel_size_h_(3), kernel_size_w_(3) {}
};

inline std::ostream &operator<<(std::ostream &os,
                         const PoolingParam &pool_param) {
  os << std::endl;
  os << "[Pooling Param] Pad H: "
     << pool_param.pad_h_ << std::endl;
  os << "[Pooling Param] Pad W: "
     << pool_param.pad_w_ << std::endl;
  os << "[Pooling Param] Stride H: "
     << pool_param.stride_h_ << std::endl;
  os << "[Pooling Param] Stride W: "
     << pool_param.stride_w_ << std::endl;
  os << "[Pooling Param] Kernel Size H: "
     << pool_param.kernel_size_h_ << std::endl;
  os << "[Pooling Param] Kernel Size W: "
     << pool_param.kernel_size_w_ << std::endl; 

  return os;
}


struct LRNParam {
  cudnnLRNMode_t mode_;
  int local_size_;
  double alpha_;
  double beta_;
  double k_;
  LRNParam()
  : mode_(CUDNN_LRN_CROSS_CHANNEL_DIM1),
    local_size_(5),
    alpha_(0.0001), beta_(0.75), k_(2.0) {}
};

inline std::ostream &operator<<(std::ostream &os,
                         const LRNParam &lrn_param) {
  os << std::endl;
  os << "[LRN Param] Local size: "
     << lrn_param.local_size_ << std::endl;
  os << "[LRN Param] Alpha: "
     << lrn_param.alpha_ << std::endl;
  os << "[LRN Param] Beta: "
     << lrn_param.beta_ << std::endl;
  os << "[LRN Param] K: "
     << lrn_param.k_ << std::endl;

  return os;
}

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_PARAM_H_
