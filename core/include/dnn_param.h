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

#include <cudnn.h>

namespace dnnmark {

struct DataParam {
  int n_;
  int c_;
  int h_;
  int w_;
  DataParam()
  : n_(0), c_(0), h_(0), w_(0) {}
};

class ConvolutionParam {
  std::string name_;
  cudnnConvolutionMode_t mode_;
  int output_num;
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
    stride_u_(1), strid_v_(1),
    upscale_x_(1), upscale_y_(1),
    kernel_size_h_(5), kernel_size_w_(5),
    conv_fwd_pref(CUDNN_CONVOLUTION_FWD_PREFER_FASTEST),
    conv_bwd_filter_pref(CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST),
    conv_bwd_data_pref(CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST) {}
};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_PARAM_H_
