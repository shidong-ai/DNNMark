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

class ConvolutionParam {
private:
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
public:
  ConvolutionParam(int output_num,
                   int pad,
                   int stride,
                   int kernel_size)
  : mode_(CUDNN_CROSS_CORRELATION), output_num_(output_num),
    pad_h_(pad), pad_w_(pad),
    stride_u_(stride), strid_v_(stride),
    upscale_x_(1), upscale_y_(1),
    kernel_size_h_(kernel_size), kernel_size_w_(kernel_size) {}

  ConvolutionParam(cudnnConvolutionMode_t mode,
                   int output_num,
                   int pad,
                   int stride,
                   int upscale,
                   int kernel_size)
  : mode_(mode), output_num_(output_num),
    pad_h_(pad), pad_w_(pad),
    stride_u_(stride), strid_v_(stride),
    upscale_x_(upscale), upscale_y_(upscale),
    kernel_size_h_(kernel_size), kernel_size_w_(kernel_size) {}

  ConvolutionParam(cudnnConvolutionMode_t mode,
                   int output_num,
                   int pad_h,
                   int pad_w,
                   int stride_u,
                   int stride_v,
                   int upscale_x,
                   int upscale_y,
                   int kernel_size_h,
                   int kernel_size_w)
  : mode_(mode), output_num_(output_num),
    pad_h_(pad_h), pad_w_(pad_w),
    stride_u_(stride_u), strid_v_(stride_v),
    upscale_x_(upscale_x), upscale_y_(upscale_y),
    kernel_size_h_(kernel_size_h), kernel_size_w_(kernel_size_w) {}

  cudnnConvolutionMode_t getMode() { return mode_; }
  int getOutputNum() { return output_num_; }
  int getPadH() { return pad_h_; }
  int getPadW() { return pad_w_; }
  int getStrideU() { return stride_u_; }
  int getStrideV() { return stride_v_; }
  int getUpscaleX() { return upscale_x_; }
  int getUpscaleY() { return upscale_y_; }
  int getKernelSizeH() { return kernel_size_h_; }
  int getKernelSizeW() { return kernel_size_w_; }
};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_PARAM_H_
