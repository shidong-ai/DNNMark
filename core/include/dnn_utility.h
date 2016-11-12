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

#ifndef CORE_INCLUDE_DNN_UTILITY_H_
#define CORE_INCLUDE_DNN_UTILITY_H_

#include <cudnn.h>
#include "dnnmark.h"

namespace dnnmark {

class Handle {
 private:
  cudnnHandle_t *handles_;
  int size_;
 public:
  Handle();
  Handle(int num);
  ~Handle();
  cudnnHandle_t getHandle();
  cudnnHandle_t getHandle(int index);
  int size() { return size_; }

};

class Descriptor {
 protected:
  bool set_;
 public:
  Descriptor();
  ~Descriptor();
  bool isSet();
};

template <typename T>
class DataTensor : public Descriptor {
 private:
  // Tensor dimensions
  int n_;
  int c_;
  int h_;
  int w_;

  cudnnTensorDescriptor_t desc_;
  
 public:
  DataTensor()
  : Descriptor() {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc_));
  }

  ~DataTensor() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc_));
  }

  void Set(int n, int c, int h, int w) {
    if (!set_) {
      n_ = n;
      c_ = c;
      h_ = h;
      w_ = w;
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc_,
                                            CUDNN_TENSOR_NCHW,
                                            DataType<T>::type,
                                            n_, c_, h_, w_));
    }
    set_ = true;
  }

  cudnnTensorDescriptor_t Get() {
    if (set_)
      return desc_;
    return nullptr;
  }

};

template <typename T>
class ConvolutionDesc : public Descriptor {
 private:
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;

 public:
  ConvolutionDesc()
  : Descriptor() {
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
  }

  ~ConvolutionDesc() {
    cudnnDestroyConvolutionDescriptor(conv_desc_);
    cudnnDestroyFilterDescriptor(filter_desc_);
  }

  void Set(const ConvolutionParam &param, int num_channel) {
    if (!set_) {
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                 param.pad_h_, param.pad_w_,
                 param.stride_u_, param.stride_v_,
                 param.upscale_x_, param.upscale_y_,
                 param.mode_));

      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                 DataType<T>::type, CUDNN_TENSOR_NCHW,
                 param.output_num_, num_channel,
                 param.kernel_size_h_, param.kernel_size_w_));
    }
    set_ = true;
  }

  cudnnFilterDescriptor_t GetFilter() {
    if (set_)
      return filter_desc_;
    return nullptr;
  }

  cudnnFilterDescriptor_t GetConv() {
    if (set_)
      return conv_desc_;
    return nullptr;
  }


};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_UTILITY_H_
