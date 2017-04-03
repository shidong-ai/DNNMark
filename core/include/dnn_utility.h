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

#include <iostream>
#include "cudnn.h"
#include "common.h"
#include "dnn_param.h"

namespace dnnmark {

class Handle {
 private:
  cudnnHandle_t *cudnn_handles_;
  cublasHandle_t *blas_handles_;
  int num_cudnn_handles_;
  int num_blas_handles_;
 public:
  Handle();
  Handle(int num);
  ~Handle();
  cudnnHandle_t GetCudnn();
  cudnnHandle_t GetCudnn(int index);
  cublasHandle_t GetBlas();
  cublasHandle_t GetBlas(int index);
  int num_cudnn() { return num_cudnn_handles_; }
  int num_blas() { return num_blas_handles_; }

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
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc_,
                                            CUDNN_TENSOR_NCHW,
                                            DataType<T>::type,
                                            n, c, h, w));
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
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
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

  cudnnConvolutionDescriptor_t GetConv() {
    if (set_)
      return conv_desc_;
    return nullptr;
  }


};

template <typename T>
class PoolingDesc : public Descriptor {
 private:
  cudnnPoolingDescriptor_t pooling_desc_;
 public:
  PoolingDesc()
  : Descriptor() {
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
  }

  ~PoolingDesc() {
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  void Set(const PoolingParam &param) {
    if (!set_) {
      CUDNN_CALL(cudnnSetPooling2dDescriptor_v4(pooling_desc_,
                 param.mode_, CUDNN_PROPAGATE_NAN,
                 param.kernel_size_h_, param.kernel_size_w_,
                 param.pad_h_, param.pad_w_,
                 param.stride_h_, param.stride_w_));
    }

    set_ = true;
  }

  cudnnPoolingDescriptor_t Get() {
    if (set_)
      return pooling_desc_;
    return nullptr;
  }

};

template <typename T>
class LRNDesc : public Descriptor {
 private:
  cudnnLRNDescriptor_t lrn_desc_;
 public:
  LRNDesc()
  : Descriptor() {
    CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
  }

  ~LRNDesc() {
    CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc_));
  }

  void Set(const LRNParam &param) {
    if (!set_) {
      CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                 param.local_size_,
                 param.alpha_, param.beta_,
                 param.k_));
    }

    set_ = true;
  }

  cudnnLRNDescriptor_t Get() {
    if (set_)
      return lrn_desc_;
    return nullptr;
  }

};

template <typename T>
class ActivationDesc : public Descriptor {
 private:
  cudnnActivationDescriptor_t activation_desc_;
 public:
  ActivationDesc()
  : Descriptor() {
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
  }

  ~ActivationDesc() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
  }

  void Set(const ActivationParam &param) {
    if (!set_) {
      CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_,
                 param.mode_,
                 CUDNN_PROPAGATE_NAN,
                 double(0.0)));
    }

    set_ = true;
  }

  cudnnActivationDescriptor_t Get() {
    if (set_)
      return activation_desc_;
    return nullptr;
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_UTILITY_H_
