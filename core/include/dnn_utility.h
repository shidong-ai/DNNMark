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

#ifdef NVIDIA_CUDNN
#include "cudnn.h"
#endif

#ifdef AMD_MIOPEN
#include <miopen/miopen.h>
#endif

#include "common.h"
#include "dnn_param.h"

namespace dnnmark {

#ifdef NVIDIA_CUDNN
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
#endif

#ifdef AMD_MIOPEN
class Handle {
 private:
  miopenHandle_t *miopen_handles_;
  int num_handles_;
 public:
  Handle();
  Handle(int num);
  ~Handle();
  miopenHandle_t Get();
  miopenHandle_t Get(int index);
  int num() { return num_handles_; }

};
#endif

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
#ifdef NVIDIA_CUDNN
  cudnnTensorDescriptor_t desc_;
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t desc_;
#endif
  
 public:
  DataTensor()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateTensorDescriptor(&desc_));
#endif
  }

  ~DataTensor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyTensorDescriptor(desc_));
#endif
  }

  void Set(int n, int c, int h, int w) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc_,
                                            CUDNN_TENSOR_NCHW,
                                            DataType<T>::type,
                                            n, c, h, w));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSet4dTensorDescriptor(desc_,
                                              DataType<T>::type,
                                              n, c, h, w));
#endif
    }
    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnTensorDescriptor_t Get() {
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t Get() {
#endif
    if (set_)
      return desc_;
    return nullptr;
  }

};

template <typename T>
class ConvolutionDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t filter_desc_;
  miopenConvolutionDescriptor_t conv_desc_;
#endif

 public:
  ConvolutionDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&conv_desc_));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&filter_desc_));
#endif
  }

  ~ConvolutionDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(conv_desc_));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(filter_desc_));
#endif
  }

  void Set(const ConvolutionParam &param, int num_channel) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                 param.pad_h_, param.pad_w_,
                 param.stride_u_, param.stride_v_,
                 param.upscale_x_, param.upscale_y_,
                 param.mode_));

      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                 DataType<T>::type, CUDNN_TENSOR_NCHW,
                 param.output_num_, num_channel,
                 param.kernel_size_h_, param.kernel_size_w_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenInitConvolutionDescriptor(conv_desc_,
                 param.mode_,
                 param.pad_h_, param.pad_w_,
                 param.stride_u_, param.stride_v_,
                 param.upscale_x_, param.upscale_y_));

      MIOPEN_CALL(miopenSet4dTensorDescriptor(filter_desc_,
                 DataType<T>::type,
                 param.output_num_, num_channel,
                 param.kernel_size_h_, param.kernel_size_w_));
#endif
    }
    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnFilterDescriptor_t GetFilter() {
#endif
#ifdef AMD_MIOPEN
  miopenTensorDescriptor_t GetFilter() {
#endif
    if (set_)
      return filter_desc_;
    return nullptr;
  }

#ifdef NVIDIA_CUDNN
  cudnnConvolutionDescriptor_t GetConv() {
#endif
#ifdef AMD_MIOPEN
  miopenConvolutionDescriptor_t GetConv() {
#endif
    if (set_)
      return conv_desc_;
    return nullptr;
  }


};

template <typename T>
class PoolingDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnPoolingDescriptor_t pooling_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenPoolingDescriptor_t pooling_desc_;
#endif
 public:
  PoolingDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreatePoolingDescriptor(&pooling_desc_));
#endif
  }

  ~PoolingDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyPoolingDescriptor(pooling_desc_));
#endif
  }

  void Set(const PoolingParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetPooling2dDescriptor_v4(pooling_desc_,
                 param.mode_, CUDNN_PROPAGATE_NAN,
                 param.kernel_size_h_, param.kernel_size_w_,
                 param.pad_h_, param.pad_w_,
                 param.stride_h_, param.stride_w_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetPooling2dDescriptor_v4(pooling_desc_,
                 param.mode_,
                 param.kernel_size_h_, param.kernel_size_w_,
                 param.pad_h_, param.pad_w_,
                 param.stride_h_, param.stride_w_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnPoolingDescriptor_t Get() {
#endif
#ifdef AMD_MIOPEN
  miopenPoolingDescriptor_t Get() {
#endif
    if (set_)
      return pooling_desc_;
    return nullptr;
  }

};

template <typename T>
class LRNDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnLRNDescriptor_t lrn_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenLRNDescriptor_t lrn_desc_;
#endif
 public:
  LRNDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateLRNDescriptor(&lrn_desc_));
#endif
#endif
  }

  ~LRNDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyLRNDescriptor(lrn_desc_));
#endif
  }

  void Set(const LRNParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                 param.local_size_,
                 param.alpha_, param.beta_,
                 param.k_));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetLRNDescriptor(lrn_desc_,
                 param.local_size_,
                 param.alpha_, param.beta_,
                 param.k_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnLRNDescriptor_t Get() {
#endif
#ifdef AMD_MIOPEN
  miopenLRNDescriptor_t Get() {
#endif
    if (set_)
      return lrn_desc_;
    return nullptr;
  }

};

template <typename T>
class ActivationDesc : public Descriptor {
 private:
#ifdef NVIDIA_CUDNN
  cudnnActivationDescriptor_t activation_desc_;
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t activation_desc_;
#endif
 public:
  ActivationDesc()
  : Descriptor() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenCreateActivationDescriptor(&activation_desc_));
#endif
  }

  ~ActivationDesc() {
#ifdef NVIDIA_CUDNN
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
#endif
#ifdef AMD_MIOPEN
    MIOPEN_CALL(miopenDestroyActivationDescriptor(activation_desc_));
#endif
  }

  void Set(const ActivationParam &param) {
    if (!set_) {
#ifdef NVIDIA_CUDNN
      CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_,
                 param.mode_,
                 CUDNN_PROPAGATE_NAN,
                 double(0.0)));
#endif
#ifdef AMD_MIOPEN
      MIOPEN_CALL(miopenSetActivationDescriptor(activation_desc_,
                 param.mode_,
                 param.alpha_,
                 param.beta_,
                 param.power_));
#endif
    }

    set_ = true;
  }

#ifdef NVIDIA_CUDNN
  cudnnActivationDescriptor_t Get() {
#endif
#ifdef AMD_MIOPEN
  miopenActivationDescriptor_t Get() {
#endif
    if (set_)
      return activation_desc_;
    return nullptr;
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_UTILITY_H_
