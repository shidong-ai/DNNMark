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

#ifndef CORE_INCLUDE_DNN_WRAPPER_H_ 
#define CORE_INCLUDE_DNN_WRAPPER_H_

#include "common.h"
#include "dnn_utility.h"

namespace dnnmark {

//
// Convolution forward/backward functions
//

template <typename T>
inline void dnnmarkConvolutionForward(const Handle &handle,
                                      RunMode mode, int idx,
                                      const void *alpha,
                                      const DataTensor<T> &bottom_desc,
                                      const void *x,
                                      const ConvolutionDesc<T> &conv_desc,
                                      const void *w,
                                      const ConvAlgo<T> &conv_algo,
                                      void *workspace,
                                      size_t workspace_in_bytes,
                                      const void *beta,
                                      const DataTensor<T> &top_desc,
                                      void *y) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnConvolutionForward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             alpha,
             bottom_desc.Get(), x,
             conv_desc.GetFilter(), w,
             conv_desc_.GetConv(),
             conv_algo.GetFwdAlgo(), workspace, workspace_in_bytes,
             beta,
             top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenConvolutionForward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              alpha,
              bottom_desc.Get(), x,
              conv_desc.GetFilter(), w,
              conv_desc.GetConv(),
              conv_algo.GetFwdAlgo(),
              beta,
              top_desc.Get(), y,
              workspace, workspace_in_bytes));
#endif

}

template <typename T>
inline void dnnmarkConvolutionBackwardData(const Handle &handle,
                                           RunMode mode, int idx,
                                           const void *alpha,
                                           const DataTensor<T> &top_desc,
                                           const void *dy,
                                           const ConvolutionDesc<T> &conv_desc,
                                           const void *w,
                                           const ConvAlgo<T> &conv_algo,
                                           void *workspace,
                                           size_t workspace_in_bytes,
                                           const void *beta,
                                           const DataTensor<T> &bottom_desc,
                                           void *dx) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnConvolutionBackwardData(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             alpha,
             conv_desc.GetFilter(), w,
             top_desc.Get(), dy,
             conv_desc.GetConv(),
             conv_algo.GetBwdDataAlgo(),
             workspace, workspace_in_bytes,
             beta,
             bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenConvolutionBackwardData(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              alpha,
              top_desc.Get(), dy,
              conv_desc.GetFilter(), w,
              conv_desc.GetConv(),
              conv_algo.GetBwdDataAlgo(),
              beta,
              bottom_desc.Get(), dx,
              workspace, workspace_in_bytes));
#endif
}

template <typename T>
inline void dnnmarkConvolutionBackwardFilter(const Handle &handle,
                                             RunMode mode, int idx,
                                             const void *alpha,
                                             const DataTensor<T> &bottom_desc,
                                             const void *x,
                                             const DataTensor<T> &top_desc,
                                             const void *dy,
                                             const ConvolutionDesc<T> &conv_desc,
                                             const ConvAlgo<T> &conv_algo,
                                             void *workspace,
                                             size_t workspace_in_bytes,
                                             const void *beta,
                                             void *dw) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnConvolutionBackwardFilter(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             alpha,
             bottom_desc.Get(), x,
             top_desc.Get(), dy,
             conv_desc.GetConv(),
             conv_algo.GetBwdFilterAlgo(),
             workspace, workspace_in_bytes,
             beta,
             conv_desc.GetFilter(), dw));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenConvolutionBackwardWeights(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              alpha,
              top_desc.Get(), dy,
              bottom_desc.Get(), x,
              conv_desc.GetConv(),
              conv_algo.GetBwdDataAlgo(),
              beta,
              conv_desc.GetFilter(), dw,
              workspace, workspace_in_bytes));
#endif
}

//
// Pooling forward/backward functions
//

//
// Activation forward/backward functions
//

//
// LRN forward/backward functions
//

//
// Fully Connected forward/backward functions
//

//
// Softmax forward/backward functions
//

//
// Batch Normalization forward/backward functions
//

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_WRAPPER_H_
