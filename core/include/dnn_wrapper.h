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

template <typename T>
inline void dnnmarkActivationForward(const Handle &handle,
                         RunMode mode, int idx,
                         const ActivationDesc<T> &activation_desc,
                         const void *alpha,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         const void *beta,
                         const DataTensor<T> &top_desc,
                         void *y) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnActivationForward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             activation_desc.Get(),
             alpha,
             bottom_desc.Get(), x,
             beta,
             top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationForward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              activation_desc.Get(),
              alpha,
              bottom_desc.Get(), x,
              beta,
              top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkActivationBackward(const Handle &handle,
                         RunMode mode, int idx,
                         const ActivationDesc<T> &activation_desc,
                         const void *alpha,
                         const DataTensor<T> &top_desc,
                         const void *y,
                         const void *dy,
                         const void *beta,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         void *dx) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnActivationBackward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             activation_desc.Get(),
             alpha,
             top_desc.Get(), y,
             top_desc.Get(), dy,
             bottom_desc.Get(), x,
             beta,
             bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationBackward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              activation_desc.Get(),
              alpha,
              top_desc.Get(), y,
              top_desc.Get(), dy,
              bottom_desc.Get(), x,
              beta,
              bottom_desc.Get(), dx));
#endif
}

//
// LRN forward/backward functions
//

template <typename T>
inline void dnnmarkLRNForward(const Handle &handle,
                         RunMode mode, int idx,
                         const LRNDesc<T> &lrn_desc,
                         const LRNParam &lrn_param,
                         const void *alpha,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         const void *beta,
                         const DataTensor<T> &top_desc,
                         void *y,
                         void *workspace) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnLRNCrossChannelForward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             lrn_desc.Get(),
             lrn_param.mode_,
             alpha,
             bottom_desc.Get(), x,
             beta,
             top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenLRNForward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              lrn_desc.Get(),
              alpha,
              bottom_desc.Get(), x,
              beta,
              top_desc.Get(), y,
              true, workspace));
#endif
}

template <typename T>
inline void dnnmarkLRNBackward(const Handle &handle,
                         RunMode mode, int idx,
                         const LRNDesc<T> &lrn_desc,
                         const LRNParam &lrn_param,
                         const void *alpha,
                         const DataTensor<T> &top_desc,
                         const void *y,
                         const void *dy,
                         const void *beta,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         void *dx,
                         void *workspace) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnLRNCrossChannelBackward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             lrn_desc.Get(),
             lrn_param.mode_,
             alpha,
             top_desc.Get(), y,
             top_desc.Get(), dy,
             bottom_desc.Get(), x,
             beta,
             bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenLRNBackward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              lrn_desc.Get(),
              alpha,
              top_desc.Get(), y,
              top_desc.Get(), dy,
              bottom_desc.Get(), x,
              beta,
              bottom_desc.Get(), dx,
              workspace));
#endif
}

//
// Fully Connected forward/backward functions
//

//
// Softmax forward/backward functions
//

template <typename T>
inline void dnnmarkSoftmaxForward(const Handle &handle,
                         RunMode mode, int idx,
                         const SoftmaxParam &softmax_param,
                         const void *alpha,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         const void *beta,
                         const DataTensor<T> &top_desc,
                         void *y) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnSoftmaxForward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             softmax_param.algo_,
             softmax_param.mode_,
             alpha,
             bottom_desc.Get(), x,
             beta,
             top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenSoftmaxForward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              alpha,
              bottom_desc.Get(), x,
              beta,
              top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkSoftmaxBackward(const Handle &handle,
                         RunMode mode, int idx,
                         const SoftmaxParam &softmax_param,
                         const void *alpha,
                         const DataTensor<T> &top_desc,
                         const void *y,
                         const void *dy,
                         const void *beta,
                         const DataTensor<T> &bottom_desc,
                         void *dx) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnSoftmaxBackward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             softmax_param.algo_,
             softmax_param.mode_,
             alpha,
             top_desc.Get(), y,
             top_desc.Get(), dy,
             beta,
             bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenSoftmaxBackward(
             mode == COMPOSED ?
             handle.Get(idx) : handle.Get(),
             alpha,
             top_desc.Get(), y,
             top_desc.Get(), dy,
             beta,
             bottom_desc.Get(), dx));

#endif
}

//
// Batch Normalization forward/backward functions
//

template <typename T>
inline void dnnmarkBackNormalizationForwardTraining(
            const Handle &handle,
            RunMode mode, int idx,
            const BatchNormParam &bn_param,
            const void *alpha,
            const void *beta,
            const DataTensor<T> &bottom_desc,
            const void *x,
            const DataTensor<T> &top_desc,
            void *y,
            const DataTensor<T> &scale_bias_mean_var_desc,
            void *bn_scale,
            void *bn_bias,
            double exp_avg_factor,
            void *result_running_mean,
            void *result_running_var,
            double epsilon,
            void *result_save_mean,
            void *result_save_var) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             bn_param.mode_,
             alpha,
             beta,
             bottom_desc.Get(), x,
             top_desc.Get(), y,
             scale_bias_mean_var_desc.Get(),
             bn_scale, bn_bias,
             exp_avg_factor,
             result_running_mean, result_running_var,
             epsilon,
             result_save_mean, result_save_var));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenBatchNormalizationForwardTraining(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              bn_param.mode_,
              alpha,
              beta,
              bottom_desc.Get(), x,
              top_desc.Get(), y,
              scale_bias_mean_var_desc.Get(),
              bn_scale, bn_bias,
              exp_avg_factor,
              result_running_mean, result_running_var,
              epsilon,
              result_save_mean, result_save_var));
#endif
}

template <typename T>
inline void dnnmarkBackNormalizationBackward(
            const Handle &handle,
            RunMode mode, int idx,
            const BatchNormParam &bn_param,
            const void *alpha_data_diff,
            const void *beta_data_diff,
            const void *alpha_param_diff,
            const void *beta_param_diff,
            const DataTensor<T> &bottom_desc,
            const void *x,
            void *dx,
            const DataTensor<T> &top_desc,
            const void *dy,
            const DataTensor<T> &scale_bias_mean_var_desc,
            const void *bn_scale,
            void *result_bn_scale_diff,
            void *result_bn_bias_diff,
            double epsilon,
            const void *saved_mean,
            const void *saved_var) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnBatchNormalizationBackward(
             mode == COMPOSED ?
             handle.GetCudnn(idx) : handle.GetCudnn(),
             bn_param.mode_,
             alpha_data_diff,
             beta_data_diff,
             alpha_param_diff,
             beta_param_diff,
             bottom_desc.Get(), x, dx
             top_desc.Get(), dy,
             scale_bias_mean_var_desc.Get(),
             bn_scale,
             result_bn_scale_diff, result_bn_bias_diff,
             epsilon,
             saved_mean, saved_var));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenBatchNormalizationBackward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              bn_param.mode_,
              alpha_data_diff,
              beta_data_diff,
              alpha_param_diff,
              beta_param_diff,
              bottom_desc.Get(), x, dx,
              top_desc.Get(), dy,
              scale_bias_mean_var_desc.Get(),
              bn_scale,
              result_bn_scale_diff, result_bn_bias_diff,
              epsilon,
              saved_mean, saved_var));
#endif
}

//
// Bypass layer
//

template <typename T>
inline void dnnmarkBypassForward(const Handle &handle,
                         RunMode mode, int idx,
                         const BypassDesc<T> &bypass_desc,
                         const void *alpha,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         const void *beta,
                         const DataTensor<T> &top_desc,
                         void *y) {
#ifdef NVIDIA_CUDNN
  CUDA_CALL(cudaMemcpy(tops_[i]->Get(),
                       bottoms_[i]->Get(),
                       sizeof(T) * bypass_desc.Get().n_
                                 * bypass_desc.Get().c_
                                 * bypass_desc.Get().h_
                                 * bypass_desc.Get().w_,
                       cudaMemcpyDeviceToDevice
                       ));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationForward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              bypass_desc.Get(),
              alpha,
              bottom_desc.Get(), x,
              beta,
              top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkBypassBackward(const Handle &handle,
                         RunMode mode, int idx,
                         const BypassDesc<T> &bypass_desc,
                         const void *alpha,
                         const DataTensor<T> &top_desc,
                         const void *y,
                         const void *dy,
                         const void *beta,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         void *dx) {
#ifdef NVIDIA_CUDNN
  CUDA_CALL(cudaMemcpy(top_diffs_[i]->Get(),
                       bottom_diffs_[i]->Get(),
                       sizeof(T) * bypass_desc.Get().n_
                                 * bypass_desc.Get().c_
                                 * bypass_desc.Get().h_
                                 * bypass_desc.Get().w_,
                       cudaMemcpyDeviceToDevice
                       ));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationBackward(
              mode == COMPOSED ?
              handle.Get(idx) : handle.Get(),
              bypass_desc.Get(),
              alpha,
              top_desc.Get(), y,
              top_desc.Get(), dy,
              bottom_desc.Get(), x,
              beta,
              bottom_desc.Get(), dx));
#endif
}

//
// Dropout layer
//

template <typename T>
inline void dnnmarkDropoutForward(const Handle &handle,
                         RunMode mode, int idx,
                         const DropoutDesc<T> &dropout_desc,
                         const DataTensor<T> &bottom_desc,
                         const void *x,
                         const DataTensor<T> &top_desc,
                         void *y,
                         void *reserve_space, size_t reserve_space_size) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnDropoutForward(
          mode == COMPOSED ?
          handle.GetCudnn(idx) : handle.GetCudnn(),
          dropout_desc_,
          bottom_desc_.Get(), x->Get(),
          top_desc_.Get(), y->Get(),
          reserve_space,
          reserve_space_size
          ));
#endif
#ifdef AMD_MIOPEN
#endif
}

template <typename T>
inline void dnnmarkDropoutBackward(const Handle &handle,
                         RunMode mode, int idx,
                         const DropoutDesc<T> &dropout_desc,
                         const DataTensor<T> &top_desc,
                         const void *dy,
                         const DataTensor<T> &bottom_desc,
                         void *dx,
                         void *reserve_space, size_t reserve_space_size) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnDropoutBackward(
          mode == COMPOSED ?
          handle.GetCudnn(idx) : handle.GetCudnn(),
          dropout_desc_,
          top_desc_.Get(), dy->Get(),
          bottom_desc_.Get(), dx->Get(),
          reserve_space,
          reserve_space_size
          ));
#endif
#ifdef AMD_MIOPEN
#endif
}

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_WRAPPER_H_
