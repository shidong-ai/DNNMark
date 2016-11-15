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

#ifndef CORE_INCLUDE_DNN_LAYER_H_ 
#define CORE_INCLUDE_DNN_LAYER_H_

#include <vector>
#include <glog/logging.h>
#include "cudnn.h"
#include "common.h"
#include "dnn_param.h"
#include "dnn_utility.h"
#include "data_manager.h"

namespace dnnmark {

template <typename T>
class Layer {
 protected:
  // Handle
  Handle *p_handle_;

  bool has_learnable_params_;
  LayerType type_;
  int layer_id_;
  std::string layer_name_;
  std::string previous_layer_name_;
  DataDim input_dim_;
  DataTensor<T> bottom_desc_;
  DataManager<T> *data_manager_;  

  int num_bottoms_;
  // Layer bottom data
  std::vector<Data<T> *> bottoms_;
  std::vector<int> bottom_chunk_ids_;
  std::vector<Data<T> *> bottom_diffs_;
  std::vector<int> bottom_diff_chunk_ids_;
 public:
  Layer(Handle *p_handle)
  : p_handle_(p_handle),
    layer_id_(0), has_learnable_params_(false),
    input_dim_(), bottom_desc_(),
    num_bottoms_(1) {
    data_manager_ = DataManager<T>::GetInstance();
  }
  ~Layer() {
    data_manager_->DataManager<T>::~DataManager(); 
  }
  DataDim *getInputDim() { return &input_dim_; }
  void setLayerName(const char *layer_name) {
    layer_name_.assign(layer_name);
  }
  void setPrevLayerName(const char *previous_layer_name) {
    previous_layer_name_.assign(previous_layer_name);
  }
  void setLayerId(int layer_id) { layer_id_ = layer_id; }
  int getLayerId() { return layer_id_; }
  void setLayerType(LayerType type) { type_ = type; }
  LayerType getLayerType() { return type_; }
  virtual void Setup() {
    if (input_dim_.n_ != 0 && input_dim_.c_ != 0 &&
        input_dim_.h_ != 0 && input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Set bottom tensor
      bottom_desc_.Set(input_dim_.n_,
                       input_dim_.c_,
                       input_dim_.h_,
                       input_dim_.w_);

      // Prepare bottom data
      int bottom_size = input_dim_.n_ *
                        input_dim_.c_ *
                        input_dim_.h_ *
                        input_dim_.w_;
      for (int i = 0; i < num_bottoms_; i++) {
        bottom_chunk_ids_.push_back(
          data_manager_->CreateData(bottom_size));
        bottoms_.push_back(
          data_manager_->GetData(bottom_chunk_ids_[i]));
        bottom_diff_chunk_ids_.push_back(
          data_manager_->CreateData(bottom_size));
        bottom_diffs_.push_back(
          data_manager_->GetData(bottom_diff_chunk_ids_[i]));
      }
    }
  }

  virtual void ForwardPropagation() {}
  virtual void BackwardPropagation() {}

};

template <typename T>
class ConvolutionLayer : public Layer<T> {
 private:
  ConvolutionParam conv_param_;

  // Convolution specific descriptor
  ConvolutionDesc<T> desc_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

  // Layer weights
  Data<T> *weights_;
  int weights_chunk_id_;
  Data<T> *weights_diff_;
  int weights_diff_chunk_id_;

  // Algorithm specific parameters
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  size_t fwd_workspace_size_;
  size_t bwd_workspace_data_size_;
  size_t bwd_workspace_filter_size_;
  void *workspace_;
 public:
  ConvolutionLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), conv_param_(), desc_() {
    Layer<T>::has_learnable_params_ = true;
    num_tops_ = Layer<T>::num_bottoms_;
  }

  ConvolutionParam *getConvParam() { return &conv_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set convolution related descriptors
    desc_.Set(conv_param_, Layer<T>::input_dim_.c_);

    // Set up convolution related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }

      // Only one set of weights is considered
      int weights_size = conv_param_.output_num_ *
                         Layer<T>::input_dim_.c_ *
                         conv_param_.kernel_size_h_ *
                         conv_param_.kernel_size_w_;
      weights_chunk_id_ = Layer<T>::data_manager_->CreateData(weights_size);
      weights_ = Layer<T>::data_manager_->GetData(weights_chunk_id_);
      weights_diff_chunk_id_ =
        Layer<T>::data_manager_->CreateData(weights_size);
      weights_diff_ = Layer<T>::data_manager_->GetData(weights_diff_chunk_id_);
    }

    // Set up convolution forward algorithm related parameters
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        Layer<T>::p_handle_->getHandle(),
        Layer<T>::bottom_desc_.Get(),
        desc_.GetFilter(),
        desc_.GetConv(),
        top_desc_.Get(),
        conv_param_.conv_fwd_pref_,
        -1,
        &fwd_algo_));
  
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        Layer<T>::p_handle_->getHandle(),
        Layer<T>::bottom_desc_.Get(),
        desc_.GetFilter(),
        desc_.GetConv(),
        top_desc_.Get(),
        fwd_algo_,
        &fwd_workspace_size_));
    CUDA_CALL(cudaMalloc(&workspace_, fwd_workspace_size_));

  }

  void ComputeOutputDim() {
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = conv_param_.output_num_;
    output_dim_.h_ = (Layer<T>::input_dim_.h_ +
      2 * conv_param_.pad_h_ - conv_param_.kernel_size_h_) /
      conv_param_.stride_u_ + 1;
    output_dim_.w_ = (Layer<T>::input_dim_.w_ +
      2 * conv_param_.pad_w_ - conv_param_.kernel_size_w_) /
      conv_param_.stride_v_ + 1;
  }

  void ForwardPropagation() {
    // Fill the data
    weights_->Filler();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // Convolution forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      CUDNN_CALL(cudnnConvolutionForward(
                Layer<T>::p_handle_->getHandle(),
                DataType<T>::one,
                Layer<T>::bottom_desc_.Get(), Layer<T>::bottoms_[i]->Get(),
                desc_.GetFilter(), weights_->Get(),
                desc_.GetConv(),
                fwd_algo_, workspace_, fwd_workspace_size_,
                DataType<T>::zero,
                top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

    // TODO: evaluate the necessity of freeing memory here
    // Free the workspace
    CUDA_CALL(cudaFree(workspace_));
  }
  void BackwardPropagation() {
  }

};

template <typename T>
class PoolingLayer : public Layer<T> {
 private:
  PoolingParam pool_param_;

  // Pooling specific descriptor
  PoolingDesc<T> desc_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

 public:
  PoolingLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), pool_param_(), desc_() {
    Layer<T>::has_learnable_params_ = true;
    num_tops_ = Layer<T>::num_bottoms_;
  }

  PoolingParam *getPoolParam() { return &pool_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set pooling related descriptors
    desc_.Set(pool_param_);

    // Set up pooling related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }
    }
  }

  void ComputeOutputDim() {
    // Courtesy of Caffe
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = Layer<T>::input_dim_.c_;
    output_dim_.h_ = static_cast<int>(ceil(static_cast<float>(
      Layer<T>::input_dim_.h_ + 2 * pool_param_.pad_h_ - 
      pool_param_.kernel_size_h_) / pool_param_.stride_h_)) + 1;
    output_dim_.w_ = static_cast<int>(ceil(static_cast<float>(
      Layer<T>::input_dim_.w_ + 2 * pool_param_.pad_w_ - 
      pool_param_.kernel_size_w_) / pool_param_.stride_w_)) + 1;
    if (pool_param_.pad_h_ > 0 && pool_param_.pad_w_ > 0) {
      if ((output_dim_.h_ - 1) * pool_param_.stride_h_ >= 
          Layer<T>::input_dim_.h_ + pool_param_.pad_h_) {
        --output_dim_.h_;
      }
      if ((output_dim_.w_ - 1) * pool_param_.stride_w_ >= 
          Layer<T>::input_dim_.w_ + pool_param_.pad_w_) {
        --output_dim_.w_;
      }
    }
  }

  void ForwardPropagation() {
    // Fill the data
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // pooling forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      CUDNN_CALL(cudnnPoolingForward(
             Layer<T>::p_handle_->getHandle(),
             desc_.Get(),
             DataType<T>::one, 
             Layer<T>::bottom_desc_.Get(), Layer<T>::bottoms_[i]->Get(),
             DataType<T>::zero,
             top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

  }
  void BackwardPropagation() {
  }

};

template <typename T>
class LRNLayer : public Layer<T> {
 private:
  LRNParam lrn_param_;

  // LRN specific descriptor
  LRNDesc<T> desc_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

 public:
  LRNLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), lrn_param_(), desc_() {
    num_tops_ = Layer<T>::num_bottoms_;
  }

  LRNParam *getLRNParam() { return &lrn_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set lrning related descriptors
    desc_.Set(lrn_param_);

    // Set up lrning related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }
    }
  }

  void ComputeOutputDim() {
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = Layer<T>::input_dim_.c_;
    output_dim_.h_ = Layer<T>::input_dim_.h_;
    output_dim_.w_ = Layer<T>::input_dim_.w_;
  }

  void ForwardPropagation() {
    // Fill the data
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // lrning forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      CUDNN_CALL(cudnnLRNCrossChannelForward(
             Layer<T>::p_handle_->getHandle(),
             desc_.Get(),
             lrn_param_.mode_,
             DataType<T>::one, 
             Layer<T>::bottom_desc_.Get(), Layer<T>::bottoms_[i]->Get(),
             DataType<T>::zero,
             top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

  }
  void BackwardPropagation() {
  }

};

template <typename T>
class ActivationLayer : public Layer<T> {
 private:
  ActivationParam activation_param_;

  // Activation specific descriptor
  ActivationDesc<T> desc_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

 public:
  ActivationLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), activation_param_(), desc_() {
    num_tops_ = Layer<T>::num_bottoms_;
  }

  ActivationParam *getActivationParam() { return &activation_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set activationing related descriptors
    desc_.Set(activation_param_);

    // Set up activationing related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }
    }
  }

  void ComputeOutputDim() {
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = Layer<T>::input_dim_.c_;
    output_dim_.h_ = Layer<T>::input_dim_.h_;
    output_dim_.w_ = Layer<T>::input_dim_.w_;
  }

  void ForwardPropagation() {
    // Fill the data
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // activationing forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      CUDNN_CALL(cudnnActivationForward(
             Layer<T>::p_handle_->getHandle(),
             desc_.Get(),
             DataType<T>::one, 
             Layer<T>::bottom_desc_.Get(), Layer<T>::bottoms_[i]->Get(),
             DataType<T>::zero,
             top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

  }
  void BackwardPropagation() {
  }

};

template <typename T>
class FullyConnectedLayer : public Layer<T> {
 private:
  FullyConnectedParam fc_param_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

  // Weights demension
  int num_rows_weights_;
  int num_cols_weights_;
  T scale_alpha_;
  T scale_beta_;

  // Layer weights
  Data<T> *weights_;
  int weights_chunk_id_;
  Data<T> *weights_diff_;
  int weights_diff_chunk_id_;

 public:
  FullyConnectedLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), fc_param_() {
    num_tops_ = Layer<T>::num_bottoms_;
  }

  FullyConnectedParam *getFullyConnectedParam() { return &fc_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set up fcing related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }

      // Only one set of weights is considered
      num_rows_weights_ = Layer<T>::input_dim_.c_ *
                             Layer<T>::input_dim_.h_ *
                             Layer<T>::input_dim_.w_;
      num_cols_weights_ = fc_param_.output_num_;
      int weights_size = num_rows_weights_ * num_cols_weights_;
      weights_chunk_id_ = Layer<T>::data_manager_->CreateData(weights_size);
      weights_ = Layer<T>::data_manager_->GetData(weights_chunk_id_);
      weights_diff_chunk_id_ =
        Layer<T>::data_manager_->CreateData(weights_size);
      weights_diff_ = Layer<T>::data_manager_->GetData(weights_diff_chunk_id_);

      scale_alpha_ = (T)1.0;
      scale_beta_ = (T)0.0;
    }
  }

  void ComputeOutputDim() {
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = fc_param_.output_num_;
    output_dim_.h_ = 1;
    output_dim_.w_ = 1;
  }

  void ForwardPropagation() {
    // Fill the data
    weights_->Filler();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // Prepare CuBLAS parameters
    int M = fc_param_.output_num_;
    int N = Layer<T>::input_dim_.n_;;
    int K = num_rows_weights_;
    int lda = K;
    int ldb = K;
    int ldc = M;

    // Fully connected forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      // Y = T(W) * X                                                               
      DNNMarkGEMM(Layer<T>::p_handle_->getBlasHandle(),
                  CUBLAS_OP_T, CUBLAS_OP_N,
                  M, N, K,
                  &scale_alpha_,
                  weights_->Get(), lda,
                  Layer<T>::bottoms_[i]->Get(), ldb,
                  &scale_beta_,
                  tops_[i]->Get(), ldc);
    }
    cudaProfilerStop();

  }

  void BackwardPropagation() {
  }

};

template <typename T>
class SoftmaxLayer : public Layer<T> {
 private:
  SoftmaxParam softmax_param_;

  // Layer-specific output
  int num_tops_;
  DataDim output_dim_;
  DataTensor<T> top_desc_;
  std::vector<Data<T> *> tops_;
  std::vector<int> top_chunk_ids_;
  std::vector<Data<T> *> top_diffs_;
  std::vector<int> top_diff_chunk_ids_;

 public:
  SoftmaxLayer(Handle *p_handle)
  : Layer<T>(p_handle),
    top_desc_(), output_dim_(), softmax_param_() {
    num_tops_ = Layer<T>::num_bottoms_;
  }

  SoftmaxParam *getSoftmaxParam() { return &softmax_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set up softmaxing related data
    if (Layer<T>::input_dim_.n_ != 0 && Layer<T>::input_dim_.c_ != 0 &&
        Layer<T>::input_dim_.h_ != 0 && Layer<T>::input_dim_.w_ != 0) {
      //
      // Standalone mode
      //

      // Compute dimension of output data
      ComputeOutputDim();

      // Set top tensor
      top_desc_.Set(output_dim_.n_,
                    output_dim_.c_,
                    output_dim_.h_,
                    output_dim_.w_);

      // Prepare top data
      int top_size = output_dim_.n_ *
                     output_dim_.c_ *
                     output_dim_.h_ *
                     output_dim_.w_;
      for (int i = 0; i < num_tops_; i++) {
        top_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        tops_.push_back(
          Layer<T>::data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          Layer<T>::data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          Layer<T>::data_manager_->GetData(top_diff_chunk_ids_[i]));
      }

    }
  }

  void ComputeOutputDim() {
    output_dim_.n_ = Layer<T>::input_dim_.n_;
    output_dim_.c_ = Layer<T>::input_dim_.c_;
    output_dim_.h_ = Layer<T>::input_dim_.h_;
    output_dim_.w_ = Layer<T>::input_dim_.w_;
  }

  void ForwardPropagation() {
    // Fill the data
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      Layer<T>::bottoms_[i]->Filler();
    }

    // Fully connected forward computation
    cudaProfilerStart();
    for (int i = 0; i < Layer<T>::num_bottoms_; i++) {
      CUDNN_CALL(cudnnSoftmaxForward(
              Layer<T>::p_handle_->getHandle(),
              softmax_param_.algo_,
              softmax_param_.mode_,
              DataType<T>::one,                                                  
              Layer<T>::bottom_desc_.Get(), Layer<T>::bottoms_[i]->Get(),
              DataType<T>::zero,
              top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

  }

  void BackwardPropagation() {
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_LAYER_H_
