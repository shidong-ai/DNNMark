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

#ifndef LAYERS_INCLUDE_CONV_LAYER_H_ 
#define LAYERS_INCLUDE_CONV_LAYER_H_

template <typename T>
class ConvolutionLayer : public Layer<T> {

  // using declaration for calling member from base class
  using Layer<T>::p_dnnmark_;
  using Layer<T>::layer_id_;
  using Layer<T>::previous_layer_name_;
  using Layer<T>::input_dim_;
  using Layer<T>::output_dim_;
  using Layer<T>::bottom_desc_;
  using Layer<T>::top_desc_;
  using Layer<T>::data_manager_;  

  using Layer<T>::num_bottoms_;
  using Layer<T>::bottoms_;
  using Layer<T>::bottom_chunk_ids_;
  using Layer<T>::bottom_diffs_;
  using Layer<T>::bottom_diff_chunk_ids_;

  using Layer<T>::num_tops_;
  using Layer<T>::tops_;
  using Layer<T>::top_chunk_ids_;
  using Layer<T>::top_diffs_;
  using Layer<T>::top_diff_chunk_ids_; 

 private:
  ConvolutionParam conv_param_;

  // Convolution specific descriptor
  ConvolutionDesc<T> desc_;

  // Layer weights
  Data<T> *weights_;
  int weights_chunk_id_;
  Data<T> *weights_diff_;
  int weights_diff_chunk_id_;

  // Algorithm specific parameters
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  size_t fwd_workspace_size_;
  size_t bwd_data_workspace_size_;
  size_t bwd_filter_workspace_size_;
  void *fwd_workspace_;
  void *bwd_data_workspace_;
  void *bwd_filter_workspace_;
 public:
  ConvolutionLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    conv_param_(), desc_() {
    Layer<T>::has_learnable_params_ = true;
  }

  ConvolutionParam *getConvParam() { return &conv_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set convolution related descriptors
    desc_.Set(conv_param_, input_dim_.c_);

    // Set up convolution related data
    if (input_dim_.n_ != 0 && input_dim_.c_ != 0 &&
        input_dim_.h_ != 0 && input_dim_.w_ != 0) {
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
          data_manager_->CreateData(top_size));
        tops_.push_back(
          data_manager_->GetData(top_chunk_ids_[i]));
        top_diff_chunk_ids_.push_back(
          data_manager_->CreateData(top_size));
        top_diffs_.push_back(
          data_manager_->GetData(top_diff_chunk_ids_[i]));
      }

    }

    // Only one set of weights is considered
    int weights_size = conv_param_.output_num_ *
                       input_dim_.c_ *
                       conv_param_.kernel_size_h_ *
                       conv_param_.kernel_size_w_;
    weights_chunk_id_ = data_manager_->CreateData(weights_size);
    weights_ = data_manager_->GetData(weights_chunk_id_);
    weights_diff_chunk_id_ =
      data_manager_->CreateData(weights_size);
    weights_diff_ = data_manager_->GetData(weights_diff_chunk_id_);

    // Fill the weight data
    weights_->Filler();

    // Set up convolution forward algorithm related parameters
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        bottom_desc_.Get(),
        desc_.GetFilter(),
        desc_.GetConv(),
        top_desc_.Get(),
        conv_param_.conv_fwd_pref_,
        -1,
        &fwd_algo_));
  
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        bottom_desc_.Get(),
        desc_.GetFilter(),
        desc_.GetConv(),
        top_desc_.Get(),
        fwd_algo_,
        &fwd_workspace_size_));

    CUDA_CALL(cudaMalloc(&fwd_workspace_, fwd_workspace_size_));

    // Set up convolution backward algorithm related parameters
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        bottom_desc_.Get(),
        top_desc_.Get(),
        desc_.GetConv(),
        desc_.GetFilter(),
        conv_param_.conv_bwd_filter_pref_,
        -1,
        &bwd_filter_algo_));
  
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        bottom_desc_.Get(),
        top_desc_.Get(),
        desc_.GetConv(),
        desc_.GetFilter(),
        bwd_filter_algo_,
        &bwd_filter_workspace_size_));

    CUDA_CALL(cudaMalloc(&bwd_filter_workspace_, bwd_filter_workspace_size_));

    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        desc_.GetFilter(),
        top_desc_.Get(),
        desc_.GetConv(),
        bottom_desc_.Get(),
        conv_param_.conv_bwd_data_pref_,
        -1,
        &bwd_data_algo_));
  
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
        p_dnnmark_->getRunMode() == COMPOSED ?
        p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
        p_dnnmark_->GetHandle()->GetCudnn(),
        desc_.GetFilter(),
        top_desc_.Get(),
        desc_.GetConv(),
        bottom_desc_.Get(),
        bwd_data_algo_,
        &bwd_data_workspace_size_));

    CUDA_CALL(cudaMalloc(&bwd_data_workspace_, bwd_data_workspace_size_));
  }

  void ComputeOutputDim() {
    output_dim_.n_ = input_dim_.n_;
    output_dim_.c_ = conv_param_.output_num_;
    output_dim_.h_ = (input_dim_.h_ +
      2 * conv_param_.pad_h_ - conv_param_.kernel_size_h_) /
      conv_param_.stride_u_ + 1;
    output_dim_.w_ = (input_dim_.w_ +
      2 * conv_param_.pad_w_ - conv_param_.kernel_size_w_) /
      conv_param_.stride_v_ + 1;
  }

  void ForwardPropagation() {
    // Fill the bottom data
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }
    // Convolution forward computation
    cudaProfilerStart();
    for (int i = 0; i < num_bottoms_; i++) {
      CUDNN_CALL(cudnnConvolutionForward(
                p_dnnmark_->getRunMode() == COMPOSED ?
                p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
                p_dnnmark_->GetHandle()->GetCudnn(),
                DataType<T>::one,
                bottom_desc_.Get(), bottoms_[i]->Get(),
                desc_.GetFilter(), weights_->Get(),
                desc_.GetConv(),
                fwd_algo_, fwd_workspace_, fwd_workspace_size_,
                DataType<T>::zero,
                top_desc_.Get(), tops_[i]->Get()));
    }
    cudaProfilerStop();

    // TODO: evaluate the necessity of freeing memory here
    // Free the workspace
    CUDA_CALL(cudaFree(fwd_workspace_));
  }
  void BackwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the top data and top diff data
      for (int i = 0; i < num_tops_; i++) {
        tops_[i]->Filler();
        top_diffs_[i]->Filler();
      }
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Convolution forward computation
    cudaProfilerStart();
    for (int i = 0; i < num_tops_; i++) {
      CUDNN_CALL(cudnnConvolutionBackwardFilter(
                p_dnnmark_->getRunMode() == COMPOSED ?
                p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
                p_dnnmark_->GetHandle()->GetCudnn(),
                DataType<T>::one,
                bottom_desc_.Get(), bottoms_[i]->Get(),
                top_desc_.Get(), top_diffs_[i]->Get(),
                desc_.GetConv(),
                bwd_filter_algo_,
                bwd_filter_workspace_, bwd_filter_workspace_size_,
                DataType<T>::zero,
                desc_.GetFilter(), weights_diff_->Get()));
      CUDNN_CALL(cudnnConvolutionBackwardData(
                p_dnnmark_->getRunMode() == COMPOSED ?
                p_dnnmark_->GetHandle()->GetCudnn(layer_id_):
                p_dnnmark_->GetHandle()->GetCudnn(),
                DataType<T>::one,
                desc_.GetFilter(), weights_->Get(),
                top_desc_.Get(), top_diffs_[i]->Get(),
                desc_.GetConv(),
                bwd_data_algo_,
                bwd_data_workspace_, bwd_data_workspace_size_,
                DataType<T>::zero,
                bottom_desc_.Get(), bottoms_[i]->Get()));
    }
    cudaProfilerStop();

    // TODO: evaluate the necessity of freeing memory here
    // Free the workspace
    CUDA_CALL(cudaFree(bwd_data_workspace_));
    CUDA_CALL(cudaFree(bwd_filter_workspace_));
  }

};

#endif // LAYERS_INCLUDE_CONV_LAYER_H_
