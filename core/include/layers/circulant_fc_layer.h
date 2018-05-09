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

#ifndef CORE_INCLUDE_LAYERS_CIRCULANT_FC_LAYER_H_ 
#define CORE_INCLUDE_LAYERS_CIRCULATN_FC_LAYER_H_

#include "dnn_layer.h"
#include "fft_utility.h"
#include "fft_wrapper.h"
#include "kernels.h"

namespace dnnmark {

template <typename T>
class CirculantFullyConnectedLayer : public Layer<T> {
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
  FullyConnectedParam fc_param_;

  // Weights related
  int num_rows_weights_;
  int num_cols_weights_;

  // Swap sum and IFFT
  bool swap_;

  // Use notation in the BCM paper
  int n_; // batch size
  int k_; // block size
  int p_; // number of blocks row-wise
  int q_; // number of blocks column-wise
  int fft_k_; // number of block size after fft

  // FFT plans
  FFTPlan w_plan_;
  FFTPlan x_plan_;
  FFTPlan y_plan_;
  FFTPlan ifft_plan_;
  FFTPlan ifft_forward_plan_;
  FFTPlan ifft_backward_weight_plan_;
  FFTPlan ifft_backward_data_plan_;

  // Layer weights
  Data<T> *weights_;
  int weights_chunk_id_;
  Data<T> *weights_diff_;
  int weights_diff_chunk_id_;

  // Intermediate memory
  Data<T> *fft_w_;
  int fft_w_chunk_id_;
  Data<T> *fft_x_;
  int fft_x_chunk_id_;
  Data<T> *fft_y_;
  int fft_y_chunk_id_;
  Data<T> *product_;
  int product_chunk_id_;
  Data<T> *sum_;
  int sum_chunk_id_;
  Data<T> *sum_y_;
  int sum_y_chunk_id_;
  Data<T> *sum_w_;
  int sum_w_chunk_id_;
  Data<T> *sum_x_;
  int sum_x_chunk_id_;

 public:
  CirculantFullyConnectedLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    fc_param_(),
    w_plan_(), x_plan_(), y_plan_(),
    ifft_plan_(),
    ifft_forward_plan_(),
    ifft_backward_weight_plan_(),
    ifft_backward_data_plan_() {
    Layer<T>::has_learnable_params_ = true;

    // Optimized implementation
    swap_ = true;
  }

  FullyConnectedParam *getFullyConnectedParam() { return &fc_param_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    // Set up fcing related data
    if (input_dim_.n_ != 0 && input_dim_.c_ != 0 &&
        input_dim_.h_ != 0 && input_dim_.w_ != 0) {

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
    num_rows_weights_ = fc_param_.output_num_;
    num_cols_weights_ = input_dim_.c_ *
                        input_dim_.h_ *
                        input_dim_.w_;

    k_ = fc_param_.block_size_;
    // Input dimension limitation
    if (num_rows_weights_ % k_ != 0 || num_cols_weights_ % k_) {
      LOG(FATAL) << "Input is not compatible with block size";
    }

    // The dimension of block circulant method
    n_ = output_dim_.n_;
    p_ = num_rows_weights_ / k_;
    q_ = num_cols_weights_ / k_;
    LOG(INFO) << "N: " << n_;
    LOG(INFO) << "P: " << p_;
    LOG(INFO) << "Q: " << q_;
    LOG(INFO) << "K: " << k_;

    // Set the plans
    w_plan_.Set(FFT_1D, k_, R2C, p_ * q_);
    x_plan_.Set(FFT_1D, k_, R2C, n_ * q_);
    y_plan_.Set(FFT_1D, k_, R2C, n_ * p_);
    if (swap_) {
      ifft_forward_plan_.Set(FFT_1D, k_, C2R, n_ * p_);
      ifft_backward_weight_plan_.Set(FFT_1D, k_, C2R, p_ * q_);
      ifft_backward_data_plan_.Set(FFT_1D, k_, C2R, n_ * q_);
    } else {
      ifft_plan_.Set(FFT_1D, k_, C2R, n_ * p_ * q_);
    }

    // Create weight data
    int weights_size = p_ * q_ * k_;
    weights_chunk_id_ = data_manager_->CreateData(weights_size);
    weights_ = data_manager_->GetData(weights_chunk_id_);
    weights_diff_chunk_id_ =
      data_manager_->CreateData(weights_size);
    weights_diff_ = data_manager_->GetData(weights_diff_chunk_id_);

    // Fill the weight data
    weights_->Filler();

    // Intermediate memory generation
    // Complex data requires doubling the memory
    fft_k_ = k_ / 2 + 1;

    int fft_w_size = p_ * q_ * fft_k_;
    fft_w_chunk_id_ = data_manager_->CreateData(fft_w_size * 2);
    fft_w_ = data_manager_->GetData(fft_w_chunk_id_);

    int fft_x_size = n_ * q_ * fft_k_;
    fft_x_chunk_id_ = data_manager_->CreateData(fft_x_size * 2);
    fft_x_ = data_manager_->GetData(fft_x_chunk_id_);

    int fft_y_size = n_ * p_ * fft_k_;
    fft_y_chunk_id_ = data_manager_->CreateData(fft_y_size * 2);
    fft_y_ = data_manager_->GetData(fft_y_chunk_id_);

    int product_size = n_ * p_ * q_ * fft_k_;
    product_chunk_id_ = data_manager_->CreateData(product_size * 2);
    product_ = data_manager_->GetData(product_chunk_id_);

    int sum_k;
    if (swap_) {
      sum_k = fft_k_ * 2;
      int sum_y_size = n_ * p_ * sum_k;
      sum_y_chunk_id_ = data_manager_->CreateData(sum_y_size);
      sum_y_ = data_manager_->GetData(sum_y_chunk_id_);

      int sum_w_size = p_ * q_ * sum_k;
      sum_w_chunk_id_ = 
        data_manager_->CreateData(sum_w_size);
      sum_w_ =
        data_manager_->GetData(sum_w_chunk_id_);

      int sum_x_size = n_ * q_ * sum_k;
      sum_x_chunk_id_ =
        data_manager_->CreateData(sum_x_size);
      sum_x_ = data_manager_->GetData(sum_x_chunk_id_);
    } else {
      sum_k = k_;
      int sum_size = n_ * p_ * q_ * sum_k;
      sum_chunk_id_ = data_manager_->CreateData(sum_size);
      sum_ = data_manager_->GetData(sum_chunk_id_);
    }

  }

  void ComputeOutputDim() {
    output_dim_.n_ = input_dim_.n_;
    output_dim_.c_ = fc_param_.output_num_;
    output_dim_.h_ = 1;
    output_dim_.w_ = 1;
  }

  void ForwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }


    // Fully connected forward computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BCMFcFwd");
    for (int i = 0; i < num_bottoms_; i++) {
      dnnmarkFFT(w_plan_, weights_->Get(), (Complex *)fft_w_->Get());
      dnnmarkFFT(x_plan_, bottoms_[i]->Get(), (Complex *)fft_x_->Get());
      CUDA_CALL(cudaDeviceSynchronize());

      if (fc_param_.optimized_) {
        BCMProductForwardOptimized((Complex *)fft_w_->Get(),
                   (Complex *)fft_x_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      } else {
        BCMProductForward((Complex *)fft_w_->Get(),
                   (Complex *)fft_x_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      }
      CUDA_CALL(cudaDeviceSynchronize());

      if (swap_) {
        // This is the implementation of after swap of IFFT and sum
        if (fc_param_.optimized_) {
          BCMSumForward((Complex *)product_->Get(),
                 (Complex *)sum_y_->Get(),
                 n_, p_, q_, fft_k_);
        } else {
          BCMSumForwardOptimized((Complex *)product_->Get(),
                 (Complex *)sum_y_->Get(),
                 n_, p_, q_, fft_k_);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        dnnmarkIFFT(ifft_forward_plan_, (Complex *)sum_y_->Get(), tops_[i]->Get());
      } else {
        // This is the original implementation
        dnnmarkIFFT(ifft_plan_, (Complex *)product_->Get(), sum_->Get());
        CUDA_CALL(cudaDeviceSynchronize());
        BCMSumForward(sum_->Get(), tops_[i]->Get(),
               n_, p_, q_, k_);
      }
      
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BCMFcFwd");

  }

  void BackwardPropagation() {
    if (p_dnnmark_->getRunMode() == STANDALONE ||
        !previous_layer_name_.compare("null")) {
      // Fill the top diff data
      for (int i = 0; i < num_tops_; i++) {
        top_diffs_[i]->Filler();
      }
      // Fill the bottom data
      for (int i = 0; i < num_bottoms_; i++) {
        bottoms_[i]->Filler();
      }
    }

    // Fully connected backward weights computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BCMFcBwdFilter");
    for (int i = 0; i < num_tops_; i++) {
      dnnmarkFFT(y_plan_, top_diffs_[i]->Get(), (Complex *)fft_y_->Get());
      // The FFT of x can be saved
      CUDA_CALL(cudaDeviceSynchronize());
      if (fc_param_.optimized_) {
        // Reuse the sum_y and sum_x here
        NPK2PNK((Complex *)fft_y_->Get(), (Complex *)sum_y_->Get(), n_, p_, q_, fft_k_);
        NQK2QNK((Complex *)fft_x_->Get(), (Complex *)sum_x_->Get(), n_, p_, q_, fft_k_);

        CUDA_CALL(cudaDeviceSynchronize());
        BCMProductBackwardWeightOptimized((Complex *)sum_y_->Get(),
                   (Complex *)sum_x_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      } else {
        BCMProductBackwardWeight((Complex *)sum_y_->Get(),
                   (Complex *)sum_x_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      }
      CUDA_CALL(cudaDeviceSynchronize());

      if (swap_) {
        // This is the implementation of after swap of IFFT and sum
        if (fc_param_.optimized_) {
          BCMSumBackwardWeightOptimized((Complex *)product_->Get(),
                 (Complex *)sum_w_->Get(),
                 n_, p_, q_, fft_k_);
        } else {
          BCMSumBackwardWeight((Complex *)product_->Get(),
                 (Complex *)sum_w_->Get(),
                 n_, p_, q_, fft_k_);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        dnnmarkIFFT(ifft_backward_weight_plan_,
          (Complex *)sum_w_->Get(), weights_diff_->Get());
      } else {
        // This is the original implementation
        dnnmarkIFFT(ifft_plan_, (Complex *)product_->Get(), sum_->Get());
        CUDA_CALL(cudaDeviceSynchronize());
        BCMSumBackwardWeight(sum_->Get(), tops_[i]->Get(),
               n_, p_, q_, k_);
      }
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "BCMFcBwdFilter");

    // Fully connected backward data computation
    ProfilerStart(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdData");
    for (int i = 0; i < num_tops_; i++) {
      // The FFT for dy can be saved
      // The FFT for w can be saved
      if (fc_param_.optimized_) {
        // Reuse the sum_w here
        PQK2QPK((Complex *)fft_w_->Get(), (Complex *)sum_w_->Get(), n_, p_, q_, fft_k_);

        CUDA_CALL(cudaDeviceSynchronize());
        BCMProductBackwardDataOptimized((Complex *)fft_y_->Get(),
                   (Complex *)sum_w_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      } else {
        BCMProductBackwardData((Complex *)fft_y_->Get(),
                   (Complex *)fft_w_->Get(),
                   (Complex *)product_->Get(),
                   n_, p_, q_, fft_k_);
      }
      CUDA_CALL(cudaDeviceSynchronize());

      if (swap_) {
        // This is the implementation of after swap of IFFT and sum
        if (fc_param_.optimized_) {
          BCMSumBackwardDataOptimized((Complex *)product_->Get(),
                 (Complex *)sum_x_->Get(),
                 n_, p_, q_, fft_k_);
        } else {
          BCMSumBackwardData((Complex *)product_->Get(),
                 (Complex *)sum_x_->Get(),
                 n_, p_, q_, fft_k_);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        dnnmarkIFFT(ifft_backward_data_plan_,
          (Complex *)sum_x_->Get(), bottom_diffs_[i]->Get());
      } else {
        // This is the original implementation
        dnnmarkIFFT(ifft_plan_, (Complex *)product_->Get(), sum_->Get());
        CUDA_CALL(cudaDeviceSynchronize());
        BCMSumBackwardData(sum_->Get(), tops_[i]->Get(),
               n_, p_, q_, k_);
      }
    }
    ProfilerStop(*(p_dnnmark_->GetHandle()), p_dnnmark_->getRunMode(),
                  layer_id_, p_dnnmark_->GetTimer(), "FcBwdData");
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_CIRCULANT_LAYERS_FC_LAYER_H_
