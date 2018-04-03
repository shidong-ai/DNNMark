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

#ifndef CORE_INCLUDE_LAYERS_LOSS_LAYER_H_ 
#define CORE_INCLUDE_LAYERS_LOSS_LAYER_H_

#include "dnn_layer.h"

namespace dnnmark {

template <typename T>
class LossLayer : public Layer<T> {
  // using declaration for calling member from base class
  using Layer<T>::p_dnnmark_;
  using Layer<T>::layer_id_;
  using Layer<T>::previous_layer_name_;
  using Layer<T>::input_dim_;
  using Layer<T>::bottom_desc_;
  using Layer<T>::data_manager_;  

  using Layer<T>::num_bottoms_;
  using Layer<T>::bottoms_;
  using Layer<T>::bottom_chunk_ids_;
  using Layer<T>::bottom_diffs_;
  using Layer<T>::bottom_diff_chunk_ids_;

 private:
  LossType type_;
  T loss_;
  vector<int> label_;
  T normalize_factor;

 public:
  LossLayer(DNNMark<T> *p_dnnmark)
  : Layer<T>(p_dnnmark),
    type_(CROSS_ENTROPY) {
  }

  void setLossType(LossType type) { type_ = type; }
  T getLoss() { return loss_; }

  void Setup() {
    // Set up indispensable stuff here
    Layer<T>::Setup();

    if (p_dnnmark_->getRunMode() == STANDALONE) {
      LOG(FATAL) << "Loss layer only works in composed mode";
    } else if (input_dim_.n_ == 0 || input_dim_.c_ == 0 ||
               input_dim_.h_ == 0 || input_dim_.w_ == 0) {
      LOG(FATAL) << "Wrong dimension of last layer's output";
    } else {
      normalize_factor_ = 1.0 / T(input_dim_.n_);
    }

    loss_ = 999.0;
    label_.resize(input_dim_.n_);
    data_manager_->ReadLabel(label_.data());

  }

  void ForwardPropagation() {
    //TODO
  }

  void BackwardPropagation() {
    //TODO
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_LAYERS_LOSS_LAYER_H_
