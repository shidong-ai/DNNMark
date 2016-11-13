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
#include "cudnn.h"
#include "dnn_param.h"
#include "dnn_utility.h"
#include "data_manager.h"

namespace dnnmark {

// Layer type
enum LayerType {
  CONVOLUTION = 0,
  POOLING,
  ACTIVIATION,
  LRN,
  FC,
  SOFTMAX
};

template <typename T>
class Layer {
 protected:
  bool has_learnable_params_;
  LayerType type_;
  int layer_id_;
  std::string layer_name_;
  std::string previous_layer_name_;
  DataDim data_dim_;
  DataTensor<T> bottom_desc_;
  DataTensor<T> top_desc_;
  DataManager<T> *data_manager_;  
  std::vector<Data<T> *> bottoms_;
  std::vector<Data<T> *> bottom_diffs_;
  std::vector<Data<T> *> tops_;
  std::vector<Data<T> *> top_diffs_;
 public:
  Layer()
  : layer_id_(0), has_learnable_params_(false),
    data_dim_(), bottom_desc_(), top_desc_() {
    data_manager_ = DataManager<T>::GetInstance();
  }
  virtual void Setup() {}
  virtual void ForwardPropagation() {}
  virtual void BackwardPropagation() {}
  DataDim *getDataDim() { return &data_dim_; }
  void setLayerName(const char *layer_name) {
    layer_name_.assign(layer_name);
  }
  void setPrevLayerName(const char *previous_layer_name) {
    previous_layer_name_.assign(previous_layer_name);
  }
  void setLayerId(int layer_id) { layer_id_ = layer_id; }
  int getLayerId() { return layer_id_; }
  void setLayerType(LayerType type) { type = type_; }
  LayerType getLayerType() { return type_; }
};

template <typename T>
class ConvolutionLayer : public Layer<T> {
 private:
  ConvolutionParam conv_param_;
 public:
  ConvolutionLayer()
  : conv_param_() {}
  void Setup() {
    if (Layer<T>::data_dim_.n_ != 0 && Layer<T>::data_dim_.c_ != 0 &&
        Layer<T>::data_dim_.h_ != 0 && Layer<T>::data_dim_.w_ != 0) {
      // Standalone mode
      ComputeOutputDim();
      Layer<T>::bottom_desc_.Set(Layer<T>::data_dim_.n_,
                                 Layer<T>::data_dim_.c_,
                                 Layer<T>::data_dim_.h_,
                                 Layer<T>::data_dim_.w_);
      Layer<T>::top_desc_.Set(Layer<T>::data_dim_.output_n_,
                              Layer<T>::data_dim_.output_c_,
                              Layer<T>::data_dim_.output_h_,
                              Layer<T>::data_dim_.output_w_);
    }
  }

  void ComputeOutputDim() {
    Layer<T>::data_dim_.output_n_ = Layer<T>::data_dim_.n_;
    Layer<T>::data_dim_.output_c_ = conv_param_.output_num_;
    Layer<T>::data_dim_.output_h_ = (Layer<T>::data_dim_.h_ +
      2 * conv_param_.pad_h_ - conv_param_.kernel_size_h_) /
      conv_param_.stride_u_ + 1;
    Layer<T>::data_dim_.output_w_ = (Layer<T>::data_dim_.w_ +
      2 * conv_param_.pad_w_ - conv_param_.kernel_size_w_) /
      conv_param_.stride_v_ + 1;
  }
  void ForwardPropagation() {
  }
  void BackwardPropagation() {
  }
  ConvolutionParam *getConvParam() { return &conv_param_; }

};



} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_LAYER_H_
