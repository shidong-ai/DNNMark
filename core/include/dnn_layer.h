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

#include "cudnn.h"
#include "dnn_param.h"

namespace dnnmark {

// Layer type
enum LayerType {
  DATA = 0,
  CONVOLUTION,
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
 public:
  virtual void Setup() {}
  virtual void ForwardPropagation() {}
  virtual void BackwardPropagation() {}
  int getLayerId() { return layer_id_; }
  LayerType getLayerType() { return layer_type_; }
};

template <typename T>
class ConvolutionLayer : public Layer<T> {
 private:
  ConvolutionParam conv_param_;
 public:
  ConvolutionLayer()
  : conv_param_() {}
  void Setup() {
  }
  void ForwardPropagation() {
  }
  void BackwardPropagation(){ 
  }

};



} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_LAYER_H_
