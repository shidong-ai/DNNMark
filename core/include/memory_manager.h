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

#ifndef CORE_INCLUDE_MEMORY_MANAGER_H_
#define CORE_INCLUDE_MEMORY_MANAGER_H_

#include <vector>
#include "cudnn.h"

namespace dnnmark {

template <typename T>
class TensorManager {
 private:
  map<int, cudnnTensorDescriptor_t> bottom_tensors_;
  map<int, cudnnTensorDescriptor_t> top_tensors_;

  // For convolution layer only
  map<int, cudnnConvolutionDescriptor_t> conv_tensors_;

  int number_of_layers_;
 public:
  void createTensor(int layer_id) {
  }
  void setTensor(int n, int c, int h, int w) {
  }
  void setTensor(int n, int c, int h, int w,
                 int stride_n, int stride_c, int stride_h, int stride_w) {
  }
}

} // namespace dnnmark

#endif // CORE_INCLUDE_MEMORY_MANAGER_H_
