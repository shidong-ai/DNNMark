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

#include "dnn_utility.h"

namespace dnnmark {

Handle::Handle() {
  handles_ = new cudnnHandle_t[1];
  CUDNN_CALL(cudnnCreate(&handles_[0]));
  blas_handles_ = new cublasHandle_t[1];
  CUBLAS_CALL(cublasCreate(&blas_handles_[0]));
  num_handles_ = 1;
  num_blas_handles_ = 1;
}

Handle::Handle(int num) {
  handles_ = new cudnnHandle_t[num];
  for (int i = 0; i < num; i++)
    CUDNN_CALL(cudnnCreate(&handles_[i]));
  num_handles_ = num;

  blas_handles_ = new cublasHandle_t[num];
  for (int i = 0; i < num; i++)
    CUBLAS_CALL(cublasCreate(&blas_handles_[i]));
  num_blas_handles_ = num;
}

Handle::~Handle() {
  for (int i = 0; i < num_handles_; i++)
    CUDNN_CALL(cudnnDestroy(handles_[i]));
  delete []handles_;
  for (int i = 0; i < num_blas_handles_; i++)
    CUBLAS_CALL(cublasDestroy(blas_handles_[i]));
  delete []blas_handles_;
}

cudnnHandle_t Handle::getHandle() { return handles_[0]; }
cudnnHandle_t Handle::getHandle(int index) { return handles_[index]; }
cublasHandle_t Handle::getBlasHandle() { return blas_handles_[0]; }
cublasHandle_t Handle::getBlasHandle(int index) { return blas_handles_[index]; }

Descriptor::Descriptor()
: set_(false) {}

Descriptor::~Descriptor() {
  set_ = false;
}

bool Descriptor::isSet() { return set_; }

} // namespace dnnmark

