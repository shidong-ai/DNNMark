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
  size_ = 1;
}

Handle::Handle(int num) {
  handles_ = new cudnnHandle_t[num];
  for (int i = 0; i < num; i++)
    CUDNN_CALL(cudnnCreate(&handles_[i]));
  size_ = num;
}

Handle::~Handle() {
  for (int i = 0; i < size_; i++)
    CUDNN_CALL(cudnnDestroy(handles[i]));
  delete []handles;
}

cudnnHandle_t Handle::getHandle() { return handles[0]; }
cudnnHandle_t Handle::getHandle(int index) { return handles[index]; }

Descriptor::Descriptor()
: set_(false) {}

Descriptor::~Descriptor() {
  set_ = false;
}

bool Descriptor::isSet() { return set_; }

} // namespace dnnmark

