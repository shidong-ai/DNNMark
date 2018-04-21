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

#include "kernels.h"

namespace dnnmark {

template <>
__global__ void BCMSum(Complex *x, Complex *y, int q, int k) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[y_idx].x = 0;
  y[y_idx].y = 0;
  for (int i = 0; i < q; i++) {
    int x_idx = blockIdx.x * blockDim.x + i * k + threadIdx.x;
    y[y_idx].x += x[x_idx].x;
    y[y_idx].y += x[x_idx].y;
  }

}

}
