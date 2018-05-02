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

__global__ void BCMSumForwardKernel(Real *x, Real *y, int q, int k) {
  // Dimension of X is n * p * q * k
  // Dimension of Y is n * q * k
  // Sum over q
  int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[y_idx] = 0;
  for (int i = 0; i < q; i++) {
    int x_idx = blockIdx.x * blockDim.x + i * k + threadIdx.x;
    y[y_idx] += x[x_idx];
  }

}

void BCMSumForward(Real *x, Real *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * p, 1, 1);
  BCMSumForwardKernel<<<grid_dim, block_dim>>>(x, y, q, k);
}

__global__ void BCMSumForwardKernel(Complex *x, Complex *y, int q, int k) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  // Sum over q
  int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[y_idx].x = 0;
  y[y_idx].y = 0;
  for (int i = 0; i < q; i++) {
    int x_idx = blockIdx.x * blockDim.x + i * k + threadIdx.x;
    y[y_idx].x += x[x_idx].x;
    y[y_idx].y += x[x_idx].y;
  }

}

void BCMSumForward(Complex *x, Complex *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * p, 1, 1);
  BCMSumForwardKernel<<<grid_dim, block_dim>>>(x, y, q, k);
}

__global__ void BCMSumBackwardWeightKernel(Real *x, Real *y, int n) {
  // Dimension of X is n * p * q * k
  // Dimension of Y is p * q * k
  // Sum over n
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  y[y_idx] = 0;
  for (int i = 0; i < n; i++) {
    int x_idx = i * gridDim.x + blockIdx.x * k + threadIdx.x;
    y[y_idx] += x[x_idx];
  }

}

void BCMSumBackwardWeight(Real *x, Real *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(p * q, 1, 1);
  BCMSumBackwardWeightKernel<<<grid_dim, block_dim>>>(x, y, n);
}

__global__ void BCMSumBackwardWeightKernel(Complex *x, Complex *y, int n) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  // Sum over n
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  y[y_idx].x = 0;
  y[y_idx].y = 0;
  for (int i = 0; i < n; i++) {
    int x_idx = i * gridDim.x + blockIdx.x * k + threadIdx.x;
    y[y_idx].x += x[x_idx].x;
    y[y_idx].y += x[x_idx].y;
  }
}

void BCMSumBackwardWeight(Complex *x, Complex *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(p * q, 1, 1);
  BCMSumBackwardWeightKernel<<<grid_dim, block_dim>>>(x, y, n);
}

}
