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

__global__ void BCMSumForwardKernel(Real *x, Real *y, int q) {
  // Dimension of X is n * p * q * k
  // Dimension of Y is n * q * k
  // Sum over q
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  y[y_idx] = 0;
  Real temp;
  temp = 0;
  for (int i = 0; i < q; i++) {
    int x_idx = blockIdx.x * q * k + i * k + threadIdx.x;
    temp += x[x_idx];
  }
  y[y_idx] = temp;

}

void BCMSumForward(Real *x, Real *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * p, 1, 1);
  BCMSumForwardKernel<<<grid_dim, block_dim>>>(x, y, q);
}

__global__ void BCMSumForwardKernel(Complex *x, Complex *y, int q) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  // Sum over q
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int i = 0; i < q; i++) {
    int x_idx = blockIdx.x * q * k + i * k + threadIdx.x;
    temp.x += x[x_idx].x;
    temp.y += x[x_idx].y;
  }
  y[y_idx].x = temp.x;
  y[y_idx].y = temp.y;

}

void BCMSumForward(Complex *x, Complex *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * p, 1, 1);
  BCMSumForwardKernel<<<grid_dim, block_dim>>>(x, y, q);
}

__global__ void BCMSumBackwardWeightKernel(Real *x, Real *y, int n) {
  // Dimension of X is n * p * q * k
  // Dimension of Y is p * q * k
  // Sum over n
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  Real temp = 0;
  for (int i = 0; i < n; i++) {
    int x_idx = i * gridDim.x * k + blockIdx.x * k + threadIdx.x;
    temp += x[x_idx];
  }
  y[y_idx] = temp;

}

void BCMSumBackwardWeight(Real *x, Real *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(p * q, 1, 1);
  BCMSumBackwardWeightKernel<<<grid_dim, block_dim>>>(x, y, n);
}

__global__ void BCMSumBackwardWeightKernel(Complex *x, Complex *y, int n) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is p * q * k (k is floor(n/2)+1)
  // Sum over n
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int i = 0; i < n; i++) {
    int x_idx = i * gridDim.x * k + blockIdx.x * k + threadIdx.x;
    temp.x += x[x_idx].x;
    temp.y += x[x_idx].y;
  }
  y[y_idx].x = temp.x;
  y[y_idx].y = temp.y;
}

void BCMSumBackwardWeight(Complex *x, Complex *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(p * q, 1, 1);
  BCMSumBackwardWeightKernel<<<grid_dim, block_dim>>>(x, y, n);
}

__global__ void BCMSumBackwardDataKernel(Real *x, Real *y, int p, int q) {
  // Dimension of X is n * p * q * k
  // Dimension of Y is n * q * k
  // Sum over p
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  Real temp = 0;
  int n_idx = blockIdx.x / q;
  int q_idx = blockIdx.x % q;
  for (int i = 0; i < p; i++) {
    int x_idx = n_idx * p * q * k + i * q * k + q_idx * k + threadIdx.x;
    temp += x[x_idx];
  }
  y[y_idx] = temp;

}

void BCMSumBackwardData(Real *x, Real *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * q, 1, 1);
  BCMSumBackwardDataKernel<<<grid_dim, block_dim>>>(x, y, p, q);
}

__global__ void BCMSumBackwardDataKernel(Complex *x, Complex *y, int p, int q) {
  // Dimension of X is n * p * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  // Sum over p
  int k = blockDim.x;
  int y_idx = blockIdx.x * k + threadIdx.x;
  Complex temp;
  temp.x = 0;
  temp.y = 0;
  int n_idx = blockIdx.x / q;
  int q_idx = blockIdx.x % q;
  for (int i = 0; i < p; i++) {
    int x_idx = n_idx * p * q * k + i * q * k + q_idx * k + threadIdx.x;
    temp.x += x[x_idx].x;
    temp.y += x[x_idx].y;
  }
  y[y_idx].x = temp.x;
  y[y_idx].y = temp.y;
}

void BCMSumBackwardData(Complex *x, Complex *y, int n, int p, int q, int k) {
  dim3 block_dim(k, 1 , 1);
  dim3 grid_dim(n * q, 1, 1);
  BCMSumBackwardDataKernel<<<grid_dim, block_dim>>>(x, y, p, q);
}

__global__ void BCMSumBackwardWeightO2Kernel(Complex *x, Complex *y,
                                int q, int n, int k) {
  // Dimension of X is p * q * n * k (k is floor(n/2)+1)
  // Dimension of Y is p * q * k (k is floor(n/2)+1)
  // Sum over n
  int y_idx = blockIdx.y * q * k + blockIdx.x * q + threadIdx.x;
  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int i = 0; i < n; i++) {
    int x_idx = blockIdx.y * q * n * k + blockIdx.x * n * k + i * k + threadIdx.x;
    temp.x += x[x_idx].x;
    temp.y += x[x_idx].y;
  }
  y[y_idx].x = temp.x;
  y[y_idx].y = temp.y;
}

void BCMSumBackwardWeightO2(Complex *x, Complex *y,
                            int n, int p, int q, int k) {
  dim3 block_dim(k, 1, 1);
  dim3 grid_dim(q, p, 1);
  BCMSumBackwardWeightO2Kernel<<<grid_dim, block_dim>>>(x, y, q, n, k);
}

__global__ void BCMSumBackwardDataO2Kernel(Complex *x, Complex *y,
                                int q, int p, int k) {
  // Dimension of X is n * q * p * k (k is floor(n/2)+1)
  // Dimension of Y is n * q * k (k is floor(n/2)+1)
  // Sum over n
  int y_idx = blockIdx.y * q * k + blockIdx.x * k + threadIdx.x;
  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int i = 0; i < p; i++) {
    int x_idx = blockIdx.y * q * p * k + blockIdx.x * p * k + i * k + threadIdx.x;
    temp.x += x[x_idx].x;
    temp.y += x[x_idx].y;
  }
  y[y_idx].x = temp.x;
  y[y_idx].y = temp.y;
}

void BCMSumBackwardDataO2(Complex *x, Complex *y,
                            int n, int p, int q, int k) {
  dim3 block_dim(k, 1, 1);
  dim3 grid_dim(q, n, 1);
  BCMSumBackwardDataO2Kernel<<<grid_dim, block_dim>>>(x, y, q, p, k);
}

}
