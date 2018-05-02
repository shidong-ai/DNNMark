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
#include <stdio.h>
#include <iostream>

namespace dnnmark {

__global__ void BCMProductForwardKernel(Complex *fft_w, Complex *fft_x, Complex *y) {
  // Dimension of W after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * p * q * k (k is floor(n/2)+1)
  int n = gridDim.z;
  int p = gridDim.y;
  int q = gridDim.x;
  int k = blockDim.x;
  int k_idx = threadIdx.x;
  int q_idx = blockIdx.x;
  int p_idx = blockIdx.y;
  int n_idx = blockIdx.z;
  int w_idx = p_idx * q * k + q_idx * k + k_idx;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;
  int y_idx = n_idx * p * q * k + p_idx * q * k + q_idx * k + k_idx;

  y[y_idx].x = fft_w[w_idx].x * fft_x[x_idx].x -
               fft_w[w_idx].y * fft_x[x_idx].y;
  y[y_idx].y = fft_w[w_idx].x * fft_x[x_idx].y +
               fft_w[w_idx].y * fft_x[x_idx].x;

}

void BCMProductForward(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k) {
  dim3 block_dim(k, 1, 1);
  dim3 grid_dim(q, p, n);
  BCMProductForwardKernel<<<grid_dim, block_dim>>>(fft_w, fft_x, y);
}

__global__ void BCMProductBackwardWeightKernel(Complex *fft_dy,
                                               Complex *fft_x, Complex *dw) {
  // Dimension of dY after FFT is n * p * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is n * p * q * k (k is floor(n/2)+1)
  int n = gridDim.z;
  int p = gridDim.y;
  int q = gridDim.x;
  int k = blockDim.x;
  int k_idx = threadIdx.x;
  int q_idx = blockIdx.x;
  int p_idx = blockIdx.y;
  int n_idx = blockIdx.z;
  int dy_idx = n_idx * p * k + p_idx * k + k_idx;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;
  int dw_idx = n_idx * p * q * k + p_idx * q * k + q_idx * k + k_idx;

  dw[dw_idx].x = fft_dy[dy_idx].x * fft_x[x_idx].x -
               fft_dy[dy_idx].y * fft_x[x_idx].y;
  dw[dw_idx].y = fft_dy[dy_idx].x * (0 - fft_x[x_idx].y) -
               fft_dy[dy_idx].y * fft_x[x_idx].x;

}

void BCMProductBackwardWeight(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k) {
  dim3 block_dim(k, 1, 1);
  dim3 grid_dim(q, p, n);
  BCMProductBackwardWeightKernel<<<grid_dim, block_dim>>>(fft_dy, fft_x, dw);
}

__global__ void BCMProductBackwardDataKernel(Complex *fft_dy,
                                             Complex *fft_w, Complex *dx) {
  // Dimension of dY after FFT is n * p * k (k is floor(n/2)+1)
  // Dimension of W after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of dX after this kernel is n * p * q * k (k is floor(n/2)+1)
  int n = gridDim.z;
  int p = gridDim.y;
  int q = gridDim.x;
  int k = blockDim.x;
  int k_idx = threadIdx.x;
  int q_idx = blockIdx.x;
  int p_idx = blockIdx.y;
  int n_idx = blockIdx.z;
  int dy_idx = n_idx * p * k + p_idx * k + k_idx;
  int w_idx = p_idx * q * k + q_idx * k + k_idx;
  int dx_idx = n_idx * p * q * k + p_idx * q * k + q_idx * k + k_idx;

  dx[dx_idx].x = fft_dy[dy_idx].x * fft_w[w_idx].x -
               fft_dy[dy_idx].y * fft_w[w_idx].y;
  dx[dx_idx].y = fft_dy[dy_idx].x * (0 - fft_w[w_idx].y) -
               fft_dy[dy_idx].y * fft_w[w_idx].x;

}

void BCMProductBackwardData(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k) {
  dim3 block_dim(k, 1, 1);
  dim3 grid_dim(q, p, n);
  BCMProductBackwardDataKernel<<<grid_dim, block_dim>>>(fft_dy, fft_w, dx);
}

}
