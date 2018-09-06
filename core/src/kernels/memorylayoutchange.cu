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

__global__ void NPK2PNK_Kernel(Complex *x, Complex *y, int n, int p, int k) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  int p_idx = blockDim.y;
  int n_idx = idx / k;
  int k_idx = idx % k;
  if (idx >= (n * k))
    return;
  int x_idx = n_idx * p * k + p_idx * k + k_idx;
  int y_idx = p_idx * n * k + bid * blockDim.x + tid;
  y[y_idx].x = x[x_idx].x;
  y[y_idx].y = x[x_idx].y;

}

void NPK2PNK(Complex *x, Complex *y, int n, int p, int k, int tb_size) {
  int block_size = (n * k + tb_size -1 ) / tb_size;
  dim3 block_dim(tb_size, 1, 1);
  dim3 grid_dim(block_size, p, 1);
  NPK2PNK_Kernel<<<grid_dim, block_dim>>>(x, y, n, p, k);
}

__global__ void NQK2QNK_Kernel(Complex *x, Complex *y, int n, int q, int k) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  int q_idx = blockDim.y;
  int n_idx = idx / k;
  int k_idx = idx % k;
  if (idx >= (n * k))
    return;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;
  int y_idx = q_idx * n * k + bid * blockDim.x + tid;
  y[y_idx].x = x[x_idx].x;
  y[y_idx].y = x[x_idx].y;
}

void NQK2QNK(Complex *x, Complex *y, int n, int q, int k, int tb_size) {
  int block_size = (n * k + tb_size -1 ) / tb_size;
  dim3 block_dim(tb_size, 1, 1);
  dim3 grid_dim(block_size, q, 1);
  NQK2QNK_Kernel<<<grid_dim, block_dim>>>(x, y, n, q, k);
}

__global__ void PQK2QPK_Kernel(Complex *x, Complex *y, int p, int q, int k) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  int q_idx = blockDim.y;
  int p_idx = idx / k;
  int k_idx = idx % k;
  if (idx >= (p * k))
    return;

  int x_idx = p_idx * q * k + q_idx * k + k_idx;
  int y_idx = q_idx * p * k + bid * blockDim.x + tid;
  y[y_idx].x = x[x_idx].x;
  y[y_idx].y = x[x_idx].y;

}

void PQK2QPK(Complex *x, Complex *y, int p, int q, int k, int tb_size) {
  int block_size = (p * k + tb_size -1 ) / tb_size;
  dim3 block_dim(tb_size, 1, 1);
  dim3 grid_dim(block_size, q, 1);
  PQK2QPK_Kernel<<<grid_dim, block_dim>>>(x, y, p, q, k);
}

}
