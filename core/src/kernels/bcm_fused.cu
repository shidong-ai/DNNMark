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

__global__ void BCMForwardKernel(Complex *fft_w,
                                 Complex *fft_x,
                                 Complex *y,
                                 int p, int q, int k) {
  // Dimension of W after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * p * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int p_tid = threadIdx.y;
  int p_bid = blockIdx.y;
  int n_idx = blockIdx.z;
  int p_idx = p_bid * blockDim.y + p_tid;

  extern __shared__ Complex shared_mem[];

  int y_idx = n_idx * p * k + p_idx * k + k_idx;

  if (p_tid == 0) {
    for (int q_idx = 0; q_idx < q; q_idx++) {
      int x_idx = n_idx * q * k + q_idx * k + k_idx;
      shared_mem[q_idx * k + k_idx].x = fft_x[x_idx].x;
      shared_mem[q_idx * k + k_idx].y = fft_x[x_idx].y;
    }
  }
  __syncthreads();


  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int q_idx = 0; q_idx < q; q_idx++) {
    int share_mem_idx = q_idx * k + k_idx;
    int w_idx = p_idx * q * k + q_idx * k + k_idx;
    temp.x += fft_w[w_idx].x * shared_mem[share_mem_idx].x -
                 fft_w[w_idx].y * shared_mem[share_mem_idx].y;
    temp.y += fft_w[w_idx].x * shared_mem[share_mem_idx].y +
               fft_w[w_idx].y * shared_mem[share_mem_idx].x;
  }
  y[y_idx] = temp;

}

void BCMForward(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k) {

  int block_size = (k - 1) * 2;
  int tid_p = 1024 / block_size > p ? p : 1024 / block_size; // must be power of 2
  int bid_p = p / tid_p;
  dim3 block_dim(k, tid_p, 1);
  dim3 grid_dim(1, bid_p, n);
  
  size_t shared_mem_size = q * k * sizeof(Complex);
  BCMForwardKernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_w, fft_x, y, p, q, k);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
}


__global__ void BCMBackwardWeightKernel(Complex *fft_dy,
                                        Complex *fft_x, Complex *dw,
                                        int n, int p, int q, int k) {
  // Dimension of dY after FFT is p * n * k (k is floor(n/2)+1)
  // Dimension of X after FFT is q * n * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is p * q * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int q_tid = threadIdx.y;
  int q_bid = blockIdx.y;
  int p_idx = blockIdx.z;
  int q_idx = q_bid * blockDim.y + q_tid;

  extern __shared__ Complex shared_mem[];

  int dw_idx = p_idx * q * k + q_idx * k + k_idx;

  if (q_tid == 0) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      int dy_idx = n_idx * p * k + p_idx * k + k_idx;
      shared_mem[n_idx * k + k_idx].x = fft_dy[dy_idx].x;
      shared_mem[n_idx * k + k_idx].y = fft_dy[dy_idx].y;
    }
  }
  __syncthreads();

  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int n_idx = 0; n_idx < n; n_idx++) {
    int share_mem_idx = n_idx * k + k_idx;
    int x_idx = n_idx * q * k + q_idx * k + k_idx;
    temp.x += fft_x[x_idx].x * shared_mem[share_mem_idx].x -
                 fft_x[x_idx].y * shared_mem[share_mem_idx].y;
    temp.y += fft_x[x_idx].x * shared_mem[share_mem_idx].y -
               fft_x[x_idx].y * shared_mem[share_mem_idx].x;
  }
  dw[dw_idx] = temp;
}

void BCMBackwardWeight(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k) {
  int block_size = (k - 1) * 2;
  int tid_q = 1024 / block_size > q ? q : 1024 / block_size; // must be power of 2
  int bid_q = q / tid_q;
  dim3 block_dim(k, tid_q, 1);
  dim3 grid_dim(1, bid_q, p);

  // Shared memory is the limitation
  size_t shared_mem_size = n * k * sizeof(Complex);
  BCMBackwardWeightKernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_x, dw, n, p, q, k);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
}

__global__ void BCMBackwardDataKernel(Complex *fft_dy,
                                        Complex *fft_w, Complex *dx,
                                        int n, int p, int q, int k) {
  // Dimension of dY after FFT is p * n * k (k is floor(n/2)+1)
  // Dimension of W after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of dX after this kernel is n * q * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int q_tid = threadIdx.y;
  int q_bid = blockIdx.y;
  int n_idx = blockIdx.z;
  int q_idx = q_bid * blockDim.y + q_tid;

  extern __shared__ Complex shared_mem[];

  int dx_idx = n_idx * q * k + q_idx * k + k_idx;

  if (q_tid == 0) {
    for (int p_idx = 0; p_idx < p; p_idx++) {
      int dy_idx = n_idx * p * k + p_idx * k + k_idx;
      shared_mem[p_idx * k + k_idx].x = fft_dy[dy_idx].x;
      shared_mem[p_idx * k + k_idx].y = fft_dy[dy_idx].y;
    }
  }
  __syncthreads();

  Complex temp;
  temp.x = 0;
  temp.y = 0;
  for (int p_idx = 0; p_idx < p; p_idx++) {
    int share_mem_idx = p_idx * k + k_idx;
    int w_idx = p_idx * q * k + q_idx * k + k_idx;
    temp.x += fft_w[w_idx].x * shared_mem[share_mem_idx].x -
                 fft_w[w_idx].y * shared_mem[share_mem_idx].y;
    temp.y += fft_w[w_idx].x * shared_mem[share_mem_idx].y -
               fft_w[w_idx].y * shared_mem[share_mem_idx].x;
  }
  dx[dx_idx] = temp;
}

void BCMBackwardData(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k) {
  int block_size = (k - 1) * 2;
  int tid_q = 1024 / block_size > q ? q : 1024 / block_size; // must be power of 2
  int bid_q = q / tid_q;
  dim3 block_dim(k, tid_q, 1);
  dim3 grid_dim(1, bid_q, n);

  size_t shared_mem_size = p * k * sizeof(Complex);
  BCMBackwardDataKernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_w, dx, n, p, q, k);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
}

}
