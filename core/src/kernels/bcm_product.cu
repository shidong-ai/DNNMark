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

__global__ void BCMProductForwardO1Kernel(Complex *fft_w,
                                                 Complex *fft_x,
                                                 Complex *y,
                                                 int p, int q, int k) {
  // Dimension of W after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * p * q * k (k is floor(n/2)+1)
  int tid = threadIdx.x;
  int tid_p = threadIdx.y;
  int bid = blockIdx.x;
  int bid_p = blockIdx.y;
  int idx = bid * blockDim.x + tid;
  int n_idx = blockIdx.z;
  int q_idx = idx / k;
  int k_idx = idx % k;
  int p_idx = bid_p * blockDim.y + tid_p;

  extern __shared__ Complex shared_mem[];

  if (idx >= (k * q)) {
    return;
  }

  int y_idx = n_idx * p * q * k + p_idx * q * k + idx;
  int w_idx = p_idx * q * k + q_idx * k + k_idx;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;

  if (tid_p == 0) {
    shared_mem[q_idx * k + k_idx].x = fft_x[x_idx].x;
    shared_mem[q_idx * k + k_idx].y = fft_x[x_idx].y;
  }
  __syncthreads();
  
  y[y_idx].x = fft_w[w_idx].x * shared_mem[q_idx * k + k_idx].x -
               fft_w[w_idx].y * shared_mem[q_idx * k + k_idx].y;
  y[y_idx].y = fft_w[w_idx].x * shared_mem[q_idx * k + k_idx].y +
               fft_w[w_idx].y * shared_mem[q_idx * k + k_idx].x;

}

void BCMProductForwardO1(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k, int tb_size) {
  int block_size = (k * q + tb_size -1 ) / tb_size;
  int tb_size_p = (1024 / tb_size) > p ? p : (1024 / tb_size);
  int block_size_p = p / tb_size_p;
  dim3 block_dim(tb_size, tb_size_p, 1);
  dim3 grid_dim(block_size, block_size_p, n);
  
  size_t shared_mem_size = q * k * sizeof(Complex);
  BCMProductForwardO1Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_w, fft_x, y, p, q, k);
}

__global__ void BCMProductBackwardWeightO1Kernel(Complex *fft_dy,
                                               Complex *fft_x, Complex *dw,
                                               int q, int n, int k) {
  // Dimension of dY after FFT is p * n * k (k is floor(n/2)+1)
  // Dimension of X after FFT is q * n * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is p * q * n * k (k is floor(n/2)+1)
  int tid = threadIdx.x;
  int tid_q = threadIdx.y;
  int bid = blockIdx.x;
  int bid_q = blockIdx.y;
  int idx = bid * blockDim.x + tid;
  int p_idx = blockIdx.z;
  int q_idx = bid_q * blockDim.y + tid_q;
  int n_idx = idx / k;
  int k_idx = idx % k;

  extern __shared__ Complex shared_mem[];

  if (idx >= (n * k)) {
    return;
  }
  int dw_idx = p_idx * q * n * k + q_idx * n * k + idx;
  int dy_idx = p_idx * n * k + n_idx * k + k_idx;
  int x_idx = q_idx * n * k + n_idx * k + k_idx;
  if (tid_q == 0) {
    shared_mem[n_idx * k + k_idx].x = fft_dy[dy_idx].x;
    shared_mem[n_idx * k + k_idx].y = fft_dy[dy_idx].y;
  }
  __syncthreads();

  dw[dw_idx].x = shared_mem[n_idx * k + k_idx].x * fft_x[x_idx].x -
               shared_mem[n_idx * k + k_idx].y * fft_x[x_idx].y;
  dw[dw_idx].y = shared_mem[n_idx * k + k_idx].x * (0 - fft_x[x_idx].y) -
               shared_mem[n_idx * k + k_idx].y * fft_x[x_idx].x;
}

void BCMProductBackwardWeightO1(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k, int tb_size) {
  int block_size = (n * k + tb_size -1 ) / tb_size;
  int tb_size_q = (1024 / tb_size) > q ? q : (1024 / tb_size);
  int block_size_q = q / tb_size_q;
  dim3 block_dim(tb_size, tb_size_q, 1);
  dim3 grid_dim(block_size, block_size_q, p);
  
  size_t shared_mem_size = n * k * sizeof(Complex);
  BCMProductBackwardWeightO1Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_x, dw, q, n, k);
}

__global__ void BCMProductBackwardDataO1Kernel(Complex *fft_dy,
                                               Complex *fft_w, Complex *dx,
                                               int q, int p, int k) {
  // Dimension of dY after FFT is n * p * k (k is floor(n/2)+1)
  // Dimension of X after FFT is q * p * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is n * q * p * k (k is floor(n/2)+1)
  int tid = threadIdx.x;
  int tid_q = threadIdx.y;
  int bid = blockIdx.x;
  int bid_q = blockIdx.y;
  int idx = bid * blockDim.x + tid;
  int n_idx = blockIdx.z;
  int q_idx = bid_q * blockDim.y + tid_q;
  int p_idx = idx / k;
  int k_idx = idx % k;

  extern __shared__ Complex shared_mem[];

  if (idx >= (p * k)) {
    return;
  }
  int dx_idx = n_idx * q * p * k + q_idx * p * k + idx;
  int dy_idx = n_idx * p * k + p_idx * k + k_idx;
  int w_idx = q_idx * p * k + p_idx * k + k_idx;
  if (tid_q == 0) {
    shared_mem[p_idx * k + k_idx].x = fft_dy[dy_idx].x;
    shared_mem[p_idx * k + k_idx].y = fft_dy[dy_idx].y;
  }
  __syncthreads();

  dx[dx_idx].x = shared_mem[p_idx * k + k_idx].x * fft_w[w_idx].x -
               shared_mem[p_idx * k + k_idx].y * fft_w[w_idx].y;
  dx[dx_idx].y = shared_mem[p_idx * k + k_idx].x * (0 - fft_w[w_idx].y) -
               shared_mem[p_idx * k + k_idx].y * fft_w[w_idx].x;
}

void BCMProductBackwardDataO1(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k, int tb_size) {
  int block_size = (p * k + tb_size -1 ) / tb_size;
  int tb_size_q = (1024 / tb_size) > q ? q : (1024 / tb_size);
  int block_size_q = q / tb_size_q;
  dim3 block_dim(tb_size, tb_size_q, 1);
  dim3 grid_dim(block_size, block_size_q, n);

  size_t shared_mem_size = p * k * sizeof(Complex);
  BCMProductBackwardDataO1Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_w, dx, q, p, k);
}

__global__ void BCMProductForwardO2Kernel(Complex *fft_w,
                                          Complex *fft_x,
                                          Complex *y,
                                          int p, int q, int k) {
  // Dimension of W after FFT is q * p * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of Y is n * p * q * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int p_tid = threadIdx.y;
  int p_idx = blockIdx.y * blockDim.y + p_tid;
  int q_idx = blockIdx.x;
  int n_idx = blockIdx.z;

  extern __shared__ Complex shared_mem[];

  int y_idx = n_idx * p * q * k + p_idx * q * k + q_idx * k + k_idx;
  int w_idx = q_idx * p * k + p_idx * k + k_idx;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;

  if (p_tid == 0) {
    shared_mem[q_idx * k + k_idx].x = fft_x[x_idx].x;
    shared_mem[q_idx * k + k_idx].y = fft_x[x_idx].y;
  }
  __syncthreads();
  
  y[y_idx].x = fft_w[w_idx].x * shared_mem[q_idx * k + k_idx].x -
               fft_w[w_idx].y * shared_mem[q_idx * k + k_idx].y;
  y[y_idx].y = fft_w[w_idx].x * shared_mem[q_idx * k + k_idx].y +
               fft_w[w_idx].y * shared_mem[q_idx * k + k_idx].x;

}

void BCMProductForwardO2(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k) {
  int block_size = (k - 1) * 2;
  int tid_p = (1024 / block_size) > p ? p : (1024 / block_size);
  int bid_p = p / tid_p;
  dim3 block_dim(k, tid_p, 1);
  dim3 grid_dim(q, bid_p, n);

  size_t shared_mem_size = q * k * sizeof(Complex);
  BCMProductForwardO2Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_w, fft_x, y, p, q, k);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
}

__global__ void BCMProductBackwardWeightO2Kernel(Complex *fft_dy,
                                               Complex *fft_x, Complex *dw,
                                               int n, int p, int q, int k) {
  // Dimension of dY after FFT is n * p * k (k is floor(n/2)+1)
  // Dimension of X after FFT is n * q * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is p * q * n * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int tid_q = threadIdx.y;
  int bid_q = blockIdx.y;
  int q_idx = bid_q * blockDim.x + tid_q;
  int n_idx = blockIdx.x;
  int p_idx = blockIdx.z;

  extern __shared__ Complex shared_mem[];

  int dw_idx = p_idx * q * n * k + q_idx * n * k + n_idx * k + k_idx;
  int dy_idx = n_idx * p * k + p_idx * k + k_idx;
  int x_idx = n_idx * q * k + q_idx * k + k_idx;
  if (tid_q == 0) {
    shared_mem[n_idx * k + k_idx].x = fft_dy[dy_idx].x;
    shared_mem[n_idx * k + k_idx].y = fft_dy[dy_idx].y;
  }
  __syncthreads();

  dw[dw_idx].x = shared_mem[n_idx * k + k_idx].x * fft_x[x_idx].x -
               shared_mem[n_idx * k + k_idx].y * fft_x[x_idx].y;
  dw[dw_idx].y = shared_mem[n_idx * k + k_idx].x * (0 - fft_x[x_idx].y) -
               shared_mem[n_idx * k + k_idx].y * fft_x[x_idx].x;
}

void BCMProductBackwardWeightO2(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k) {
  int block_size = (k -1 ) * 2;
  int tid_q = (1024 / block_size) > q ? q : (1024 / block_size);
  int bid_q = q / tid_q;
  dim3 block_dim(k, tid_q, 1);
  dim3 grid_dim(n, bid_q, p);
  
  size_t shared_mem_size = n * k * sizeof(Complex);
  BCMProductBackwardWeightO2Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_x, dw, n, p, q, k);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
}

__global__ void BCMProductBackwardDataO2Kernel(Complex *fft_dy,
                                               Complex *fft_w, Complex *dx,
                                               int n, int p, int q, int k) {
  // Dimension of dY after FFT is n * p * k (k is floor(n/2)+1)
  // Dimension of X after FFT is p * q * k (k is floor(n/2)+1)
  // Dimension of dW after this kernel is n * q * p * k (k is floor(n/2)+1)
  int k_idx = threadIdx.x;
  int tid_q = threadIdx.y;
  int bid_q = blockIdx.y;
  int q_idx = bid_q * blockDim.x + tid_q;
  int n_idx = blockIdx.z;
  int p_idx = blockIdx.x;

  extern __shared__ Complex shared_mem[];

  int dx_idx = n_idx * q * p * k + q_idx * p * k + p_idx * k + k_idx;
  int dy_idx = n_idx * p * k + p_idx * k + k_idx;
  int w_idx = p_idx * q * k + q_idx * k + k_idx;
  if (tid_q == 0) {
    shared_mem[p_idx * k + k_idx].x = fft_dy[dy_idx].x;
    shared_mem[p_idx * k + k_idx].y = fft_dy[dy_idx].y;
  }
  __syncthreads();

  dx[dx_idx].x = shared_mem[p_idx * k + k_idx].x * fft_w[w_idx].x -
               shared_mem[p_idx * k + k_idx].y * fft_w[w_idx].y;
  dx[dx_idx].y = shared_mem[p_idx * k + k_idx].x * (0 - fft_w[w_idx].y) -
               shared_mem[p_idx * k + k_idx].y * fft_w[w_idx].x;
}

void BCMProductBackwardDataO2(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k) {
  int block_size = (k -1 ) * 2;
  int tid_q = (1024 / block_size) > q ? q : (1024 / block_size);
  int bid_q = q / tid_q;
  dim3 block_dim(k, tid_q, 1);
  dim3 grid_dim(p, bid_q, n);

  size_t shared_mem_size = p * k * sizeof(Complex);
  BCMProductBackwardDataO2Kernel<<<grid_dim, block_dim, shared_mem_size>>>(fft_dy, fft_w, dx, n, p, q, k);
}

}
