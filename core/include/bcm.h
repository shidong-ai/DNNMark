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

#ifndef CORE_INCLUDE_BCM_H_
#define CORE_INCLUDE_BCM_H_

#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "fft_utility.h"
#include "fft_wrapper.h"

#include <sys/time.h>
#include <iostream>

namespace dnnmark {

template <typename T>
inline void dnnmarkBCMForward(const FFTPlan &w_plan,
                              const FFTPlan &x_plan,
                              const FFTPlan &ifft_plan,
                              T *w, T *x,
                              Complex *fft_w, Complex *fft_x,
                              T *y, T * workspace1, T *workspace2,
                              int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(w_plan, w, fft_w);
  dnnmarkFFT(x_plan, x, fft_x);
  CUDA_CALL(cudaDeviceSynchronize());

  BCMProductForward(fft_w, fft_x, (Complex *)workspace1,
               n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace1, workspace2);
  CUDA_CALL(cudaDeviceSynchronize());
  BCMSumForward(workspace2, y,
         n, p, q, k);
}


template <typename T>
inline void dnnmarkBCMForwardO1(const FFTPlan &w_plan,
                                const FFTPlan &x_plan,
                                const FFTPlan &ifft_plan,
                                T *w, T *x,
                                Complex *fft_w, Complex *fft_x,
                                T *y, T * workspace1, T *workspace2,
                                int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(w_plan, w, fft_w);
  dnnmarkFFT(x_plan, x, fft_x);
  CUDA_CALL(cudaDeviceSynchronize());

  BCMProductForwardO1(fft_w, fft_x,
             (Complex *)workspace1,
             n, p, q, fft_k, k/2 < 32 ? 32 : k/2);

  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumForward((Complex *)workspace1,
         (Complex *)workspace2,
         n, p, q, fft_k);
  
  CUDA_CALL(cudaDeviceSynchronize());
  dnnmarkIFFT(ifft_plan, (Complex *)workspace2, y);
}


template <typename T>
inline void dnnmarkBCMForwardO2(const FFTPlan &w_plan,
                                const FFTPlan &x_plan,
                                const FFTPlan &ifft_plan,
                                T *w, T *x,
                                Complex *fft_w, Complex *fft_x,
                                T *y, T * workspace1, T *workspace2, T *workspace3,
                                int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(w_plan, w, fft_w);
  dnnmarkFFT(x_plan, x, fft_x);
  CUDA_CALL(cudaDeviceSynchronize());

  // Memory reorgnization for weight to shorten the share distance 
  PQK2QPK(fft_w, (Complex *)workspace1, p, q, fft_k);

  BCMProductForwardO2((Complex *)workspace1, fft_x,
             (Complex *)workspace2,
             n, p, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumForward((Complex *)workspace2,
         (Complex *)workspace3,
         n, p, q, fft_k);
  
  CUDA_CALL(cudaDeviceSynchronize());
  dnnmarkIFFT(ifft_plan, (Complex *)workspace3, y);
}


template <typename T>
inline void dnnmarkBCMForwardKF(const FFTPlan &w_plan,
                                const FFTPlan &x_plan,
                                const FFTPlan &ifft_plan,
                                T *w, T *x,
                                Complex *fft_w, Complex *fft_x,
                                T *y, T * workspace,
                                int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(w_plan, w, fft_w);
  dnnmarkFFT(x_plan, x, fft_x);
  CUDA_CALL(cudaDeviceSynchronize());
  
  BCMForward(fft_w, fft_x,
            (Complex *)workspace,
            n, p, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace, y);
  
}


template <typename T>
inline void dnnmarkBCMBackwardWeight(const FFTPlan &y_plan,
                                     const FFTPlan &ifft_plan,
                                     T *dy, Complex *fft_y,
                                     Complex *fft_x,
                                     T *dw, T * workspace1, T *workspace2,
                                     int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(y_plan, dy, fft_y);
  CUDA_CALL(cudaDeviceSynchronize());
  // The FFT of x can be saved
  BCMProductBackwardWeight(fft_y, fft_x,
               (Complex *)workspace1,
               n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace1, workspace2);
  CUDA_CALL(cudaDeviceSynchronize());
  BCMSumBackwardWeight(workspace2, dw,
                       n, p, q, k);
}


template <typename T>
inline void dnnmarkBCMBackwardWeightO1(const FFTPlan &y_plan,
                                       const FFTPlan &ifft_plan,
                                       T *dy, Complex *fft_y,
                                       Complex *fft_x,
                                       T *dw, T *workspace1, T *workspace2,
                                       T *workspace3, T *workspace4,
                                       int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(y_plan, dy, fft_y);
  // The FFT of x can be saved
  CUDA_CALL(cudaDeviceSynchronize());

  // Reuse the sum_y and sum_x here
  NPK2PNK(fft_y, (Complex *)workspace1, n, p, fft_k);
  NQK2QNK(fft_x, (Complex *)workspace2, n, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());
  BCMProductBackwardWeightO1((Complex *)workspace1,
             (Complex *)workspace2,
             (Complex *)workspace3,
             n, p, q, fft_k, k/2 < 32 ? 32 : k/2);

  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumBackwardWeightO2((Complex *)workspace3,
         (Complex *)workspace4,
         n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace4, dw);
}


template <typename T>
inline void dnnmarkBCMBackwardWeightO2(const FFTPlan &y_plan,
                                       const FFTPlan &ifft_plan,
                                       T *dy, Complex *fft_y,
                                       Complex *fft_x,
                                       T *dw, T *workspace1, T *workspace2,
                                       int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(y_plan, dy, fft_y);
  // The FFT of x can be saved
  CUDA_CALL(cudaDeviceSynchronize());

  BCMProductBackwardWeightO2(fft_y, fft_x,
             (Complex *)workspace1,
             n, p, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumBackwardWeightO2((Complex *)workspace1,
         (Complex *)workspace2,
         n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace2, dw);
}


template <typename T>
inline void dnnmarkBCMBackwardWeightKF(const FFTPlan &y_plan,
                                       const FFTPlan &ifft_plan,
                                       T *dy, Complex *fft_y,
                                       Complex *fft_x,
                                       T *dw, T * workspace,
                                       int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  dnnmarkFFT(y_plan, dy, fft_y);
  // The FFT of x can be saved
  CUDA_CALL(cudaDeviceSynchronize());
  BCMBackwardWeight(fft_y, fft_x,
             (Complex *)workspace,
             n, p, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace, dw);
}


template <typename T>
inline void dnnmarkBCMBackwardData(const FFTPlan &ifft_plan,
                                   Complex *fft_y, Complex *fft_w,
                                   T *dx,
                                   T *workspace1, T *workspace2,
                                   int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  // The FFT for dy can be saved
  // The FFT for w can be saved
  BCMProductBackwardData(fft_y, fft_w,
               (Complex *)workspace1,
               n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan, (Complex *)workspace1, workspace2);
  CUDA_CALL(cudaDeviceSynchronize());
  BCMSumBackwardData(workspace2, dx,
         n, p, q, k);
}


template <typename T>
inline void dnnmarkBCMBackwardDataO1(const FFTPlan &ifft_plan,
                                     Complex *fft_y, Complex *fft_w,
                                     T *dx,
                                     T *workspace1, T *workspace2, T *workspace3,
                                     int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  // The FFT for dy can be saved
  // The FFT for w can be saved
  PQK2QPK(fft_w, (Complex *)workspace1, p, q, fft_k);

  CUDA_CALL(cudaDeviceSynchronize());
  BCMProductBackwardDataO1(fft_y,
             (Complex *)workspace1,
             (Complex *)workspace2,
             n, p, q, fft_k, k/2 < 32 ? 32 : k/2);
  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumBackwardDataO2((Complex *)workspace2,
         (Complex *)workspace3,
         n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan,
      (Complex *)workspace3, dx);
}


template <typename T>
inline void dnnmarkBCMBackwardDataO2(const FFTPlan &ifft_plan,
                                     Complex *fft_y, Complex *fft_w,
                                     T *dx,
                                     T *workspace1, T *workspace2,
                                     int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  // The FFT for dy can be saved
  // The FFT for w can be saved
  BCMProductBackwardDataO2(fft_y, fft_w,
             (Complex *)workspace1,
             n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  BCMSumBackwardDataO2((Complex *)workspace1,
         (Complex *)workspace2,
         n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan,
      (Complex *)workspace2, dx);
}


template <typename T>
inline void dnnmarkBCMBackwardDataKF(const FFTPlan &ifft_plan,
                                     Complex *fft_y, Complex *fft_w,
                                     T *dx,
                                     T *workspace,
                                     int n, int p, int q, int k) {
  int fft_k = k / 2 + 1;
  // The FFT for dy can be saved
  // The FFT for w can be saved
  BCMBackwardData(fft_y, fft_w,
             (Complex *)workspace,
             n, p, q, fft_k);
  CUDA_CALL(cudaDeviceSynchronize());

  dnnmarkIFFT(ifft_plan,
      (Complex *)workspace, dx);
}

}

#endif // CORE_INCLUDE_BCM_H_
