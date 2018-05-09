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

#ifndef CORE_INCLUDE_KERNELS_H_
#define CORE_INCLUDE_KERNELS_H_

#include <cuda_runtime.h>

#include "common.h"

namespace dnnmark {

void BCMProductForward(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k);
void BCMProductBackwardWeight(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k);
void BCMProductBackwardData(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k);

void BCMSumForward(Real *x, Real *y,
            int n, int p, int q, int k);
void BCMSumForward(Complex *x, Complex *y,
            int n, int p, int q, int k);
void BCMSumBackwardWeight(Real *x, Real *y,
            int n, int p, int q, int k);
void BCMSumBackwardWeight(Complex *x, Complex *y,
            int n, int p, int q, int k);
void BCMSumBackwardData(Real *x, Real *y,
            int n, int p, int q, int k);
void BCMSumBackwardData(Complex *x, Complex *y,
            int n, int p, int q, int k);

void BCMProductForwardOptimized(Complex *fft_w, Complex *fft_x, Complex *y,
                int n, int p, int q, int k, int tb_size);
void BCMProductBackwardWeightOptimized(Complex *fft_dy, Complex *fft_x, Complex *dw,
                int n, int p, int q, int k, int tb_size);
void BCMProductBackwardDataOptimized(Complex *fft_dy, Complex *fft_w, Complex *dx,
                int n, int p, int q, int k, int tb_size);

void BCMSumForwardOptimized(Complex *x, Complex *y, int n, int p, int k, int q, int tb_size);
void BCMSumBackwardWeightOptimized(Complex *x, Complex *y, int n, int p, int q, int k, int tb_size);
void BCMSumBackwardDataOptimized(Complex *x, Complex *y, int n, int p, int q, int k, int tb_size);

}

#endif // CORE_INCLUDE_KERNELS_H_
