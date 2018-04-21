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

#include "fft_wrapper.h"

namespace dnnmark {

//
// FFT
//

template <>
void dnnmarkFFT(const FFTPlan &plan, float *input, cuComplex *output) {
  CUFFT_CALL(cufftExecR2C(plan.Get(), input, output));
}

template <>
void dnnmarkFFT(const FFTPlan &plan, cuComplex *input, cuComplex *output) {
  CUFFT_CALL(cufftExecC2C(plan.Get(), input, output, FFT));
}

template <>
void dnnmarkFFT(const FFTPlan &plan, double *input, cuDoubleComplex *output) {
  CUFFT_CALL(cufftExecD2Z(plan.Get(), input, output));
}

template <>
void dnnmarkFFT(const FFTPlan &plan, cuDoubleComplex *input, cuDoubleComplex *output) {
  CUFFT_CALL(cufftExecZ2Z(plan.Get(), input, output, FFT));
}

//
// IFFT
//

template <>
void dnnmarkIFFT(const FFTPlan &plan, cuComplex *input, float *output) {
  CUFFT_CALL(cufftExecC2R(plan.Get(), input, output));
}

template <>
void dnnmarkIFFT(const FFTPlan &plan, cuComplex *input, cuComplex *output) {
  CUFFT_CALL(cufftExecC2C(plan.Get(), input, output, IFFT));
}

template <>
void dnnmarkIFFT(const FFTPlan &plan, cuDoubleComplex *input, double *output) {
  CUFFT_CALL(cufftExecZ2D(plan.Get(), input, output));
}

template <>
void dnnmarkIFFT(const FFTPlan &plan, cuDoubleComplex *input, cuDoubleComplex *output) {
  CUFFT_CALL(cufftExecZ2Z(plan.Get(), input, output, IFFT));
}

} // namespace dnnmark
