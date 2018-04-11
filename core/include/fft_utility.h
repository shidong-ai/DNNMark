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

#ifndef CORE_INCLUDE_FFT_UTILITY_H_
#define CORE_INCLUDE_FFT_UTILITY_H_

#include <iostream>

#include "common.h"
#include "dnn_param.h"
#include "timer.h"

namespace dnnmark {

enum FFTType {
#ifdef NVIDIA_CUDNN
  R2C = CUFFT_R2C, // Real to complex (interleaved)
  C2R = CUFFT_C2R, // Complex (interleaved) to real
  C2C = CUFFT_C2C, // Complex to complex (interleaved)
  D2Z = CUFFT_D2Z, // Double to double-complex (interleaved)
  Z2D = CUFFT_Z2D, // Double-complex (interleaved) to double
  Z2Z = CUFFT_Z2Z  // Double-complex to double-complex (interleaved)
#endif
};

enum FFTPlanType {
  FFT_1D = 0,
  FFT_2D,
  FFT_3D,
  FFT_MANY
};

enum FFTDirction {
#ifdef NVIDIA_CUDNN
  FFT = cuFFTFORWARD,
  IFFT = cuFFTINVERSE
#endif
};

class FFTPlan {
 private:
#ifdef NVIDIA_CUDNN
  cufftHandle plan_;
#endif
 public:
  FFTPlan();
  ~FFTPlan();
  int SetPlan(FFTPlanType plan_type, int nx, FFTType type, int batch);
#ifdef NVIDIA_CUDNN
  cufftHandle GetPlan() const;
#endif
#ifdef AMD_MIOPEN
#endif
};


} // namespace dnnmark

#endif // CORE_INCLUDE_FFT_UTILITY_H_
