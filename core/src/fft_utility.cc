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

#include "fft_utility.h"

namespace dnnmark {



FFTPlan::FFTPlan() {
#ifdef NVIDIA_CUDNN
  CUFFT_CALL(cufftCreate(&plan_));
#endif
#ifdef AMD_MIOPEN
#endif
}

FFTPlan::~FFTPlan() {
#ifdef NVIDIA_CUDNN
  CUFFT_CALL(cufftDestroy(plan_));
#endif
#ifdef AMD_MIOPEN
#endif
}

#ifdef NVIDIA_CUDNN
int SetPlan(FFTPlanType plan_type, int nx, FFTType type, int batch) {
  int workspace_size = 0;
  switch (plan_type) {
    case FFT_1D:
      CUFFT_CALL(cufftMakePlan1d(plan_, nx, type, batch, &workspace_size));
      break;
    default:
      LOG(FATAL) << "FFT plan type NOT supported";
  }

  return workspace_size;
}
cufftHandle GetPlan() const { return plan_ }
#endif
#ifdef AMD_MIOPEN
#endif


} // namespace dnnmark
