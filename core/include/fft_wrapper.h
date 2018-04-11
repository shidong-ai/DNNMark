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

#ifndef CORE_INCLUDE_FFT_H_
#define CORE_INCLUDE_FFT_H_

#include "common.h"
#include "fft_utility.h"

namespace dnnmark {

template <typename T1, typename T2>
void dnnmarkFFT(const FFTPlan &plan, const T1 *input, T2 *output);
template <typename T>
void dnnmarkFFT(const FFTPlan &plan, const T *input, T *output);

template <typename T1, typename T2>
void dnnmarkIFFT(const FFTPlan &plan, const T2 *input, T1 *output);
template <typename T>
void dnnmarkIFFT(const FFTPlan &plan, const T *input, T *output);

} // namespace dnnmark

#endif // CORE_INCLUDE_FFT_H_

