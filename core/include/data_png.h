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

#ifndef CORE_INCLUDE_DATA_PNG_H_
#define CORE_INCLUDE_DATA_PNG_H_

#include <vector>
#include <map>

namespace dnnmark {

// Seed of random number generator
static unsigned long long int seed = 1234;

class PseudoNumGenerator {
 private:
#ifdef NVIDIA_CUDNN
  curandGenerator_t gen_;  
#endif

  // Constructor
  PseudoNumGenerator() {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, seed));
#endif
  }

  // PNG instance
  static std::unique_ptr<PseudoNumGenerator> instance_;
 public:

  ~PseudoNumGenerator() {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandDestroyGenerator(gen_));
#endif
  }

  static PseudoNumGenerator *GetInstance() {
    if (instance_.get())
      return instance_.get();
    instance_.reset(new PseudoNumGenerator());
    return instance_.get();
  }
  void GenerateUniformData(float *dev_ptr, int size) {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandGenerateUniform(gen_, dev_ptr, size));
#endif
#ifdef AMD_MIOPEN
    float *host_ptr = new float[size];
    for (int i = 0; i < size; i++)
      host_ptr[i] = static_cast <float> (rand()) /
                    (static_cast <float> (RAND_MAX/seed));
    HIP_CALL(hipMemcpy(dev_ptr, host_ptr, size * sizeof(float),
                       hipMemcpyHostToDevice));
    delete []host_ptr;
    
#endif
  }
  void GenerateUniformData(double *dev_ptr, int size) {
#ifdef NVIDIA_CUDNN
    CURAND_CALL(curandGenerateUniformDouble(gen_, dev_ptr, size));
#endif
#ifdef AMD_MIOPEN
    double *host_ptr = new double[size];
    for (int i = 0; i < size; i++)
      host_ptr[i] = static_cast <double> (rand()) /
                    (static_cast <double> (RAND_MAX/seed));
    HIP_CALL(hipMemcpy(dev_ptr, host_ptr, size * sizeof(double),
                       hipMemcpyHostToDevice));
    delete []host_ptr;
#endif
  }  
};

std::unique_ptr<PseudoNumGenerator> PseudoNumGenerator::instance_ = nullptr;

} // namespace dnnmark

#endif // CORE_INCLUDE_DATA_PNG_H_

