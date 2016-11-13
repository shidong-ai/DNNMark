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

#ifndef CORE_INCLUDE_DATA_MANAGER_H_
#define CORE_INCLUDE_DATA_MANAGER_H_

#include <memory>
#include <map>
#include "cudnn.h"
//#include "dnn_png.h"

namespace dnnmark {

template <typename T>
class Data {
 private:
  T *gpu_ptr;
 public:
  Data(int size) {
    CUDA_CALL(cudaMalloc(&gpu_ptr, size * sizeof(T)));
  }
  ~Data() {
    CUDA_CALL(cudaFree(gpu_ptr));
  }
  void Filler() {
  }
  T *Get() { return gpu_ptr; }
};

template <typename T>
class DataManager {
 private:
  // Memory pool indexed by chunk id
  std::map<int, std::shared_ptr<Data<T>>> gpu_data_pool;
  int num_data_chunks_;

  // Constructor
  DataManager()
  : num_data_chunks_(0) {
  }

  // Memory manager instance
  static std::unique_ptr<DataManager<T>> instance;
 public:
  static DataManager<T> *GetInstance() {
    if (instance.get())
      return instance.get();
    instance.reset(new DataManager());
    return instance.get();
  }

  int CreateData(int size) {
    int gen_chunk_id = num_data_chunks_;
    num_data_chunks_++;
    gpu_data_pool.emplace(gen_chunk_id, std::make_shared<Data<T>>(size));
  }

  Data<T> *GetData(int chunk_id) {
    return gpu_data_pool[chunk_id].get();
  }
};

} // namespace dnnmark

#endif // CORE_INCLUDE_DATA_MANAGER_H_

