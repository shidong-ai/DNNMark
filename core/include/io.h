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

#ifndef CORE_INCLUDE_IO_H_
#define CORE_INCLUDE_IO_H_

#include <iostream>
#include <fstream>
#include <string>

#include "dnnmark.h"

namespace dnnmark {

enum Direction {
  INPUT = 0,
  OUTPUT
}

enum Extension {
  TXT = 0,
  CSV
}

std::string GenFileName(int n,
                        int c,
                        int h,
                        int w,
                        Direction direction,
                        Extension ext) {
  std::string common_file_name = "n" + std::to_string(n) + "_c" +
    std::to_string(c) + "_h" + std::to_string(h) + "_w" + std::to_string(w);
  if (ext == TXT) {
    common_file_name += ".txt";
  } else if (ext == CSV) {
    common_file_name += ".csv";
  }
  return direction == INPUT ? "input_" + common_file_name :
                      "output_" + common_file_name;
}

template <typename T>
void ToFile(const T *data,
            const std::string &output_file,
            DataDim dim, Extension ext) {
  std::ofstream output(output_file, std::ofstream::out);
  int num_rows = dim.c_ * dim_.h * dim_.w;
  int num_cols = dim.n_;
  std::string delimiter = ext == CSV ? "," : " ";
  if (output) {
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols; j++) {
        // All column major
        output << data[i * num_rows + j];
        if (j == num_cols - 1)
          output << std::endl;
        else
          output << delimiter;
      }
    }
  } else {
    LOG(FATAL) << "Cannot open file " + output_file_path + ", exiting...";
  }
  output.close();
}

} // namespace dnnmark

#endif // CORE_INCLUDE_IO_H_

