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

#ifndef CORE_INCLUDE_CONFIG_KEYWORDS_H_
#define CORE_INCLUDE_CONFIG_KEYWORDS_H_

#include <vector>
#include <string>

namespace dnnmark {

// Configuration section keywords
std::vector<std::string> section_kerwords = {
  "[DNNMark]",
  "[Data]",
  "[Convolution]"
};

// DNNMark keywords
std::vector<std::string> dnnmark_config_keywords = {
  "run_mode"
};

// Data layer keywords
std::vector<std::string> data_config_keywords = {
  "n",
  "c",
  "h",
  "w"
};

// Convolution layer keywords
std::vector<std::string> conv_config_keywords = {
  "name",
  "previous_layer",
  "next_layer",
  "mode",
  "num_output",
  "kernel_size",
  "pad",
  "stride",
  "kernel_size_h",
  "kernel_size_w",
  "pad_h",
  "pad_w",
  "stride_h",
  "stride_w",
  "conv_fwd_pref",
  "conv_bwd_filter_pref",
  "conv_bwd_data_pref"
};

bool isSection(std::string &s);
bool isDNNMarkSection(std::string &s);
bool isDNNMarkKeywordExist(std::string &s);
bool isDataSection(std::string &s);
bool isDataKeywordExist(std::string &s);
bool isConvSection(std::string &s);
bool isConvKeywordExist(std::string &s);

} // namespace dnnmark

#endif // CORE_INCLUDE_CONFIG_KEYWORDS_H_

