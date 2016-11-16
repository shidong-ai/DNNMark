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

#ifndef CORE_INCLUDE_DNNMARK_H_
#define CORE_INCLUDE_DNNMARK_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <map>
#include <memory>
#include <glog/logging.h>

#include "common.h"
#include "utility.h"
#include "gpu_utility.h"
#include "dnn_config_keywords.h"
#include "dnn_param.h"
#include "dnn_utility.h"
#include "data_manager.h"
#include "dnn_layer.h"


namespace dnnmark {

template <typename T>
class DNNMark {
 private:
  RunMode run_mode_;
  Handle handle_;
  std::map<int, std::shared_ptr<Layer<T>>> layers_map_;
  std::map<std::string, int> name_id_map_;
  std::list<std::shared_ptr<Layer<T>>> composed_model_;
  int num_layers_;

  // Private functions
  void SetLayerParams(LayerType layer_type,
                      int current_layer_id,
                      const std::string &var,
                      const std::string &val);

 public:

  DNNMark();
  int ParseAllConfig(const std::string &config_file);
  int ParseGeneralConfig(const std::string &config_file);
  int ParseSpecifiedConfig(const std::string &config_file,
                           LayerType layer_type);
  int Initialize();
  int RunAll();
  int Forward();
  int Backward();

};

} // namespace dnnmark

#endif // CORE_INCLUDE_DNNMARK_H_
