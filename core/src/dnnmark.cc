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

#include "cudnn.h"

#include "dnnmark.h"

namespace dnnmark {

//
// DNNMark class definition
//
template <typename T>
DNNMark<T>::DNNMark()
: run_mode_(NONE), handle_() {}

template <typename T>
int DNNMark<T>::ParseAllConfig(const std::string &config_file) {
  // TODO: use multithread in the future
  // Parse DNNMark specific config
  ParseDNNMarkConfig(config_file);

  // Parse Convolution specific config
  ParseConvolutionConfig(config_file);

}

template <typename T>
int DNNMark<T>::ParseDNNMarkConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);
  LOG(INFO) << "Search and parse general DNNMark configuration";

  // TODO: insert assert regarding run_mode_

  // Parse DNNMark config
  std::string s;
  bool is_dnnmark_section = false;
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isSpecifiedSection(s, "[DNNMark]") ||
        isCommentStr(s) || isEmptyStr(s)) {
      is_dnnmark_section = true;
      continue;
    } else if (isSection(s) && is_dnnmark_section) {
      break;
    } else if (!is_dnnmark_section) {
      continue;
    }

    // Obtain the acutal variable and value
    std::string var;
    std::string val;
    SplitStr(s, &var, &val);
    TrimStr(&var);
    TrimStr(&val);

    // Process all the keywords in config
    if(isSectionKeywordExist(var, dnnmark_config_keywords)) {
      if (!var.compare("run_mode")) {
        if (!val.compare("none"))
          run_mode_ = NONE;
        else if(!val.compare("standalone"))
          run_mode_ = STANDALONE;
        else if(!val.compare("composed"))
          run_mode_ = COMPOSED;
        else
          std::cerr << "Unknown run mode" << std::endl;
      }
    } else {
      std::cerr << var << "Keywords not exists" << std::endl;
      //TODO return error
    }
  }

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParseConvolutionConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);
  LOG(INFO) << "Search and parse convolution layer configuration";

  // Parse DNNMark config
  std::string s;
  int current_layer_id;
  DataDim *input_dim;
  ConvolutionParam *conv_param;
  bool is_conv_section = false;
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)){
      continue;
    } else if (isSpecifiedSection(s, "[Convolution]")) {
      LOG(INFO) << "Add convolution layer";
      is_conv_section = true;
      // Create a layer in the main class
      current_layer_id = num_layers_;
      layers_map_.emplace(current_layer_id,
        std::make_shared<ConvolutionLayer<T>>(&handle_));
      layers_map_[current_layer_id]->setLayerId(current_layer_id);
      layers_map_[current_layer_id]->setLayerType(CONVOLUTION);
      num_layers_++;
      continue;
    } else if (isSection(s) && is_conv_section) {
      break;
    } else if (!is_conv_section) {
      continue;
    }

    // Obtain the acutal variable and value
    std::string var;
    std::string val;
    SplitStr(s, &var, &val);
    TrimStr(&var);
    TrimStr(&val);

    // Obtain the data dimension and parameters variable within layer class
    input_dim = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                (layers_map_[current_layer_id])->getInputDim();
    conv_param = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                 (layers_map_[current_layer_id])->getConvParam();

    // Process all the keywords in config
    if(isSectionKeywordExist(var, conv_config_keywords)) {
      if (!var.compare("n")) {
        input_dim->n_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("c")) {
        input_dim->c_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("h")) {
        input_dim->h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("w")) {
        input_dim->w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("name")) {
        std::dynamic_pointer_cast<ConvolutionLayer<T>>
          (layers_map_[current_layer_id])->setLayerName(val.c_str());
        name_id_map_[val] = current_layer_id;
        continue;
      }
      if (!var.compare("previous_layer_name")) {
        std::dynamic_pointer_cast<ConvolutionLayer<T>>
          (layers_map_[current_layer_id])->setPrevLayerName(val.c_str());
        continue;
      }
      if (!var.compare("conv_mode")) {
        if (!val.compare("convolution"))
          conv_param->mode_ = CUDNN_CONVOLUTION;
        else if (!val.compare("cross_correlation"))
          conv_param->mode_ = CUDNN_CROSS_CORRELATION;
        continue;
      }
      if (!var.compare("num_output")) {
        conv_param->output_num_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("kernel_size")) {
        conv_param->kernel_size_h_ = atoi(val.c_str());
        conv_param->kernel_size_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad")) {
        conv_param->pad_h_ = atoi(val.c_str());
        conv_param->pad_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride")) {
        conv_param->stride_u_ = atoi(val.c_str());
        conv_param->stride_v_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("kernel_size_h")) {
        conv_param->kernel_size_h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("kernel_size_w")) {
        conv_param->kernel_size_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad_h")) {
        conv_param->pad_h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad_w")) {
        conv_param->pad_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride_h")) {
        conv_param->stride_u_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride_w")) {
        conv_param->stride_v_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("conv_fwd_pref")) {
        if (!val.compare("no_workspace"))
          conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
        else if (!val.compare("fastest"))
          conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
        else if (!val.compare("specify_workspace_limit"))
          conv_param->conv_fwd_pref_ =
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
        continue;
      }
      if (!var.compare("conv_bwd_filter_pref")) {
        if (!val.compare("no_workspace"))
          conv_param->conv_bwd_filter_pref_ =
            CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
        else if (!val.compare("fastest"))
          conv_param->conv_bwd_filter_pref_ =
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
        else if (!val.compare("specify_workspace_limit"))
          conv_param->conv_bwd_filter_pref_ =
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
        continue;
      }
      if (!var.compare("conv_bwd_data_pref")) {
        if (!val.compare("no_workspace"))
          conv_param->conv_bwd_data_pref_ =
            CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
        else if (!val.compare("fastest"))
          conv_param->conv_bwd_data_pref_ =
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        else if (!val.compare("specify_workspace_limit"))
          conv_param->conv_bwd_data_pref_ =
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        continue;
      }

    } else {
      std::cerr << var << "Keywords not exists" << std::endl;
      //TODO return error
    }
  }

  LOG(INFO) << *conv_param;

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParsePoolingConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);
  LOG(INFO) << "Search and parse pooling layer configuration";

  // Parse DNNMark config
  std::string s;
  int current_layer_id;
  DataDim *input_dim;
  PoolingParam *pool_param;
  bool is_pooling_section = false;
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)){
      continue;
    } else if (isSpecifiedSection(s, "[Pooling]")) {
      LOG(INFO) << "Add pooling layer";
      is_pooling_section = true;
      // Create a layer in the main class
      current_layer_id = num_layers_;
      layers_map_.emplace(current_layer_id,
        std::make_shared<PoolingLayer<T>>(&handle_));
      layers_map_[current_layer_id]->setLayerId(current_layer_id);
      layers_map_[current_layer_id]->setLayerType(POOLING);
      num_layers_++;
      continue;
    } else if (isSection(s) && is_pooling_section) {
      break;
    } else if (!is_pooling_section) {
      continue;
    }

    // Obtain the acutal variable and value
    std::string var;
    std::string val;
    SplitStr(s, &var, &val);
    TrimStr(&var);
    TrimStr(&val);

    // Obtain the data dimension and parameters variable within layer class
    input_dim = std::dynamic_pointer_cast<PoolingLayer<T>>
                (layers_map_[current_layer_id])->getInputDim();
    pool_param = std::dynamic_pointer_cast<PoolingLayer<T>>
                 (layers_map_[current_layer_id])->getPoolParam();

    // Process all the keywords in config
    if(isSectionKeywordExist(var, pool_config_keywords)) {
      if (!var.compare("n")) {
        input_dim->n_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("c")) {
        input_dim->c_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("h")) {
        input_dim->h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("w")) {
        input_dim->w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("name")) {
        std::dynamic_pointer_cast<PoolingLayer<T>>
          (layers_map_[current_layer_id])->setLayerName(val.c_str());
        name_id_map_[val] = current_layer_id;
        continue;
      }
      if (!var.compare("previous_layer_name")) {
        std::dynamic_pointer_cast<PoolingLayer<T>>
          (layers_map_[current_layer_id])->setPrevLayerName(val.c_str());
        continue;
      }
      if (!var.compare("pool_mode")) {
        if (!val.compare("max"))
          pool_param->mode_ = CUDNN_POOLING_MAX;
        else if (!val.compare("avg_include_padding"))
          pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        else if (!val.compare("avg_exclude_padding"))
          pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        continue;
      }
      if (!var.compare("kernel_size")) {
        pool_param->kernel_size_h_ = atoi(val.c_str());
        pool_param->kernel_size_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad")) {
        pool_param->pad_h_ = atoi(val.c_str());
        pool_param->pad_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride")) {
        pool_param->stride_h_ = atoi(val.c_str());
        pool_param->stride_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("kernel_size_h")) {
        pool_param->kernel_size_h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("kernel_size_w")) {
        pool_param->kernel_size_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad_h")) {
        pool_param->pad_h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("pad_w")) {
        pool_param->pad_w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride_h")) {
        pool_param->stride_h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("stride_w")) {
        pool_param->stride_w_ = atoi(val.c_str());
        continue;
      }
    } else {
      std::cerr << var << "Keywords not exists" << std::endl;
      //TODO return error
    }
  }

  LOG(INFO) << *pool_param;

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParseLRNConfig(const std::string &config_file) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);
  LOG(INFO) << "Search and parse LRN layer configuration";

  // Parse DNNMark config
  std::string s;
  int current_layer_id;
  DataDim *input_dim;
  LRNParam *lrn_param;
  bool is_lrn_section = false;
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)){
      continue;
    } else if (isSpecifiedSection(s, "[LRN]")) {
      LOG(INFO) << "Add LRN layer";
      is_lrn_section = true;
      // Create a layer in the main class
      current_layer_id = num_layers_;
      layers_map_.emplace(current_layer_id,
        std::make_shared<LRNLayer<T>>(&handle_));
      layers_map_[current_layer_id]->setLayerId(current_layer_id);
      layers_map_[current_layer_id]->setLayerType(LRN);
      num_layers_++;
      continue;
    } else if (isSection(s) && is_lrn_section) {
      break;
    } else if (!is_lrn_section) {
      continue;
    }

    // Obtain the acutal variable and value
    std::string var;
    std::string val;
    SplitStr(s, &var, &val);
    TrimStr(&var);
    TrimStr(&val);

    // Obtain the data dimension and parameters variable within layer class
    input_dim = std::dynamic_pointer_cast<LRNLayer<T>>
                (layers_map_[current_layer_id])->getInputDim();
    lrn_param = std::dynamic_pointer_cast<LRNLayer<T>>
                 (layers_map_[current_layer_id])->getLRNParam();

    // Process all the keywords in config
    if(isSectionKeywordExist(var, lrn_config_keywords)) {
      if (!var.compare("n")) {
        input_dim->n_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("c")) {
        input_dim->c_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("h")) {
        input_dim->h_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("w")) {
        input_dim->w_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("name")) {
        std::dynamic_pointer_cast<LRNLayer<T>>
          (layers_map_[current_layer_id])->setLayerName(val.c_str());
        name_id_map_[val] = current_layer_id;
        continue;
      }
      if (!var.compare("previous_layer_name")) {
        std::dynamic_pointer_cast<PoolingLayer<T>>
          (layers_map_[current_layer_id])->setPrevLayerName(val.c_str());
        continue;
      }
      if (!var.compare("lrn_mode")) {
        if (!val.compare("cross_channel_dim1"))
          lrn_param->mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
        continue;
      }
      if (!var.compare("local_size")) {
        lrn_param->local_size_ = atoi(val.c_str());
        continue;
      }
      if (!var.compare("alpha")) {
        lrn_param->alpha_ = atof(val.c_str());
        continue;
      }
      if (!var.compare("beta")) {
        lrn_param->beta_ = atof(val.c_str());
        continue;
      }
      if (!var.compare("k")) {
        lrn_param->k_ = atof(val.c_str());
        continue;
      }
    } else {
      std::cerr << var << "Keywords not exists" << std::endl;
      //TODO return error
    }
  }

  LOG(INFO) << *lrn_param;

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParseActivationConfig(const std::string &config_file) {
}

template <typename T>
int DNNMark<T>::ParseFullyConnectedConfig(const std::string &config_file) {
}

template <typename T>
int DNNMark<T>::ParseSoftmaxConfig(const std::string &config_file) {
}



template <typename T>
int DNNMark<T>::Initialize() {
  LOG(INFO) << "DNNMark: Initialize...";
  LOG(INFO) << "Running mode: " << run_mode_;
  LOG(INFO) << "Number of Layers: " << layers_map_.size();
  if (run_mode_ == STANDALONE) {
    for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
      LOG(INFO) << "Layer type: " << it->second->getLayerType();
      if (it->second->getLayerType() == CONVOLUTION) {
        LOG(INFO) << "DNNMark: Setup parameters of Convolution layer";
        std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)->Setup();
      }
      if (it->second->getLayerType() == POOLING) {
        LOG(INFO) << "DNNMark: Setup parameters of Pooling layer";
        std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)->Setup();
      }
      if (it->second->getLayerType() == LRN) {
        LOG(INFO) << "DNNMark: Setup parameters of LRN layer";
        std::dynamic_pointer_cast<LRNLayer<T>>(it->second)->Setup();
      }
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::RunAll() {
  if (run_mode_ == STANDALONE) {
    for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
      if (it->second->getLayerType() == CONVOLUTION) {
        std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
          ->ForwardPropagation();
        std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
          ->BackwardPropagation();
      }
      if (it->second->getLayerType() == POOLING) {
        std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
          ->ForwardPropagation();
        std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
          ->BackwardPropagation();
      }
      if (it->second->getLayerType() == LRN) {
        std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
          ->ForwardPropagation();
        std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
          ->BackwardPropagation();
      }
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Forward() {
  if (run_mode_ == STANDALONE) {
    for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
      if (it->second->getLayerType() == CONVOLUTION) {
        LOG(INFO) << "DNNMark: Running convolution forward: STARTED";
        std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
          ->ForwardPropagation();
        LOG(INFO) << "DNNMark: Running convolution forward: FINISHED";
      }
      if (it->second->getLayerType() == POOLING) {
        LOG(INFO) << "DNNMark: Running pooling forward: STARTED";
        std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
          ->ForwardPropagation();
        LOG(INFO) << "DNNMark: Running pooling forward: FINISHED";
      }
      if (it->second->getLayerType() == LRN) {
        LOG(INFO) << "DNNMark: Running LRN forward: STARTED";
        std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
          ->ForwardPropagation();
        LOG(INFO) << "DNNMark: Running LRN forward: FINISHED";
      }
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Backward() {
  if (run_mode_ == STANDALONE) {
    for (auto it = layers_map_.begin(); it != layers_map_.end(); it++) {
      if (it->second->getLayerType() == CONVOLUTION) {
        LOG(INFO) << "DNNMark: Running convolution backward: STARTED";
        std::dynamic_pointer_cast<ConvolutionLayer<T>>(it->second)
          ->BackwardPropagation();
        LOG(INFO) << "DNNMark: Running convolution backward: FINISHED";
      }
      if (it->second->getLayerType() == POOLING) {
        LOG(INFO) << "DNNMark: Running pooling backward: STARTED";
        std::dynamic_pointer_cast<PoolingLayer<T>>(it->second)
          ->BackwardPropagation();
        LOG(INFO) << "DNNMark: Running pooling backward: FINISHED";
      }
      if (it->second->getLayerType() == LRN) {
        LOG(INFO) << "DNNMark: Running LRN backward: STARTED";
        std::dynamic_pointer_cast<LRNLayer<T>>(it->second)
          ->BackwardPropagation();
        LOG(INFO) << "DNNMark: Running LRN backward: FINISHED";
      }
    }
  }
  return 0;
}


// Explicit instantiation
template class DNNMark<TestType>;

} // namespace dnnmark

