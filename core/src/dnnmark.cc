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
: run_mode_(NONE), handle_(), num_layers_added_(0) {}

template <typename T>
DNNMark<T>::DNNMark(int num_layers)
: run_mode_(NONE), handle_(num_layers), num_layers_added_(0) {}

template <typename T>
void DNNMark<T>::SetLayerParams(LayerType layer_type,
                    int current_layer_id,
                    const std::string &var,
                    const std::string &val) {
  DataDim *input_dim; 
  ConvolutionParam *conv_param;
  PoolingParam *pool_param;
  LRNParam *lrn_param;
  ActivationParam *activation_param;
  FullyConnectedParam *fc_param;
  SoftmaxParam *softmax_param;
  CHECK_GT(num_layers_added_, 0);

  switch(layer_type) {
    case CONVOLUTION: {
      // Obtain the data dimension and parameters variable
      // within specified layer
      input_dim = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      conv_param = std::dynamic_pointer_cast<ConvolutionLayer<T>>
                   (layers_map_[current_layer_id])->getConvParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the corresponding keywords in config
      if(isKeywordExist(var, conv_config_keywords)) {
        if (!var.compare("conv_mode")) {
          if (!val.compare("convolution"))
            conv_param->mode_ = CUDNN_CONVOLUTION;
          else if (!val.compare("cross_correlation"))
            conv_param->mode_ = CUDNN_CROSS_CORRELATION;
        } else if (!var.compare("num_output")) {
          conv_param->output_num_ = atoi(val.c_str());
        } else if (!var.compare("kernel_size")) {
          conv_param->kernel_size_h_ = atoi(val.c_str());
          conv_param->kernel_size_w_ = atoi(val.c_str());
        } else if (!var.compare("pad")) {
          conv_param->pad_h_ = atoi(val.c_str());
          conv_param->pad_w_ = atoi(val.c_str());
        } else if (!var.compare("stride")) {
          conv_param->stride_u_ = atoi(val.c_str());
          conv_param->stride_v_ = atoi(val.c_str());
        } else if (!var.compare("kernel_size_h")) {
          conv_param->kernel_size_h_ = atoi(val.c_str());
        } else if (!var.compare("kernel_size_w")) {
          conv_param->kernel_size_w_ = atoi(val.c_str());
        } else if (!var.compare("pad_h")) {
          conv_param->pad_h_ = atoi(val.c_str());
        } else if (!var.compare("pad_w")) {
          conv_param->pad_w_ = atoi(val.c_str());
        } else if (!var.compare("stride_h")) {
          conv_param->stride_u_ = atoi(val.c_str());
        } else if (!var.compare("stride_w")) {
          conv_param->stride_v_ = atoi(val.c_str());
        } else if (!var.compare("conv_fwd_pref")) {
          if (!val.compare("no_workspace"))
            conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
          else if (!val.compare("fastest"))
            conv_param->conv_fwd_pref_ = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
          else if (!val.compare("specify_workspace_limit"))
            conv_param->conv_fwd_pref_ =
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
        } else if (!var.compare("conv_bwd_filter_pref")) {
          if (!val.compare("no_workspace"))
            conv_param->conv_bwd_filter_pref_ =
              CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
          else if (!val.compare("fastest"))
            conv_param->conv_bwd_filter_pref_ =
              CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
          else if (!val.compare("specify_workspace_limit"))
            conv_param->conv_bwd_filter_pref_ =
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
        } else if (!var.compare("conv_bwd_data_pref")) {
          if (!val.compare("no_workspace"))
            conv_param->conv_bwd_data_pref_ =
              CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
          else if (!val.compare("fastest"))
            conv_param->conv_bwd_data_pref_ =
              CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
          else if (!val.compare("specify_workspace_limit"))
            conv_param->conv_bwd_data_pref_ =
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case CONVOLUTION
    case POOLING: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<PoolingLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      pool_param = std::dynamic_pointer_cast<PoolingLayer<T>>
                   (layers_map_[current_layer_id])->getPoolParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(isKeywordExist(var, pool_config_keywords)) {
        if (!var.compare("pool_mode")) {
          if (!val.compare("max"))
            pool_param->mode_ = CUDNN_POOLING_MAX;
          else if (!val.compare("avg_include_padding"))
            pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
          else if (!val.compare("avg_exclude_padding"))
            pool_param->mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        } else if (!var.compare("kernel_size")) {
          pool_param->kernel_size_h_ = atoi(val.c_str());
          pool_param->kernel_size_w_ = atoi(val.c_str());
        } else if (!var.compare("pad")) {
          pool_param->pad_h_ = atoi(val.c_str());
          pool_param->pad_w_ = atoi(val.c_str());
        } else if (!var.compare("stride")) {
          pool_param->stride_h_ = atoi(val.c_str());
          pool_param->stride_w_ = atoi(val.c_str());
        } else if (!var.compare("kernel_size_h")) {
          pool_param->kernel_size_h_ = atoi(val.c_str());
        } else if (!var.compare("kernel_size_w")) {
          pool_param->kernel_size_w_ = atoi(val.c_str());
        } else if (!var.compare("pad_h")) {
          pool_param->pad_h_ = atoi(val.c_str());
        } else if (!var.compare("pad_w")) {
          pool_param->pad_w_ = atoi(val.c_str());
        } else if (!var.compare("stride_h")) {
          pool_param->stride_h_ = atoi(val.c_str());
        } else if (!var.compare("stride_w")) {
          pool_param->stride_w_ = atoi(val.c_str());
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case POOLING
    case LRN: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<LRNLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      lrn_param = std::dynamic_pointer_cast<LRNLayer<T>>
                   (layers_map_[current_layer_id])->getLRNParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(isKeywordExist(var, lrn_config_keywords)) {
        if (!var.compare("lrn_mode")) {
          if (!val.compare("cross_channel_dim1"))
            lrn_param->mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
        } else if (!var.compare("local_size")) {
          lrn_param->local_size_ = atoi(val.c_str());
        } else if (!var.compare("alpha")) {
          lrn_param->alpha_ = atof(val.c_str());
        } else if (!var.compare("beta")) {
          lrn_param->beta_ = atof(val.c_str());
        } else if (!var.compare("k")) {
          lrn_param->k_ = atof(val.c_str());
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case LRN
    case ACTIVATION: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<ActivationLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      activation_param = std::dynamic_pointer_cast<ActivationLayer<T>>
                   (layers_map_[current_layer_id])->getActivationParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(isKeywordExist(var, activation_config_keywords)) {
        if (!var.compare("activation_mode")) {
          if (!val.compare("sigmoid"))
            activation_param->mode_ = CUDNN_ACTIVATION_SIGMOID;
          else if (!val.compare("relu"))
            activation_param->mode_ = CUDNN_ACTIVATION_RELU;
          else if (!val.compare("tanh"))
            activation_param->mode_ = CUDNN_ACTIVATION_TANH;
          else if (!val.compare("relu"))
            activation_param->mode_ = CUDNN_ACTIVATION_CLIPPED_RELU;
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case ACTIVATION
    case FC: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<FullyConnectedLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      fc_param = std::dynamic_pointer_cast<FullyConnectedLayer<T>>
                 (layers_map_[current_layer_id])->getFullyConnectedParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(isKeywordExist(var, fc_config_keywords)) {
        if (!var.compare("num_output")) {
          fc_param->output_num_ = atoi(val.c_str());
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case FC
    case SOFTMAX: {
      // Obtain the data dimension and parameters variable within layer class
      input_dim = std::dynamic_pointer_cast<SoftmaxLayer<T>>
                  (layers_map_[current_layer_id])->getInputDim();
      softmax_param = std::dynamic_pointer_cast<SoftmaxLayer<T>>
                 (layers_map_[current_layer_id])->getSoftmaxParam();

      if(isKeywordExist(var, data_config_keywords))
        break;

      // Process all the keywords in config
      if(isKeywordExist(var, softmax_config_keywords)) {
        if (!var.compare("softmax_algo")) {
          if (!val.compare("fast"))
            softmax_param->algo_ = CUDNN_SOFTMAX_FAST;
          else if (!val.compare("accurate"))
            softmax_param->algo_ = CUDNN_SOFTMAX_ACCURATE;
          else if (!val.compare("log"))
            softmax_param->algo_ = CUDNN_SOFTMAX_LOG;
        }
        if (!var.compare("softmax_mode")) {
          if (!val.compare("instance"))
            softmax_param->mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
          else if (!val.compare("channel"))
            softmax_param->mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
        }
      } else {
        LOG(FATAL) << var << ": Keywords not exists" << std::endl;
      }
      break;
    } // End of case SOFTMAX
    default: {
      LOG(WARNING) << "NOT supported layer";
      break;
    } // End of case default

  }

  // Set data configuration at last, since all layers share same parameters
  if(isKeywordExist(var, data_config_keywords)) {
    if (!var.compare("n")) {
      input_dim->n_ = atoi(val.c_str());
    } else if (!var.compare("c")) {
      input_dim->c_ = atoi(val.c_str());
    } else if (!var.compare("h")) {
      input_dim->h_ = atoi(val.c_str());
    } else if (!var.compare("w")) {
      input_dim->w_ = atoi(val.c_str());
    } else if (!var.compare("name")) {
      layers_map_[current_layer_id]->setLayerName(val.c_str());
      name_id_map_[val] = current_layer_id;
    } else if (!var.compare("previous_layer")) {
      layers_map_[current_layer_id]->setPrevLayerName(val.c_str());
    }
  }
}

template <typename T>
int DNNMark<T>::ParseAllConfig(const std::string &config_file) {
  // TODO: use multithread in the future
  // Parse DNNMark specific config
  ParseGeneralConfig(config_file);

  // Parse Convolution specific config
  ParseSpecifiedConfig(config_file, CONVOLUTION);

  // Parse Pooling specific config
  ParseSpecifiedConfig(config_file, POOLING);

  // Parse LRN specific config
  ParseSpecifiedConfig(config_file, LRN);

  // Parse Activation specific config
  ParseSpecifiedConfig(config_file, ACTIVATION);

  // Parse FullyConnected specific config
  ParseSpecifiedConfig(config_file, FC);

  // Parse Softmax specific config
  ParseSpecifiedConfig(config_file, SOFTMAX);
}

template <typename T>
int DNNMark<T>::ParseGeneralConfig(const std::string &config_file) {
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

    // Process all the keywords in config
    if(isKeywordExist(var, dnnmark_config_keywords)) {
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
      LOG(FATAL) << var << ": Keywords not exists" << std::endl;
    }
  }

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::ParseSpecifiedConfig(const std::string &config_file,
                                     LayerType layer_type) {
  std::ifstream is;
  is.open(config_file.c_str(), std::ifstream::in);

  // Parse DNNMark config
  std::string s;
  int current_layer_id;
  bool is_specified_section = false;
  std::string section;
  switch(layer_type) {
    case CONVOLUTION: {
      section.assign("[Convolution]");
      break;
    }
    case POOLING: {
      section.assign("[Pooling]");
      break;
    }
    case LRN: {
      section.assign("[LRN]");
      break;
    }
    case ACTIVATION: {
      section.assign("[Activation]");
      break;
    }
    case FC: {
      section.assign("[FullyConnected]");
      break;
    }
    case SOFTMAX: {
      section.assign("[Softmax]");
      break;
    }
    default: {
      break;
    }
  }
  LOG(INFO) << "Search and parse layer "
            << section << " configuration";
  while (!is.eof()) {
    // Obtain the string in one line
    std::getline(is, s);
    TrimStr(&s);

    // Check the specific configuration section markers
    if (isCommentStr(s) || isEmptyStr(s)){
      continue;
    } else if (isSpecifiedSection(s, section.c_str())) {
      LOG(INFO) << "Add "
                << section
                << " layer";
      is_specified_section = true;
      // Create a layer in the main class
      current_layer_id = num_layers_added_;
      if (layer_type == CONVOLUTION)
        layers_map_.emplace(current_layer_id,
          std::make_shared<ConvolutionLayer<T>>(this));
      else if (layer_type == POOLING)
        layers_map_.emplace(current_layer_id,
          std::make_shared<PoolingLayer<T>>(this));
      else if (layer_type == LRN)
        layers_map_.emplace(current_layer_id,
          std::make_shared<LRNLayer<T>>(this));
      else if (layer_type == ACTIVATION)
        layers_map_.emplace(current_layer_id,
          std::make_shared<ActivationLayer<T>>(this));
      else if (layer_type == FC)
        layers_map_.emplace(current_layer_id,
          std::make_shared<FullyConnectedLayer<T>>(this));
      else if (layer_type == SOFTMAX)
        layers_map_.emplace(current_layer_id,
          std::make_shared<SoftmaxLayer<T>>(this));
      layers_map_[current_layer_id]->setLayerId(current_layer_id);
      layers_map_[current_layer_id]->setLayerType(layer_type);
      num_layers_added_++;
      continue;
    } else if (isSection(s) && is_specified_section) {
      break;
    } else if (!is_specified_section) {
      continue;
    }

    // Obtain the acutal variable and value
    std::string var;
    std::string val;
    SplitStr(s, &var, &val);

    // Obtain the data dimension and parameters variable within layer class
    SetLayerParams(layer_type,
                   current_layer_id,
                   var, val);
  }

  is.close();
  return 0;
}

template <typename T>
int DNNMark<T>::Initialize() {
  LOG(INFO) << "DNNMark: Initialize...";
  LOG(INFO) << "Running mode: " << run_mode_;
  LOG(INFO) << "Number of Layers: " << layers_map_.size();
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
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Setup parameters of Activation layer";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Setup parameters of Fully Connected layer";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)->Setup();
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Setup parameters of Softmax layer";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)->Setup();
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::RunAll() {
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
    if (it->second->getLayerType() == ACTIVATION) {
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == FC) {
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->BackwardPropagation();
    }
    if (it->second->getLayerType() == SOFTMAX) {
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->ForwardPropagation();
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->BackwardPropagation();
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Forward() {
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
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Running Activation forward: STARTED";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Activation forward: FINISHED";
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Running FullyConnected forward: STARTED";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running FullyConnected forward: FINISHED";
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Running Softmax forward: STARTED";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->ForwardPropagation();
      LOG(INFO) << "DNNMark: Running Softmax forward: FINISHED";
    }
  }
  return 0;
}

template <typename T>
int DNNMark<T>::Backward() {
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
    if (it->second->getLayerType() == ACTIVATION) {
      LOG(INFO) << "DNNMark: Running Activation backward: STARTED";
      std::dynamic_pointer_cast<ActivationLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Activation backward: FINISHED";
    }
    if (it->second->getLayerType() == FC) {
      LOG(INFO) << "DNNMark: Running FullyConnected backward: STARTED";
      std::dynamic_pointer_cast<FullyConnectedLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running FullyConnected backward: FINISHED";
    }
    if (it->second->getLayerType() == SOFTMAX) {
      LOG(INFO) << "DNNMark: Running Softmax backward: STARTED";
      std::dynamic_pointer_cast<SoftmaxLayer<T>>(it->second)
        ->BackwardPropagation();
      LOG(INFO) << "DNNMark: Running Softmax backward: FINISHED";
    }
  }
  return 0;
}


// Explicit instantiation
template class DNNMark<TestType>;

} // namespace dnnmark

