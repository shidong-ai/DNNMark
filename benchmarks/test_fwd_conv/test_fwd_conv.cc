#include <iostream>
#include <gflags/gflags.h>
#include "common.h"
#include "dnnmark.h"

using namespace dnnmark;

DEFINE_string(config, "",
    "The self defined DNN config file.");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
    DNNMark<TestType> dnnmark;
  dnnmark.ParseDNNMarkConfig(FLAGS_config);
  dnnmark.ParseConvolutionConfig(FLAGS_config);
  dnnmark.Initialize();
  dnnmark.Forward();
  return 0;
}
