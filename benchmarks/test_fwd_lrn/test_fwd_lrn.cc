#include <iostream>
#include <gflags/gflags.h>
#include "common.h"
#include "dnnmark.h"

using namespace dnnmark;

DEFINE_string(config, "",
    "The self defined DNN config file.");
DEFINE_int32(debug_info, 0,
    "The debug info switch to turn on/off debug information.");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = FLAGS_debug_info;
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseDNNMarkConfig(FLAGS_config);
  dnnmark.ParseLRNConfig(FLAGS_config);
  dnnmark.Initialize();
  dnnmark.Forward();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
