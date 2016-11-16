#include <iostream>
#include <gflags/gflags.h>
#include "common.h"
#include "dnnmark.h"

using namespace dnnmark;

DEFINE_string(config, "",
    "The self defined DNN config file.");
DEFINE_int32(debuginfo, 0,
    "The debug info switch to turn on/off debug information.");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = FLAGS_debuginfo;
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseSpecifiedConfig(FLAGS_config, LRN);
  dnnmark.Initialize();
  dnnmark.Backward();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
