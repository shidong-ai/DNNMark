#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseSpecifiedConfig(FLAGS_config, LRN);
  dnnmark.Initialize();
  dnnmark.Backward();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
