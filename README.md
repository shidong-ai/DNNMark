# DNNMark
Configurable benchmark suite of Deep Neura Networks

DNNMark is a highly configurable, extensible, and flexible Deep Neural Network benchmark framework. In this benchmark suite, each of DNN primitive workloads can be easily invoked separately, without any sacrifice on configurability. One can specify any type of scenarios for benchmarking on an algorithm-specific level. Second, the actual measurement can be done during the execution of any specific kernel. Our framework allows us to ignore the setup stages and focus on only the training steps. Unlike other deep learning frameworks, attaching a real database
for training purposes is not mandatory anymore. This capability will greatly aid the computer architecture community, which is
more interested in designing/tuning hardware/software, and less interested in the details or configuration of the deep neural net.

Depending on the specific configuration, deep neural networks can involve combinations of DNN primitives. A model composed of two or more primitive functions may be more desirable in terms of performance evaluation. In such cases, a composed model rather than standalone primitives, are preferred. To provide this capability, DNNmark can be extended to more sophisticated DNN models, where layers are connected to, and dependent upon, each other.

## Configurability
This frame work provides configurability in both general and algorithm specific parameters. Users can do this through a plain-text configuration file. Several examples are provided inside config_example directory.
## Extensibility
New DNN models/scenarios can be built for benchmarking through 
## Convenience
Designing benchmarks takes little effort thanks to its centralized library
## Diversity
DNNMark contains commonly-used DNN primitives and also provides an easy approach to compose a model

# Features

1. Configurable
2. Provide detailed GPU metrics
3. Separatable DNN primitives benchmarking
4. Building either standalone or composed benchmarks through plain-text configuration files

# Supported DNN primitives:

1. Convolution forward and backward
2. Pooling forward and backward
3. LRN forward and backward
4. Activation forward and backward
5. Fully Connected forward and backward
6. Softmax forward and backward

# Build and Usage

## OS, Library, and Software Prerequisite
OS:
  Ubuntu
CUDA related library:
  CUDA tool kit v8.0
  CuDNN v8.0
Other Software:
  CMake
  g++

## Build
After you download and unzip the DNNMark, you should go to its root directory and edit `setup.sh` to set up path to cuDNN. And then run `./setup.sh`. This will create a build directory and run cmake automatically. To build the code, go to build directory `build` and run `make`

## Usage
To run the benchmarks that have been built, go to the directory `build` and you will see a directory `benchmarks`. Go inside and select the benchmark you want to run. Run command `./[name of benchmark] -config [path to config file] -debuginfo [1 or 0]` to execute the benchmark
