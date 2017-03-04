# DNNMark
Configurable benchmark suite of Deep Neural Networks

DNNMark is a highly configurable, extensible, and flexible Deep Neural Network benchmark framework. In this benchmark suite, each of DNN primitive workloads can be easily invoked separately, without any sacrifice on configurability. One can specify any type of scenarios for benchmarking on an algorithm-specific level. Second, the actual measurement can be done during the execution of any specific kernel. Our framework allows us to ignore the setup stages and focus on only the training steps. Unlike other deep learning frameworks, attaching a real database
for training purposes is not mandatory anymore. This capability will greatly aid the computer architecture community, which is
more interested in designing/tuning hardware/software, and less interested in the details or configuration of the deep neural net.

Depending on the specific configuration, deep neural networks can involve combinations of DNN primitives. A model composed of two or more primitive functions may be more desirable in terms of performance evaluation. In such cases, a composed model rather than standalone primitives, are preferred. To provide this capability, DNNmark can be extended to more sophisticated DNN models, where layers are connected to, and dependent upon, each other.

## Configurability
This frame work provides configurability in both general and algorithm specific parameters. Users can do this through a plain-text configuration file. Several examples are provided inside config_example directory.
## Extensibility
New DNN models/scenarios can be easily built for benchmarking through configuration files
## Convenience
Designing benchmarks takes little effort thanks to its centralized library
## Diversity
DNNMark contains commonly-used DNN primitives and also provides an easy approach to compose a model

# Features

1. Configurable
2. Provide insights of GPU execution behavior using detailed GPU metrics/events
3. Separatable or combined DNN primitives benchmarking

# Supported DNN primitives:

1. Convolution forward and backward
2. Pooling forward and backward
3. LRN forward and backward
4. Activation forward and backward
5. Fully Connected forward and backward
6. Softmax forward and backward
7. Batch Normalization forward and backward

# Build and Usage

## OS, Library, and Software Prerequisite
OS: Ubuntu

CUDA related library: CUDA tool kit v8.0; CuDNN v8.0

Other Software: CMake; g++

## Build
After you download and unzip the DNNMark, you should go to its root directory and edit `setup.sh` to set up path to cuDNN. And then run `./setup.sh`. This will create a build directory and run cmake automatically. To build the code, go to build directory `build` and run `make`

## Usage
To run the benchmarks that have been built, go to the directory `build` and you will see a directory `benchmarks`. Go inside and select the benchmark you want to run. Run command `./[name of benchmark] -config [path to config file] -debuginfo [1 or 0]` to execute the benchmark

# For Contributors
1. Fork the repository to your own remote repository.
2. Git clone the repository: `git clone git@github.com/your_account_name/DNNMark.git`
3. Add this project as an upstream to your local repository by `git remote add upstream https://github.com/doody1986/DNNMark.git`. You can use `git remote -v` to view the upstream.
4. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to your own remote repository (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push --set-upstream origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest change from upstream develop branch to your own origin develop branch: 1) `git checkout develop` 2) `git pull upstream develop` 3) `git push origin develop`
6. Since that you have the latest change in your own origin develop branch from upstream one, now you can update your own feature branch on the your own remote repository by: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop` 3) `git push origin your-own-feature-branch`
7. Make a pull request from your own feature branch on your own remote repository on github to the develop branch of this repository.
8. After the pull request is merged, you can delete your own feature branch by 1) `git push origin --delete your-own-feature-branch` to delete the remote branch and 2) `git branch -d your-own-feature-branch` to delete your local branch.
9. More instructions on using fork can be found [here](https://help.github.com/articles/fork-a-repo/).
