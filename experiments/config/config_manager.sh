#! /bin/bash

LIST=( conv pool lrn activation fc softmax )

for bm in ${LIST[@]}
do
  echo ${bm}
  cd ${bm}
  #Write down your own management code here
  cp ../alexnet/alexnet_config_128.dnnmark ${bm}_config_128.dnnmark
  cp ../alexnet/alexnet_config_16.dnnmark ${bm}_config_16.dnnmark
  cp ../alexnet/alexnet_config_64.dnnmark ${bm}_config_64.dnnmark
  cd ..
done
