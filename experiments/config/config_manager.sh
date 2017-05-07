#! /bin/bash

LIST=( conv pool lrn activation fc softmax )

for bm in ${LIST[@]}
do
  echo ${bm}
  cd ${bm}
  cp ../../../config_example/${bm}_config.dnnmark ./${bm}_config_100.dnnmark
  cp ${bm}_config_100.dnnmark ${bm}_config_10.dnnmark
  cp ${bm}_config_100.dnnmark ${bm}_config_50.dnnmark
  sed -i -e 's/n=[0-9]\+/n=10/g' ${bm}_config_10.dnnmark
  sed -i -e 's/n=[0-9]\+/n=50/g' ${bm}_config_50.dnnmark
  cd ..
done
