#! /bin/bash

LIST=( alexnet )

for bm in ${LIST[@]}
do
  echo ${bm}
  cd ${bm}
  #Write down your own management code here
  cd ..
done
