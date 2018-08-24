#!/bin/bash

# Wrapper API for DNNMark
# Uses conf_multiconv_mod.dnntemplate configuration file template.
# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

usage=$(cat <<- USAGEBLOCK
Run DNNMark with parameters from CLI.
Usage:
$(basename $0)  [-n <number of images, batch size>]
                [-c <number of channels in input images>]
                [-h <height of input images>]
                [-w <widht of input images>]
                [-k <number of filters, ouput channels>]
                [-s <size of filter kernel>]
                [-u <stride>]
                [-p <padding>]
                [ --algo <cudnnConvolutionBwdFilterAlgo_t> - cuDNN algorithm for backward filter convolution]
                [ --iter <int> - number of FWD+BWD passes to measure time]
                [ --debug - debug info ]
                [ --help  - usage info ]

Configuration saved in temporary file conf_tmp.dnnmark
USAGEBLOCK
)

template="conf_multiconv_mod.dnntemplate"
config_file="conf_tmp.dnnmark"
# Defaults
N=64
C=3
H=32
W=32
K=128
S=3
U=1
P=1
BENCH="test_composed_model"
ITER=1
debug=0


while test $# -gt 0; do
    case "$1" in
        --help)
            echo "$usage"
            exit 0
            ;;
        -n)
            N="$2";shift;
            ;;
        -c)
            C="$2";shift;
            ;;
        -h)
            H="$2";shift;
            ;;
        -w)
            W="$2";shift;
            ;;
        -k)
            K="$2";shift;
            ;;
        -s)
            S="$2";shift;
            ;;
        -u)
            U="$2";shift;
            ;;
        -p)
            P="$2";shift;
            ;;
        --algo)
            CBFA="$2";shift;
            ;;
        --iter)
            ITER="$2";shift;
            ;;
        --debug)
            debug=1
            ;;
        --)
            shift
            break;;
        -*)
            echo "Unknown option $1";
            echo "$usage"
            exit 1
            ;;
        *)
            break;;
    esac
    shift
done

if [ $CBFA ];then
    CUDNN_CBFA="algo=$CBFA"
fi

conf="$(echo EOF;cat $template;echo EOF)"

eval "cat <<$conf" >$config_file
echo "Config: ---"
cat $config_file
echo "-----------"
echo "Benchmark: $BENCH"
echo "Iterations:$ITER"

./build/benchmarks/"$BENCH"/dnnmark_"$BENCH" -config $config_file --warmup 1 --iterations $ITER --debuginfo $debug



