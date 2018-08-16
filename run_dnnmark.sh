#!/bin/bash

# API for DNNMark
# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

usage=$(cat <<USAGEBLOCK
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
                [-b <benchmark executable, default=test_bwd_conv>]
                [ --activation - add RELU activation]
                [ --debug - debug info ]
                [ --help  - usage info ]

Configuration saved in temporary file conf_tmp.dnnmark
USAGEBLOCK
)

config_file="conf_tmp.dnnmark"
# Defaults
N=64
C=3
H=256
W=256
K=128
S=3
U=1
P=0
BENCH="test_bwd_conv"
ACTIVATION=""
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
        -b)
            BENCH="$2";shift;
            ;;
        --algo)
            CBFA="$2";shift;
            ;;
        --debug)
            debug=1
            ;;
        --relu)
            ACTIVATION=$(cat <<SETACT
[Activation]
name=relu1
previous_layer=conv1
activation_mode=relu
SETACT
)
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

conf=$(cat <<SETVAR
[DNNMark]
run_mode=composed

[Convolution]
name=conv1
n=$N
c=$C
h=$H
w=$W
previous_layer=null
conv_mode=convolution
num_output=$K
kernel_size=$S
pad=$P
stride=$U
conv_fwd_pref=fastest
conv_bwd_filter_pref=fastest
conv_bwd_data_pref=fastest
$CUDNN_CBFA

$ACTIVATION

SETVAR
)

echo "$conf" > $config_file

./build/benchmarks/"$BENCH"/dnnmark_"$BENCH" -config $config_file -debuginfo $debug -warmup 1



