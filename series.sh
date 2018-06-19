#!/bin/bash


arr=( $(seq 10 2 200) )

echo "Run ${#arr[@]} times"

grepp()
{
    echo $1 | grep "wgrad"
}

for N in ${arr[@]};do

    conf=$(cat <<SETVAR
    [DNNMark]
    run_mode=standalone

    [Convolution]
    name=conv1
    n=$N
    c=3
    h=256
    w=256
    previous_layer=null
    conv_mode=convolution
    num_output=32
    kernel_size=5
    pad=2
    stride=1
    conv_fwd_pref=fastest
    conv_bwd_filter_pref=fastest
    conv_bwd_data_pref=fastest
SETVAR
)
    echo "$conf" > conf_conv.dnnmark
    s="$(nvprof --profile-api-trace none --unified-memory-profiling off ./build/benchmarks/test_bwd_conv/dnnmark_test_bwd_conv -config conf_conv.dnnmark 2>&1)"
    echo "$s" | grep "wgrad" | xargs -x echo "$N $x" | awk '{ print $1 "\t" $2 "\t" $3 }'
done
