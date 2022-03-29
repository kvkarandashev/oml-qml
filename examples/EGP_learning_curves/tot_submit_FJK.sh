#!/bin/bash

global_name=$1

quant_names=("reduction_lithium" "oxidation_lithium")

for quant_name in ${quant_names[@]}
do
    for method in FJK OFD
    do
        job_name=${global_name}_${quant_name}_${method}
        rm -Rf data_$job_name
        spython FJK_EGP.py $job_name $quant_name $method
    done
done
