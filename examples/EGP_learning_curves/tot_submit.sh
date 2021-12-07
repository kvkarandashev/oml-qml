#!/bin/bash

global_name=$1

quant_names=("reduction_lithium" "oxidation_lithium")

script_name=SLATM_EGP # SLATM_EGP FJK_EGP

for quant_name in ${quant_names[@]}
do
    job_name=${global_name}_$quant_name
    rm -Rf data_$job_name
    spython $script_name.py $job_name $quant_name
done
