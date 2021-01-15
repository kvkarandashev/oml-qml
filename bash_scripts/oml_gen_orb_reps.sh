#!/bin/bash

export OML_NUM_PROCS=$OMP_NUM_THREADS
for var in "OMP_NUM_THREADS" "OPENBLAS_NUM_THREADS" "MKL_NUM_THREADS" "VECLIB_MAXIMUM_THREADS" "NUMEXPR_NUM_THREADS"
do
	eval "export $var=1"
done

arr_filename=$1

rep_filename=$2

cat > temp.py.tmp << EOF
import numpy as np
import pickle as pl

plf=open("$arr_filename", 'rb')
comp_list=pl.load(plf)
plf.close()

plf=open("$rep_filename", 'rb')
rep=pl.load(plf)
plf.close()

comp_list.generate_orb_reps(rep)
plf=open("$arr_filename", 'wb')
pl.dump(comp_list, plf)
plf.close()

EOF
python temp.py.tmp
