#!/bin/bash

for var in "OMP_NUM_THREADS" "OPENBLAS_NUM_THREADS" "MKL_NUM_THREADS" "VECLIB_MAXIMUM_THREADS" "NUMEXPR_NUM_THREADS"
do
	eval "export $var=1"
done

dump_filename=$1

pyscript=$(mktemp -p $(pwd))

cat > $pyscript << EOF
import pickle as pl
from qml.python_parallelization import embarassingly_parallel_openmp_enabled

plf=open("$dump_filename", 'rb')
executed=pl.load(plf)
plf.close()

plf=open("$dump_filename", 'wb')
pl.dump(embarassingly_parallel_openmp_enabled(executed["func"], executed["array"], executed["args"]), plf)
plf.close()

EOF

python $pyscript
rm -f $pyscript
