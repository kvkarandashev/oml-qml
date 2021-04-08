from joblib import Parallel, delayed
import os, subprocess
from .utils import mktmpdir, rmdir

num_procs_name="OML_NUM_PROCS"

def oml_num_procs():
    try:
        return int(os.environ[num_procs_name])
    except LookupError:
        return 1

def create_func_exec(dump_filename, tmpdir):
    contents='''#!/bin/bash
    dump_filename='''+dump_filename+'''
for var in "OMP_NUM_THREADS" "OPENBLAS_NUM_THREADS" "MKL_NUM_THREADS" "VECLIB_MAXIMUM_THREADS" "NUMEXPR_NUM_THREADS"
do
	eval "export $var=1"
done

pyscript='''+tmpdir+'''/tmppyscript.py

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

python $pyscript'''
    exec_scr="./"+tmpdir+"/tmpscript.sh"
    output=open(exec_scr, 'w')
    output.write(contents)
    output.close()
    subprocess.run(["chmod", "+x", exec_scr])
    return exec_scr

def embarassingly_parallel(func, array, other_args, disable_openmp=True):
    if type(other_args) is not tuple:
        other_args=(other_args,)
    if disable_openmp:
        from .utils import dump2pkl, loadpkl
        tmpdir=mktmpdir()
        dump_filename=tmpdir+"/dump.pkl"
        dump2pkl({"func" : func, "array" : array, "args" : other_args}, dump_filename)
        exec_scr=create_func_exec(dump_filename, tmpdir)
        subprocess.run([exec_scr, dump_filename])
        output=loadpkl(dump_filename)
        rmdir(tmpdir)
        return output
    else:
        return embarassingly_parallel_openmp_enabled(func, array, other_args)
    
def embarassingly_parallel_openmp_enabled(func, array, args):
    return Parallel(n_jobs=oml_num_procs(), backend="multiprocessing")(delayed(func)(el, *args) for el in array)
