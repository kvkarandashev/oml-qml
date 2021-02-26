from joblib import Parallel, delayed
import os, subprocess

num_procs_name="OML_NUM_PROCS"

def oml_num_procs():
    try:
        return int(os.environ[num_procs_name])
    except LookupError:
        return 1

def embarassingly_parallel(func, array, other_args, disable_openmp=True):
    if type(other_args) is not tuple:
        other_args=(other_args,)
    if disable_openmp:
        from .utils import dump2pkl, loadpkl
        dump_filename=subprocess.check_output(["mktemp", "-p", os.getcwd()])[:-1]
        dump2pkl({"func" : func, "array" : array, "args" : other_args}, dump_filename)
        subprocess.run(["oml_emb_paral_func_exec.sh", dump_filename])
        output=loadpkl(dump_filename)
        subprocess.run(["rm", '-f', dump_filename])
        return output
    else:
        return embarassingly_parallel_openmp_enabled(func, array, other_args)
    
def embarassingly_parallel_openmp_enabled(func, array, args):
    return Parallel(n_jobs=oml_num_procs(), backend="multiprocessing")(delayed(func)(el, *args) for el in array)
