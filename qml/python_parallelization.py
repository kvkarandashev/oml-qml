from joblib import Parallel, delayed
import os, subprocess

num_procs_name="OML_NUM_PROCS"

def oml_num_procs():
    try:
        return int(os.environ[num_procs_name])
    except:
        return 1

def embarassingly_parallel(func, array, other_args, disable_openmp=True):
    if type(other_args) is not tuple:
        other_args=(other_args,)
    if disable_openmp:
        import pickle as pl
        dump_filename=subprocess.check_output(["mktemp", "-p", os.getcwd()])[:-1]
        plf=open(dump_filename, 'wb')
        pl.dump({"func" : func, "array" : array, "args" : other_args}, plf)
        plf.close()
        subprocess.run(["oml_emb_paral_func_exec.sh", dump_filename])
        plf=open(dump_filename, 'rb')
        output=pl.load(plf)
        plf.close()
        subprocess.run(["rm", '-f', dump_filename])
        return output
    else:
        return embarassingly_parallel_openmp_enabled(func, array, other_args)
    
def embarassingly_parallel_openmp_enabled(func, array, args):
    return Parallel(n_jobs=oml_num_procs(), backend="multiprocessing")(delayed(func)(el, *args) for el in array)
