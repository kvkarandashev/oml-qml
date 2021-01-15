from joblib import Parallel, delayed
import os

num_procs_name="OML_NUM_PROCS"

def oml_num_procs():
    try:
        return int(os.environ[num_procs_name])
    except:
        return 1

def embarassingly_parallel(func, array, *args):
    return Parallel(n_jobs=oml_num_procs(), backend="multiprocessing")(delayed(func)(el, *args) for el in array)
