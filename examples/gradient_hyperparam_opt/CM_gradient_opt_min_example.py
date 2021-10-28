# Minimal example for calculating energy corrections.
# Note that it requireds presence of QM7bT directory, that can be created with
# procedures from qm7b_t_format_specs.


# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve_wcheck, import_quantity_array
from qm9_format_specs import Quantity
import os, random, qml
import numpy as np
from qml import Compound
from qml.kernels_wders import gaussian_sym_kernel_conv_wders, CM_kernel, CM_kernel_input
from qml.hyperparameter_optimization import min_sep_IBO_random_walk_optimization
from qml.utils import dump2pkl

def_float_format='{:.8E}'

quant_name='HOMO eigenvalue'
seed=1

use_Gauss=True

# Replace with path to QM9 directory
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

train_num=1000
check_num=2000

delta_learning_params=Delta_learning_parameters(use_delta_learning=True)

xyz_list=dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)

dump2pkl(xyz_list, "shuffled_list.pkl")

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

def get_quants_comps(xyz_list, quantity, dl_params):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=[Compound(xyz_file) for xyz_file in xyz_list]
    return comps, np.array(quant_vals)

quant=Quantity(quant_name)
training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, delta_learning_params)

optimized_hyperparams=min_sep_IBO_random_walk_optimization(training_comps, training_quants, init_lambda=1e-6, init_param_guess=np.array([1.0, 100.0]), max_stagnating_iterations=8,
                                    hyperparam_red_type="default", randomized_iterator_kwargs={"default_step_magnitude" : 0.05}, iter_dump_name_add="test_min_CM",
                                    additional_BFGS_iters=8, iter_dump_name_add_BFGS="test_min_BFGS", sym_kernel_func=gaussian_sym_kernel_conv_wders, kernel_input_converter=CM_kernel_input)



sigmas=optimized_hyperparams["sigmas"]
lambda_val=optimized_hyperparams["lambda_val"]

print("Finalized parameters:", sigmas)
print("Finalized lambda:", lambda_val)

sigma=sigmas[0]

K_train=CM_kernel(training_comps, training_comps, sigma, use_Gauss=use_Gauss)
K_train[np.diag_indices_from(K_train)]+=lambda_val
alphas=np_cho_solve_wcheck(K_train, training_quants, eigh_rcond=1e-9)
del(K_train)

check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, delta_learning_params)
K_check=CM_kernel(check_comps, training_comps, sigma, use_Gauss=use_Gauss)
predicted_quants=np.dot(K_check, alphas)
MAE=np.mean(np.abs(predicted_quants-check_quants))
print("Quantity: ", quant_name, ", MAE:", MAE)
