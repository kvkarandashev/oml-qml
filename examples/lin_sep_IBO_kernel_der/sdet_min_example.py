# Minimal example for calculating energy corrections.
# Note that it requireds presence of QM7bT directory, that can be created with
# procedures from qm7b_t_format_specs.


# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve_wcheck, import_quantity_array
from qm7b_t_format_specs import Quantity
import os, random, qml
import numpy as np
from qml import OML_compound_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import gauss_sep_IBO_sym_kernel, gauss_sep_IBO_kernel
from qml.hyperparameter_optimization import min_sep_IBO_random_walk_optimization
from qml.utils import dump2pkl

def_float_format='{:.8E}'

quant_name='MP2/cc-pVTZ'
seed=1

basis='sto-3g'
max_angular_momentum=2

# Replace with path to QM9 directory
QM7bT_dir=os.environ["DATA"]+"/QM7bT_reformatted"

train_num=1000
check_num=2000

use_Gauss=True
keep_init_lambda=False

delta_learning_params=Delta_learning_parameters(use_delta_learning=True)

xyz_list=dirs_xyz_list(QM7bT_dir)
random.seed(seed)
random.shuffle(xyz_list)

dump2pkl(xyz_list, "shuffled_list.pkl")

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=max_angular_momentum, use_Fortran=True)


def get_quants_comps(xyz_list, quantity, dl_params, oml_representation_parameters):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=OML_compound_list_from_xyzs(xyz_list, calc_type="HF", basis=basis)
    comps.generate_orb_reps(oml_representation_parameters)
    return comps, np.array(quant_vals)

quant=Quantity(quant_name)
training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, delta_learning_params, oml_representation_parameters)

optimized_hyperparams=min_sep_IBO_random_walk_optimization(training_comps, training_quants, init_lambda=1e-6, use_Gauss=use_Gauss, max_stagnating_iterations=8, num_kfolds=128,
                                    hyperparam_red_type="ang_mom_classified", randomized_iterator_kwargs={"keep_init_lambda" : keep_init_lambda, "default_step_magnitude" : 0.25}, iter_dump_name_add="test_min",
                                    rep_params=oml_representation_parameters)



inv_sq_width_params=optimized_hyperparams["inv_sq_width_params"]
lambda_val=optimized_hyperparams["lambda_val"]

print("Finalized parameters:", inv_sq_width_params)
print("Finalized lambda:", lambda_val)

K_train=gauss_sep_IBO_sym_kernel(training_comps, inv_sq_width_params)
K_train[np.diag_indices_from(K_train)]+=lambda_val
alphas=np_cho_solve_wcheck(K_train, training_quants, eigh_rcond=1e-9)
del(K_train)

check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, delta_learning_params, oml_representation_parameters)
K_check=gauss_sep_IBO_kernel(check_comps, training_comps, inv_sq_width_params)
predicted_quants=np.dot(K_check, alphas)
MAE=np.mean(np.abs(predicted_quants-check_quants))
print("Quantity: ", quant_name, ", MAE:", MAE)
