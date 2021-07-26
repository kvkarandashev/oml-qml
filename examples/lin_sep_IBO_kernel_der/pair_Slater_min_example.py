# Minimal example that calculates model MAEs for ionization potentials and electron affinities.
# Note that it requires having a directory with QM7b xyz files; it can be created with procedured from qm7b_format_specs.


# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve_wcheck, import_quantity_array
from qm7b_format_specs import Quantity
import os, random, qml
import numpy as np
from qml import OML_Slater_pair_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import lin_sep_IBO_sym_kernel, lin_sep_IBO_kernel
from qml.hyperparameter_optimization import  min_sep_IBO_MSE_random_walk
from qml.utils import dump2pkl

def_float_format='{:.8E}'

quant_names=['ip', 'ea']
charges=[1, -1]
seed=1

# Replace with path to QM7b directory
QM7b_dir=os.environ["DATA"]+"/QM7b/xyzs"

train_num=100
check_num=200

delta_learning_params=Delta_learning_parameters(use_delta_learning=True)

xyz_list=dirs_xyz_list(QM7b_dir)
random.seed(seed)
random.shuffle(xyz_list)

dump2pkl(xyz_list, "shuffled_list.pkl")

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=1, use_Fortran=True)


def get_quants_comps(xyz_list, quantity, dl_params, oml_representation_parameters, charge):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=OML_Slater_pair_list_from_xyzs(xyz_list, calc_type="HF", second_calc_type="UHF",
                                second_charge=charge, basis='sto-3g')
    comps.generate_orb_reps(oml_representation_parameters)
    return comps, quant_vals

for quant_name, charge in zip(quant_names, charges):
    quant=Quantity(quant_name)
    training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, delta_learning_params, oml_representation_parameters, charge)

    optimized_hyperparams=min_sep_IBO_MSE_random_walk(training_comps, training_quants, init_lambda_guess=1e-6, training_set_ratio=0.5, initial_guess_sigma_rescale=1.0, step_magnitude=2.0, num_iters=10, negligible_below_default=-9.21034, negligible_below_lambda=-16.1181, exclusion_permanent=True)

    inv_sq_width_params=optimized_hyperparams["inv_sq_width_params"]
    lambda_val=optimized_hyperparams["lambda_val"]

    K_train=lin_sep_IBO_sym_kernel(training_comps, inv_sq_width_params)
    K_train[np.diag_indices_from(K_train)]+=lambda_val
    alphas=np_cho_solve_wcheck(K_train, training_quants, eigh_rcond=1e-9)
    del(K_train)
    #
    check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, delta_learning_params, oml_representation_parameters, charge)
    K_check=lin_sep_IBO_kernel(check_comps, training_comps, inv_sq_width_params)
    predicted_quants=np.dot(K_check, alphas)
    MAE=np.mean(np.abs(predicted_quants-check_quants))
    print("Quantity: ", quant_name, ", MAE:", MAE)
