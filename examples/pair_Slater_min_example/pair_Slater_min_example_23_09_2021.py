# Minimal example that calculates model MAEs for LUMO and HOMO energies, and the gap.
# Note that it requires having a directory with QM9 xyz files.

# 23/09/2021 - this version uses stochastic gradient descent to optimize hyperparameters before training the model.
# Next version will hopefully use a representation based on just the Fock matrix which is currently in development.

# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve_wcheck, import_quantity_array
from qm9_format_specs import Quantity
import os, random
import numpy as np
from qml import OML_Slater_pair_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import gauss_sep_IBO_sym_kernel, gauss_sep_IBO_kernel
from qml.hyperparameter_optimization import min_sep_IBO_random_walk_optimization

def_float_format='{:.8E}'

quant_names=['HOMO eigenvalue', 'LUMO eigenvalue', 'HOMO-LUMO gap']
ibo_types=["IBO_HOMO_removed", "IBO_LUMO_added", "IBO_first_excitation"]

seed=1

# Replace with path to QM9 directory
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

train_num=500
check_num=500

delta_learning_params=Delta_learning_parameters(use_delta_learning=True)

xyz_list=dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=1, use_Fortran=True)

def get_quants_comps(xyz_list, quantity, dl_params, oml_representation_parameters, ibo_type):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=OML_Slater_pair_list_from_xyzs(xyz_list, calc_type="HF", second_calc_type="UHF",
                                basis='sto-3g', second_orb_type=ibo_type)
    comps.generate_orb_reps(oml_representation_parameters)
    return comps, np.array(quant_vals)

for quant_name, ibo_type in zip(quant_names, ibo_types):
    quant=Quantity(quant_name)
    training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, delta_learning_params, oml_representation_parameters, ibo_type)

    optimized_hyperparams=min_sep_IBO_random_walk_optimization(training_comps, training_quants, init_lambda=1e-6, use_Gauss=True, max_stagnating_iterations=8, num_kfolds=128,
                                    hyperparam_red_type="ang_mom_classified", randomized_iterator_kwargs={"default_step_magnitude" : 0.25}, iter_dump_name_add="test_min",
                                    rep_params=oml_representation_parameters)

    inv_sq_width_params=optimized_hyperparams["inv_sq_width_params"]
    lambda_val=optimized_hyperparams["lambda_val"]

    K_train=gauss_sep_IBO_sym_kernel(training_comps, inv_sq_width_params)
    K_train[np.diag_indices_from(K_train)]+=lambda_val
    alphas=np_cho_solve_wcheck(K_train, training_quants, eigh_rcond=1e-9)
    del(K_train)
    #
    check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, delta_learning_params, oml_representation_parameters, ibo_type)
    K_check=gauss_sep_IBO_kernel(check_comps, training_comps, inv_sq_width_params)
    predicted_quants=np.dot(K_check, alphas)
    MAE=np.mean(np.abs(predicted_quants-check_quants))
    print("Quantity: ", quant_name, ", MAE:", MAE)
