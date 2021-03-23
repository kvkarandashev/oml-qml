# Builds a learning curve for a very small number of datapoints; mainly here to test that learning_curve_building module is working properly.

# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve, import_quantity_array
from qm9_format_specs import Quantity
import os, random, qml
import numpy as np
from qml import OML_Slater_pair_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import GMO_kernel_params, random_ibo_sample, oml_ensemble_widths_estimate, generate_GMO_kernel

def_float_format='{:.8E}'

quant_names=['HOMO eigenvalue', 'LUMO eigenvalue', 'HOMO-LUMO gap']
ibo_types=["IBO_HOMO_removed", "IBO_LUMO_added", "IBO_first_excitation"]
# Those are parameters I found through scanning at 8000 training points.
opt_sigma_rescalings=[0.5, 0.25, 1.0]
opt_final_sigmas=[8.0, 2.0, 1.0]

seed=1

lambda_val=1e-7

# Replace with path to QM9 directory
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

train_num=500
check_num=100

delta_learning_params=Delta_learning_parameters(use_delta_learning=True)

xyz_list=dirs_xyz_list(QM9_dir)
random.seed(seed)
random.shuffle(xyz_list)

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=1, use_Fortran=True)


def get_quants_comps(xyz_list, quantity, dl_params, oml_representation_parameters, ibo_type):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=OML_Slater_pair_list_from_xyzs(xyz_list, first_calc_type="HF", second_calc_type="UHF",
                                second_orb_type=ibo_type, basis='sto-3g')
    comps.generate_orb_reps(oml_representation_parameters)
    return comps, quant_vals

for quant_name, ibo_type, opt_sigma_rescaling, opt_final_sigma in zip(quant_names, ibo_types, opt_sigma_rescalings, opt_final_sigmas):
    quant=Quantity(quant_name)
    training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quant, delta_learning_params, oml_representation_parameters, ibo_type)
    # Find standard deviations of representation vectors.
    orb_sample=random_ibo_sample(training_comps, pair_reps=True)
    stddevs=oml_ensemble_widths_estimate(orb_sample)
    # (Rescale those deviations to get estimates of optimal hyperparameters.)
    width_params=stddevs/opt_sigma_rescaling
    gmo_kernel_parameters=GMO_kernel_params(final_sigma=opt_final_sigma, use_Fortran=True,
                        normalize_lb_kernel=True, use_Gaussian_kernel=True, pair_reps=True, width_params=width_params)
    # Train the model.
    K_train=generate_GMO_kernel(training_comps, training_comps, gmo_kernel_parameters, sym_kernel_mat=True)
    K_train[np.diag_indices_from(K_train)]+=lambda_val
    alphas=np_cho_solve(K_train, training_quants, eigh_rcond=1e-9)
    del(K_train)
    #
    check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], quant, delta_learning_params, oml_representation_parameters, ibo_type)
    K_check=generate_GMO_kernel(check_comps, training_comps, gmo_kernel_parameters)
    predicted_quants=np.dot(K_check, alphas)
    MAE=np.mean(np.abs(predicted_quants-check_quants))
    print("Quantity: ", quant_name, ", MAE:", MAE)
