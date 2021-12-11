# Minimal example that 

# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve_wcheck, import_quantity_array
from qm9_format_specs import Quantity
import os, random, qml
import numpy as np
from qml import Compound
from qml.kernels_wders import laplacian_sym_kernel_conv_wders, SLATM_kernel, SLATM_kernel_input
from qml.hyperparameter_optimization import min_sep_IBO_random_walk_optimization
from qml.utils import dump2pkl

def_float_format='{:.8E}'

quant_names=['HOMO eigenvalue', 'LUMO eigenvalue']
seed=1

use_Gauss=False

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

unavailable_ratio=0.5 #-0.1 #0.25

none_ignored=False #True

def ignore_array(real_arr):
    if none_ignored:
        return None
    else:
        dim1=real_arr.shape[0]
        dim2=real_arr.shape[1]
        output=np.zeros((dim1, dim2), dtype=bool)
        if unavailable_ratio>0.0:
            for i in range(dim1):
                for j in range(1, dim2):
                    if unavailable_ratio>0.0:
                        if random.random()<unavailable_ratio:
                            output[i, j]=True
        return output

def get_quants_comps(xyz_list, quantities, dl_params):
    quant_vals=[]
    for quantity in quantities:
        quant_vals.append(import_quantity_array(xyz_list, quantity, dl_params))
    comps=[Compound(xyz_file) for xyz_file in xyz_list]
    quant_vals=np.array(quant_vals)
    return comps, quant_vals.T

quantities=[Quantity(quant_name) for quant_name in quant_names]
training_comps, training_quants=get_quants_comps(xyz_list[:train_num], quantities, delta_learning_params)
training_ignore_array=ignore_array(training_quants)


optimized_hyperparams=min_sep_IBO_random_walk_optimization(training_comps, training_quants,
            quant_ignore_list=training_ignore_array, init_lambda=1e-6, init_param_guess=np.array([1.0, 1000.0]),
            max_stagnating_iterations=8, hyperparam_red_type="default", randomized_iterator_kwargs={"default_step_magnitude" : 0.25},
            iter_dump_name_add="test_min_SLATM", additional_BFGS_iters=8, iter_dump_name_add_BFGS="test_min_BFGS",
            sym_kernel_func=laplacian_sym_kernel_conv_wders, kernel_input_converter=SLATM_kernel_input)



sigmas=optimized_hyperparams["sigmas"]
lambda_val=optimized_hyperparams["lambda_val"]

print("Finalized parameters:", sigmas)
print("Finalized lambda:", lambda_val)

sigma=sigmas[0]

K_train=SLATM_kernel(training_comps, training_comps, sigma, use_Gauss=use_Gauss)
K_train[np.diag_indices_from(K_train)]+=lambda_val



for quant_id in range(len(quant_names)):
    alphas=np_cho_solve_wcheck(K_train, training_quants[:, quant_id], eigh_rcond=1e-9)

    check_comps, check_quants=get_quants_comps(xyz_list[-check_num:], [quantities[quant_id]], delta_learning_params)
    K_check=SLATM_kernel(check_comps, training_comps, sigma, use_Gauss=use_Gauss)
    predicted_quants=np.dot(K_check, alphas)
    MAE=np.mean(np.abs(predicted_quants-check_quants[:, 0]))
    print("Quantity: ", quant_names[quant_id], ", MAE:", MAE)
