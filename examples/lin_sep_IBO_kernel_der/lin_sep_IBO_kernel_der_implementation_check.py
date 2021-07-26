# A script that compares analytical derivative values with the ones obtained via finite difference.

import glob, random, copy
from qml.oml_kernels import lin_sep_IBO_sym_kernel, lin_sep_IBO_kernel
from qml.oml_compound_list import OML_compound_list_from_xyzs
from qml.oml_representations import OML_rep_params
import numpy as np
from qml.hyperparameter_optimization import Gradient_optimization_obj
from learning_curve_building import np_cho_solve

def model_MSE(K_train, K_check, train_quant, check_quant, lambda_val):
    K_train_mod=copy.deepcopy(K_train)
    K_train_mod[np.diag_indices_from(K_train_mod)]+=lambda_val
    alphas=np_cho_solve(K_train_mod, train_quant)
    predictions=np.matmul(K_check, alphas)
    return np.mean((check_quant-predictions)**2)

xyz_list=glob.glob("../../tests/qm7/*.xyz")

inv_sq_width_params=np.repeat(1.0, 22)

lambda_val=0.5

random.seed(1)

xyz_list.sort()
random.shuffle(xyz_list)

num_A_mols=3

num_B_mols=2

xyz_list_A=xyz_list[:num_A_mols]
xyz_list_B=xyz_list[num_A_mols:num_B_mols+num_A_mols]

train_quant=np.array(list(range(num_A_mols)))
check_quant=np.array(list(range(num_A_mols, num_B_mols+num_A_mols)))

rep_params=OML_rep_params(max_angular_momentum=1, ibo_atom_rho_comp=0.95)

A=OML_compound_list_from_xyzs(xyz_list_A)
B=OML_compound_list_from_xyzs(xyz_list_B)

A.generate_orb_reps(rep_params)
B.generate_orb_reps(rep_params)

K_train=lin_sep_IBO_sym_kernel(A, inv_sq_width_params)
K_check=lin_sep_IBO_kernel(B, A, inv_sq_width_params)

K_train_fd_der=np.zeros((num_A_mols, num_A_mols))
K_check_fd_der=np.zeros((num_B_mols, num_A_mols))
MSE_der=0.0
MSE_der_lambda=0.0

der_id=2

fd_step=0.001
for fd_grid_step in [-1, 1]:
    cur_inv_sq_width_params=copy.deepcopy(inv_sq_width_params)
    cur_inv_sq_width_params[der_id]+=fd_step*fd_grid_step
    cur_K_train=lin_sep_IBO_sym_kernel(A, cur_inv_sq_width_params)
    cur_K_check=lin_sep_IBO_kernel(B, A, cur_inv_sq_width_params)
    MSE_der+=model_MSE(cur_K_train, cur_K_check, train_quant, check_quant, lambda_val)*fd_grid_step
    MSE_der_lambda+=model_MSE(K_train, K_check, train_quant, check_quant, lambda_val+fd_step*fd_grid_step)*fd_grid_step

    K_train_fd_der+=fd_grid_step*cur_K_train
    K_check_fd_der+=fd_grid_step*cur_K_check
K_train_fd_der/=2*fd_step
MSE_der/=2*fd_step
K_check_fd_der/=2*fd_step
MSE_der_lambda/=2*fd_step

K_train_wders=lin_sep_IBO_sym_kernel(A, inv_sq_width_params, with_ders=True)
K_check_wders=lin_sep_IBO_kernel(B, A, inv_sq_width_params, with_ders=True)

print("K_train")
print(K_train)
print(K_train_wders[:, :, 0])
print(K_train-K_train_wders[:, :, 0])

print("K_train_ders")
print(K_train_fd_der)
print(K_train_wders[:,:,der_id+1])
print(K_train_fd_der-K_train_wders[:,:,der_id+1])

print("K_check")
print(K_check)
print(K_check_wders[:,:,0])
print(K_check-K_check_wders[:,:,0])

print("K_check_ders")
print(K_check_fd_der)
print(K_check_wders[:,:,der_id+1])
print(K_check_fd_der-K_check_wders[:,:,der_id+1])

GOO=Gradient_optimization_obj(A, train_quant, B, check_quant)
params0=np.array([lambda_val, *inv_sq_width_params])
print("MSE")
MSE_ref=model_MSE(K_train, K_check, train_quant, check_quant, lambda_val)
MSE_GOO=GOO.MSE(params0)
print(MSE_ref)
print(GOO.MSE(params0))
print(MSE_ref-MSE_GOO)

MSE_der_GOO_all=GOO.MSE_der(params0)

print("MSE_der")
MSE_der_GOO=MSE_der_GOO_all[der_id+1]
print(MSE_der)
print(MSE_der_GOO)
print(MSE_der_GOO-MSE_der)

MSE_der_lambda_GOO=MSE_der_GOO_all[0]

print("MSE_der_lambda")
print(MSE_der_lambda)
print(MSE_der_lambda_GOO)
print(MSE_der_lambda-MSE_der_lambda_GOO)
