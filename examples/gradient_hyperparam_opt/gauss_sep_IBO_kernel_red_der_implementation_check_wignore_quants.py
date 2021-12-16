# A script that compares analytical derivative values with the ones obtained via finite difference
# for all reduced hyperparameter functions currently implemented in the code.

import glob, random, copy
from qml.oml_kernels import gauss_sep_IBO_sym_kernel, gauss_sep_IBO_kernel, oml_ensemble_avs_stddevs
from qml.oml_compound_list import OML_compound_list_from_xyzs
from qml.oml_representations import OML_rep_params
import numpy as np
from qml.hyperparameter_optimization import Gradient_optimization_obj, Ang_mom_classified_rhf, Single_rescaling_rhf, Cho_multi_factors
from learning_curve_building import np_cho_solve

def model_MSE_MAE(K_train, K_check, train_quant, check_quant, train_quant_ignore, check_quant_ignore, lambda_val):
    K_train_mod=copy.deepcopy(K_train)
    K_train_mod[np.diag_indices_from(K_train_mod)]+=lambda_val
    cho_factors=Cho_multi_factors(K_train_mod, indices_to_ignore=train_quant_ignore)
    alphas=cho_factors.solve_with(train_quant)
    predictions=np.matmul(K_check, alphas.T)
    error_vals=check_quant-predictions
    dim1=error_vals.shape[0]
    dim2=error_vals.shape[1]
    for i1 in range(dim1):
        for i2 in range(dim2):
            if check_quant_ignore[i1, i2]:
                error_vals[i1, i2]=0.0
    return np.mean(error_vals**2), np.mean(np.abs(error_vals))

def print_for_red_param_rep_der_id(param_func, A, B, train_quant, check_quant, train_quant_ignore,
                check_quant_ignore, global_Gauss, der_id, init_red_param_guess):
    parameters=red_hyp_func.reduced_params_to_full(init_red_param_guess)

    sigmas=parameters[1:]
    lambda_val=parameters[0]

    model_quant_args=(train_quant, check_quant, train_quant_ignore, check_quant_ignore)

    K_train=gauss_sep_IBO_sym_kernel(A, sigmas, global_Gauss=global_Gauss)

    K_check=gauss_sep_IBO_kernel(B, A, sigmas, global_Gauss=global_Gauss)

    MSE_der=0.0
    MSE_der_lambda=0.0

    MAE_der=0.0
    MAE_der_lambda=0.0

    fd_step=0.001
    for fd_grid_step in [-1, 1]:
        cur_red_params=copy.deepcopy(init_red_param_guess)
        cur_red_params[der_id]+=fd_step*fd_grid_step

        cur_parameters=red_hyp_func.reduced_params_to_full(cur_red_params)

        cur_inv_sq_width_params=cur_parameters[1:]
        cur_lambda_val=cur_parameters[0]

        cur_K_train=gauss_sep_IBO_sym_kernel(A, cur_inv_sq_width_params, global_Gauss=global_Gauss)
        cur_K_check=gauss_sep_IBO_kernel(B, A, cur_inv_sq_width_params, global_Gauss=global_Gauss)

        MSE_fd, MAE_fd=model_MSE_MAE(cur_K_train, cur_K_check, *model_quant_args, cur_lambda_val)

        MSE_der+=MSE_fd*fd_grid_step

        MAE_der+=MAE_fd*fd_grid_step

    MSE_der/=2*fd_step
    MAE_der/=2*fd_step

    MSE_ref, MAE_ref=model_MSE_MAE(K_train, K_check, *model_quant_args, lambda_val)

    # For MSE.
    goo_args=[A, train_quant, B, check_quant]
    goo_kwargs={"use_Gauss" : True, "reduced_hyperparam_func" : red_hyp_func, "global_Gauss" : global_Gauss,
                "training_quants_ignore" : train_quant_ignore, "check_quants_ignore" : check_quant_ignore}

    GOO=Gradient_optimization_obj(*goo_args, use_MAE=False, **goo_kwargs)

    MSE_GOO=GOO.error_measure(parameters)
    print("MSE diffs", MSE_ref, MSE_GOO, MSE_ref-MSE_GOO)

    MSE_der_GOO_all=GOO.error_measure_ders(parameters)

    MSE_der_GOO=MSE_der_GOO_all[der_id]
    print("MSE_der diffs", MSE_der, MSE_der_GOO, MSE_der_GOO-MSE_der)

    # For MAE.
    GOO=Gradient_optimization_obj(*goo_args, **goo_kwargs)

    MAE_GOO=GOO.error_measure(parameters)
    print("MAE diffs", MAE_ref, MAE_GOO, MAE_ref-MAE_GOO)

    MAE_der_GOO_all=GOO.error_measure_ders(parameters)

    MAE_der_GOO=MAE_der_GOO_all[der_id]
    print("MAE_der diffs", MAE_der, MAE_der_GOO, MAE_der_GOO-MAE_der)

def print_for_red_param_representation(red_hyp_func, *other_args):
    num_params=red_hyp_func.num_reduced_params

    init_red_param_guess=[]

    for param_id in range(num_params):
        init_red_param_guess.append(0.25*param_id-0.5)

    init_red_param_guess=np.array(init_red_param_guess)

    parameters=red_hyp_func.reduced_params_to_full(init_red_param_guess)
    # Firstly, check that forward and inverse functions for reduced parameters are actually inverse to each other.
    inv_from_parameters=red_hyp_func.full_params_to_reduced(parameters)
    print("Reduced hyperparameters:", init_red_param_guess, "from inversed function:", inv_from_parameters)
    print("Difference:", init_red_param_guess-inv_from_parameters)
    print("Parameters:", parameters)

    for der_id in range(num_params):
        print("der_id", der_id)
        print_for_red_param_rep_der_id(red_hyp_func, *other_args, der_id, init_red_param_guess)

xyz_list=glob.glob("../../tests/qm7/*.xyz")

random.seed(1)

xyz_list.sort()
random.shuffle(xyz_list)

num_A_mols=40

num_B_mols=20

xyz_list_A=xyz_list[:num_A_mols]
xyz_list_B=xyz_list[num_A_mols:num_B_mols+num_A_mols]

change_irrelevant_train_quant=True #False # to compare results obtained with both values; should be same

num_quants=3

unavailable_ratio=0.25 #-0.1 #0.25
def ignore_array(real_arr):
    dim1=real_arr.shape[0]
    dim2=real_arr.shape[1]
    output=np.zeros((dim1, dim2), dtype=bool)
    if unavailable_ratio>0.0:
        for i in range(dim1):
            for j in range(1, dim2):
                if output[i, j-1]:
                    output[i, j]=True
                else:
                    if random.random()<unavailable_ratio:
                        output[i, j]=True
    return output

train_quant=np.zeros((num_A_mols, num_quants))
check_quant=np.zeros((num_B_mols, num_quants))

train_quant_ignore=ignore_array(train_quant)
check_quant_ignore=ignore_array(check_quant)

cur_val=1.0

for (q, q_ignore, q_change_zero) in [(train_quant, train_quant_ignore, change_irrelevant_train_quant), (check_quant, check_quant_ignore, True)]:
    dim1=q.shape[0]
    dim2=q.shape[1]
    for i1 in range(dim1):
        for i2 in range(dim2):
            q[i1, i2]=cur_val
            if (q_change_zero and q_ignore[i1, i2]):
                q[i1, i2]=0.0
            cur_val+=1.0

rep_params=OML_rep_params(max_angular_momentum=1, ibo_atom_rho_comp=0.95)

A=OML_compound_list_from_xyzs(xyz_list_A)
B=OML_compound_list_from_xyzs(xyz_list_B)

A.generate_orb_reps(rep_params)
B.generate_orb_reps(rep_params)

avs, stddevs=oml_ensemble_avs_stddevs(A)

use_Gauss=True

red_hyp_funcs=[]
red_hyp_funcs.append(Ang_mom_classified_rhf(rep_params=rep_params, stddevs=stddevs, use_Gauss=use_Gauss))
red_hyp_funcs.append(Single_rescaling_rhf(stddevs=stddevs, use_Gauss=use_Gauss,
                                            rep_params=rep_params, use_global_mat_prop_coeffs=False))
red_hyp_funcs.append(Single_rescaling_rhf(stddevs=stddevs, use_Gauss=use_Gauss,
                                            rep_params=rep_params, use_global_mat_prop_coeffs=True))

for red_hyp_func in red_hyp_funcs:
    print("red_hyp_func:", red_hyp_func)
    for global_Gauss in [False, True]:
        print("global_Gauss", global_Gauss)
        print_for_red_param_representation(red_hyp_func, A, B, train_quant, check_quant, train_quant_ignore, check_quant_ignore, global_Gauss)


