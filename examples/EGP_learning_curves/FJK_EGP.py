# For brevity import some functions from python_script_modules/learning_curve_building
from learning_curve_building import Delta_learning_parameters, dirs_xyz_list, np_cho_solve, import_quantity_array
from electrolyte_genome_format_specs import Quantity
import os, random, qml, sys
import numpy as np
from qml import OML_Slater_pair_list_from_xyzs, OML_compound_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import gauss_sep_IBO_sym_kernel, gauss_sep_IBO_kernel
from qml.hyperparameter_optimization import min_sep_IBO_random_walk_optimization
from qml.utils import dump2pkl
from qml.oml_compound import OML_pyscf_calc_params

used_second_orb_type={"reduction_lithium" : "IBO_LUMO_added", "oxidation_lithium" : "IBO_HOMO_removed"}

used_ofd_coup_mats={"FJK" : False, "OFD" : True}

global_Gauss=True # True for old kernel outlined in the manuscript, False for the new one

quant_name=sys.argv[1] #'reduction_lithium'

method_name=sys.argv[2]

localization_procedure="Boys"

second_orb_type=used_second_orb_type[quant_name]

ofd_coup_mats=used_ofd_coup_mats[method_name]

pyscf_calc_params=OML_pyscf_calc_params(scf_max_cycle=200)

def_float_format='{:.8E}'

max_lambda_diag_el_ratio=10.0 # 1e-3

basis='sto-3g'

seed=1

# Replace with path to QM7b directory
EGP_dir=os.environ["DATA"]+"/EGP_xyzs_50"

max_train_num=8000 # 1000
#max_train_num=500
check_num=4000

#train_nums=[100, 250, 500]
train_nums=[500, 1000, 2000, 4000, 8000]

#scan_num=100
scan_num=2000

delta_learning_params=Delta_learning_parameters(use_delta_learning=False, pyscf_calc_params=pyscf_calc_params)

xyz_list=dirs_xyz_list(EGP_dir)
random.seed(seed)
random.shuffle(xyz_list)

train_xyzs=xyz_list[:max_train_num]
scan_xyzs=xyz_list[:scan_num]
check_xyzs=xyz_list[-check_num:]

dump2pkl(xyz_list, "shuffled_list.pkl")

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=2,
                                                use_Fortran=True, ofd_coup_mats=ofd_coup_mats)


def get_quants_comps(xyz_list, quantity, dl_params, baseline_val=None):
    quant_vals=import_quantity_array(xyz_list, quantity, dl_params)
    comps=OML_Slater_pair_list_from_xyzs(xyz_list, calc_type="HF", second_calc_type="UHF", pyscf_calc_params=pyscf_calc_params,
                                second_orb_type=second_orb_type, basis=basis, localization_procedure=localization_procedure)
    comps.generate_orb_reps(oml_representation_parameters)
    np_quant_vals=np.array(quant_vals)
    if baseline_val is None:
        new_baseline_val=np.mean(np_quant_vals)
    else:
        new_baseline_val=baseline_val
    np_quant_vals-=new_baseline_val
    if baseline_val is None:
        return comps, np_quant_vals, new_baseline_val
    else:
        return comps, np_quant_vals

def MAE_from_kernels(train_kernel, check_kernel, train_quants, check_quants):
    alphas=np_cho_solve(train_kernel, train_quants)
    predicted_quants=np.dot(check_kernel, alphas)
    return np.mean(np.abs(predicted_quants-check_quants))

quant=Quantity(quant_name)

train_comps, train_quants, baseline_val=get_quants_comps(train_xyzs, quant, delta_learning_params)

scan_comps, scan_quants=get_quants_comps(scan_xyzs, quant, delta_learning_params, baseline_val=baseline_val)

optimized_hyperparams=min_sep_IBO_random_walk_optimization(scan_comps, scan_quants, init_lambda=1e-6,
            use_Gauss=True, max_stagnating_iterations=8, num_kfolds=128, hyperparam_red_type="ang_mom_classified",
            randomized_iterator_kwargs={"default_step_magnitude" : 0.1, "max_lambda_diag_el_ratio" : max_lambda_diag_el_ratio},
            iter_dump_name_add="test_min_EGP_"+quant_name, rep_params=oml_representation_parameters,
            other_opt_goo_ensemble_kwargs={"global_Gauss" : global_Gauss})

sigmas=optimized_hyperparams["sigmas"]
lambda_val=optimized_hyperparams["lambda_val"]

K_train=gauss_sep_IBO_sym_kernel(train_comps, sigmas, global_Gauss=global_Gauss)
K_train[np.diag_indices_from(K_train)]+=lambda_val

check_comps, check_quants=get_quants_comps(check_xyzs, quant, delta_learning_params, baseline_val=baseline_val)
K_check=gauss_sep_IBO_kernel(check_comps, train_comps, sigmas, global_Gauss=global_Gauss)

MAE=MAE_from_kernels(K_train, K_check, train_quants, check_quants)
MAE_red_str="other training size MAEs:"

for train_num in train_nums:
    MAE_red=0.0
    MAE_red_sq=0.0
    num_red=0
    lower_bound=0
    upper_bound=train_num
    while upper_bound <= max_train_num:
        red_slices=slice(lower_bound, upper_bound)
        red_K_train=K_train[red_slices, :][:, red_slices]
        red_K_check=K_check[:, red_slices]
        red_train_quants=train_quants[red_slices]
        num_red+=1
        cur_MAE=MAE_from_kernels(red_K_train, red_K_check, red_train_quants, check_quants)
        MAE_red+=cur_MAE
        MAE_red_sq+=cur_MAE**2
        lower_bound=upper_bound
        upper_bound+=train_num
    MAE_red/=num_red
    MAE_red_sq/=num_red
    MSE=MAE_red_sq-MAE_red**2
    if MSE>0.0:
        RMSE=np.sqrt(MSE)
    else:
        RMSE=MSE
    MAE_red_str+=str(train_num)+" "+str(MAE_red)+" "+str(RMSE)+";"

output_str="Quantity: "+quant_name+", MAE:"+str(MAE)+";"+MAE_red_str

print(output_str)

lc_file=open("lc_"+quant_name+".dat", 'w')

print(output_str, file=lc_file)
lc_file.close()

