# Builds a learning curve for a very small number of datapoints; mainly here to test that learning_curve_building module is working properly.

# WARNING: The script currently does not work because one of the UHF calculations does not converge; need to recheck the reasons why.

from learning_curve_building import KR_model, best_model_params, make_learning_curves_with_stdev, print_means_errs_to_log_file,\
                    Delta_learning_parameters, geom_progression, linear_interpolation_points, create_shuffled_xyz_list, dirs_xyz_list,\
                    OML_Slater_pair_rep, OML_GMO_kernel_function
from qm9_format_specs import Quantity
import random, os

training_sizes=[20, 40]
model_scanning_size=20
check_size=10

QM9_dir="/data/"+os.environ["USER"]+"/QM9_formatted"
num_iters=2

seed=0
lambda_val=1e-6

seed_OML_hyperparams=1
hyperparam_opt_number=40

use_delta_learning=True

current_representation=OML_Slater_pair_rep(max_angular_momentum=1, use_Fortran=True, ibo_atom_rho_comp=0.95, calc_type="HF", second_calc_type="UHF",
                                fock_based_coup_mat=True, second_orb_type="IBO_HOMO_removed", num_prop_times=2, prop_delta_t=1.0)

xyz_hyperparam_opt=random.Random(seed_OML_hyperparams).sample(dirs_xyz_list(QM9_dir), hyperparam_opt_number)

hyperparam_opt_comp=current_representation.init_compound_list(xyz_list=xyz_hyperparam_opt)

scanned_models=[]
for sigma_rescale in geom_progression(0.25, 2.0, 2):
    cur_model=KR_model()
    cur_model.representation=current_representation
    cur_model.kernel_function=OML_GMO_kernel_function(sigma_rescale=sigma_rescale, lambda_val=lambda_val)
    print("Generating hyperparameters: ", cur_model)
    cur_model.adjust_hyperparameters(init_compound_list=hyperparam_opt_comp)
    scanned_models.append(cur_model)

dl_params=Delta_learning_parameters(use_delta_learning=use_delta_learning)

quantity=Quantity('HOMO eigenvalue')
print("Searching best model parameters.")
cur_model=best_model_params(quantity, model_scanning_size, check_size, scanned_models, QM9_dir, num_iters=num_iters,
            seed=seed, delta_learning_params=dl_params, calc_logs="model_scan_"+quantity.name, quant_logs="temp_vals_model_scan"+quantity.name, output_file="fin_data_model_scan")
print("Making learning curves.")
lc_data=make_learning_curves_with_stdev(quantity, training_sizes, check_size, cur_model, QM9_dir, num_iters=num_iters,
            seed=seed, delta_learning_params=dl_params, calc_logs="learning_curve_"+quantity.name, quant_logs="temp_vals_learning_curve"+quantity.name, output_file="fin_data_learning_curve")
print_means_errs_to_log_file(training_sizes, lc_data, 'final_learning_curve'+quantity.name)
print(cur_model)

