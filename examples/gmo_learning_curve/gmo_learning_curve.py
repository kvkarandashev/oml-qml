# This script reproduces the first learning curve I got for HOMO energy values.
# WARNING: You need to have python_script_modules and bash_scripts in PYTHONPATH and PATH for the script to run properly.
# See the test_script.sh in tests/ to see how these variables are added.
# The script will parallelize over OMP_NUM_THREADS processes.

from learning_curve_building import KR_model, best_model_params, make_learning_curves_with_stdev, print_means_errs_to_log_file,\
                    Delta_learning_parameters, geom_progression, linear_interpolation_points, create_shuffled_xyz_list, dirs_xyz_list,\
                    OML_GMO_kernel_function, OML_representation
from qm9_format_specs import Quantity
import random, os

training_sizes=[800, 1600, 3200, 6400, 12800] #[ 7368, 14736, 29472, 58944, 117888 ]
model_scanning_size=3200
check_size=800


QM9_dir=os.path.expanduser("~")+"/QM9_formatted"
num_iters=1


seed=0
lambda_val=1e-9

seed_OML_hyperparams=60
hyperparam_opt_number=1600
#hyperparam_opt_number=10

rep_type="OML_HOMO_opt"

use_delta_learning=True

fock_based_coup_mat=False

true_char_en=0.5

num_fbcm_omegas=4

kernel_type="GMO_HOMO_opt"

# TO-DO ensure that different seeds are used for learning curves, hyperparameter optimization, and model parameter scanning.

if rep_type=="CM":
    scanned_models=[KR_model(kernel_type="Gaussian", rep_type="CM", Gaussian_sigma=sigma, lambda_val=lambda_val) for sigma in geom_progression(0.125, 2.0, 16)]
else:
    xyz_hyperparam_opt=random.Random(seed_OML_hyperparams).sample(dirs_xyz_list(QM9_dir), hyperparam_opt_number)
    scanned_models=[]
    for sigma_rescale in geom_progression(0.125, 2.0, 8):
        for final_sigma in geom_progression(0.125, 2.0, 8):
            cur_model=KR_model()
            cur_model.kernel_function=OML_GMO_kernel_function(use_Fortran=True, final_sigma=final_sigma, sigma_rescale=sigma_rescale, lambda_val=lambda_val)
            cur_model.representation=OML_representation(ibo_atom_rho_comp=0.95, use_Fortran=True, max_angular_momentum=1, 
                fock_based_coup_mat=fock_based_coup_mat) #, characteristic_energy=true_char_en*num_fbcm_omegas, num_fbcm_omegas=num_fbcm_omegas)
            print("Generating hyperparameters: ", cur_model)
            cur_model.adjust_hyperparameters(xyz_hyperparam_opt)
            scanned_models.append(cur_model)

dl_params=Delta_learning_parameters(use_delta_learning=use_delta_learning)

quantity=Quantity('HOMO eigenvalue')
print("Searching best model parameters.")
cur_model=best_model_params(quantity, model_scanning_size, check_size, scanned_models, QM9_dir, num_iters=num_iters, seed=seed, calc_logs="sigma_scan_log"+quantity.name, delta_learning_params=dl_params, quant_logs="temp_vals_model_scan"+quantity.name, output_file="fin_data_model_scan")
print("Making learning curves.")
lc_data=make_learning_curves_with_stdev(quantity, training_sizes, check_size, cur_model, QM9_dir, num_iters=num_iters, seed=seed, output_file="learning_curve_data"+quantity.name, delta_learning_params=dl_params, calc_logs="learning_curve_log"+quantity.name, quant_logs="temp_vals_learning_curve"+quantity.name)
#job_name="learning curve "+quant_names[quant]
print_means_errs_to_log_file(training_sizes, lc_data, 'final_learning_curve'+quantity.name)
print(cur_model)
#plot_learning_curve(training_sizes, lc_data, job_name, 'training size', 'MAE ('+quant_names[quant]+')')
