# Builds a learning curve for a very small number of datapoints; mainly here to test that learning_curve_building module is working properly.

from learning_curve_building import KR_model, best_model_params, make_learning_curves_with_stdev, print_means_errs_to_log_file,\
                    Delta_learning_parameters, geom_progression, linear_interpolation_points, create_shuffled_xyz_list, dirs_xyz_list,\
                    SLATM_representation, Laplacian_kernel_function
from qm9_format_specs import Quantity
import random
from os.path import expanduser

training_sizes=[20, 40]
model_scanning_size=20
check_size=10

QM9_dir=expanduser("~")+"/QM9_formatted"
num_iters=2

seed=0
lambda_val=1e-6

seed_OML_hyperparams=1
hyperparam_opt_number=40

use_delta_learning=True

xyz_hyperparam_opt=random.Random(seed_OML_hyperparams).sample(dirs_xyz_list(QM9_dir), hyperparam_opt_number)
scanned_models=[]

for sigma in geom_progression(1.0, 2.0, 10):
    cur_model=KR_model()
    cur_model.representation=SLATM_representation()
    cur_model.kernel_function=Laplacian_kernel_function(sigma, lambda_val=lambda_val)
    scanned_models.append(cur_model)

dl_params=Delta_learning_parameters(use_delta_learning=use_delta_learning)

quantity=Quantity('HOMO eigenvalue')
print("Searching best model parameters.")
cur_model=best_model_params(quantity, model_scanning_size, check_size, scanned_models, QM9_dir, num_iters=num_iters, seed=seed, delta_learning_params=dl_params, calc_logs="model_scan_"+quantity.name, quant_logs="temp_vals_model_scan"+quantity.name, output_file="fin_data_model_scan")
print("Making learning curves.")
lc_data=make_learning_curves_with_stdev(quantity, training_sizes, check_size, cur_model, QM9_dir, num_iters=num_iters, seed=seed, delta_learning_params=dl_params, calc_logs="learning_curve_"+quantity.name, quant_logs="temp_vals_learning_curve"+quantity.name, output_file="fin_data_learning_curve")
#job_name="learning curve "+quant_names[quant]
print_means_errs_to_log_file(training_sizes, lc_data, 'final_learning_curve'+quantity.name)
print(cur_model)
#plot_learning_curve(training_sizes, lc_data, job_name, 'training size', 'MAE ('+quant_names[quant]+')')
