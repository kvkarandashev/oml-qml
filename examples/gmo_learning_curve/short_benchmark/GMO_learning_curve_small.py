# Builds a learning curve for a very small number of datapoints; mainly here to test that learning_curve_building module is working properly.

from learning_curve_building import KR_model, best_model_params, make_learning_curves_with_stdev, print_means_errs_to_log_file,\
                    Delta_learning_parameters, geom_progression, linear_interpolation_points, create_shuffled_xyz_list, dirs_xyz_list,\
                    OML_representation, OML_GMO_kernel_function
from qm9_format_specs import Quantity
import random
from os.path import expanduser

#training_sizes=[ 1 ]
#model_scanning_size=1
#check_size=1

#QM9_dir="./test_mols"
#num_iters=1

training_sizes=[20, 40] #[ 7368, 14736, 29472, 58944, 117888 ]
model_scanning_size=20 #100
check_size=10

#training_sizes=[10, 20, 40]
#model_scanning_size=5
#check_size=10


QM9_dir=expanduser("~")+"/QM9_formatted"
num_iters=2


seed=0
lambda_val=1e-6

seed_OML_hyperparams=1
hyperparam_opt_number=40
#hyperparam_opt_number=10

rep_type="GMO_HOMO_opt"

use_delta_learning=True

#hyperparam_grid_separation=6 #6

# TO-DO ensure that different seeds are used for learning curves, hyperparameter optimization, and model parameter scanning.

if rep_type=="CM":
    scanned_models=[KR_model(kernel_type="Gaussian", rep_type="CM", Gaussian_sigma=sigma, lambda_val=lambda_val) for sigma in geom_progression(0.125, 2.0, 16)]
else:
    xyz_hyperparam_opt=random.Random(seed_OML_hyperparams).sample(dirs_xyz_list(QM9_dir), hyperparam_opt_number)
    scanned_models=[]
    for sigma_rescale in geom_progression(0.25, 2.0, 2):
        for final_sigma in geom_progression(1.0, 2.0, 2):
            cur_model=KR_model()
            cur_model.representation=OML_representation(ibo_atom_rho_comp=0.9, use_Fortran=True, max_angular_momentum=1)
            cur_model.kernel_function=OML_GMO_kernel_function(sigma_rescale=sigma_rescale, lambda_val=lambda_val, final_sigma=final_sigma, normalize_lb_kernel=True)
            print("Generating hyperparameters: ", cur_model)
            cur_model.adjust_hyperparameters(xyz_hyperparam_opt)
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
