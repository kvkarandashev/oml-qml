import numpy as np
from scipy.linalg import cho_factor, cho_solve
import math, random, copy
from .oml_kernels import lin_sep_IBO_kernel_conv, lin_sep_IBO_sym_kernel_conv, GMO_sep_IBO_kern_input, oml_ensemble_avs_stddevs, gauss_sep_IBO_kernel_conv, gauss_sep_IBO_sym_kernel_conv
from .oml_representations import component_id_ang_mom_map
from .utils import dump2pkl
from scipy.optimize import minimize
from .python_parallelization import embarassingly_parallel

# Procedures from Lambda_opt_step to MAE bisection_optimization were the old iteration for gradient-based optimization, but only limited to the lambda variable.
class Lambda_opt_step:
    def __init__(self, lambda_val=None):
        self.lambda_val=lambda_val
        self.alphas=None
        self.cho_decomp=None # Cholesky decomposition of (traning kernel + lambda*unity_matrix)
        self.cho_decomp_valid=None # whether Cholesky decomposition of (traning kernel + lambda*unity_matrix) exists.
        self.MAE_val=None
        self.err_der_val=None
        self.predicted_vals=None
    def check_cho_decomp(self, train_kernel):
        if (self.cho_decomp is None) and (self.cho_decomp_valid is not False):
            if self.lambda_val is None:
                mod_train_kernel=train_kernel
            else:
                mod_train_kernel=np.copy(train_kernel)
                mod_train_kernel[np.diag_indices_from(train_kernel)]+=self.lambda_val
            try:
                self.cho_decomp=cho_factor(mod_train_kernel)
                self.cho_decomp_valid=True
            except np.linalg.LinAlgError: # means mod_train_kernel is not invertible
                self.cho_decomp_valid=False
    def check_alphas(self, train_kernel, train_vals):
        if (self.alphas is None) and (self.cho_decomp_valid is not False):
            self.check_cho_decomp(train_kernel)
            if self.cho_decomp_valid:
                self.alphas=cho_solve(self.cho_decomp, train_vals)
    def check_predicted_vals(self, train_kernel, train_vals, check_kernel):
        if (self.predicted_vals is None) and (self.cho_decomp_valid is not False):
            self.check_alphas(train_kernel, train_vals)
            if self.cho_decomp_valid:
                self.predicted_vals=np.dot(check_kernel, self.alphas)
    def MAE(self, train_kernel, train_vals, check_kernel, check_vals):
        if (self.MAE_val is None) and (self.cho_decomp_valid is not False):
            self.check_predicted_vals(train_kernel, train_vals, check_kernel)
            if self.cho_decomp_valid:
                self.MAE_val=np.mean(np.abs(self.predicted_vals-check_vals))
        return self.MAE_val
    def err_der(self, train_kernel, train_vals, check_kernel, check_vals, square_der=False):
        if self.err_der_val is None:
            self.check_predicted_vals(train_kernel, train_vals, check_kernel)
            if self.cho_decomp_valid:
                der_predicted_vals=-np.dot(check_kernel, cho_solve(self.cho_decomp, self.alphas))
                if square_der:
                    self.err_der_val=np.mean(der_predicted_vals*(self.predicted_vals-check_vals))
                else:
                    self.err_der_val=np.mean(der_predicted_vals*np.sign(self.predicted_vals-check_vals))
            else:
                self.err_der_val=-1.0 # need to increase lambda.
        return self.err_der_val
    def __str__(self):
        output="optimization_step;lambda: "+str(self.lambda_val)
        if self.cho_decomp_valid is False:
            output+=" ,Cholesky failed"
        output+=kwdstr("MAE", self.MAE_val)
        output+=kwdstr("err_der", self.err_der_val)
        return output
    def __repr__(self):
        return str(self)

def kwdstr(keyword, value):
    if value is None:
        return ""
    else:
        return " ,"+keyword+": "+str(value)

def optimized_lambda_MAE(train_kernel, train_vals, check_kernel, check_vals, scan_multiplier=10.0,
                    initial_lambda_val=1e-9, initial_scan_num=9, additional_bisection=True, **bisec_kwargs):
    data_args=(train_kernel, train_vals, check_kernel, check_vals)
    cur_lambda=initial_lambda_val
    min_MAE=None
    # Scan for a lambda that minimizes MAE.
    for scan_counter in range(initial_scan_num):
        cur_lambda_opt_step=Lambda_opt_step(cur_lambda)
        cur_MAE=cur_lambda_opt_step.MAE(*data_args)
        print("scanning:", cur_lambda_opt_step)
        if cur_MAE is not None:
            update_min=True
            if min_MAE is None:
                update_min=True
            else:
                update_min=(cur_MAE<min_MAE)
            if update_min:
                min_MAE=cur_MAE
                min_opt_step=cur_lambda_opt_step
                min_lambda=cur_lambda
        cur_lambda*=scan_multiplier
    # Try finding an even smaller MAE with bisection.
    if additional_bisection:
        bisection_min_lambda, bisection_min_MAE=MAE_bisection_optimization(min_opt_step, *data_args, **bisec_kwargs)
        if bisection_min_MAE<min_MAE:
            print("Improvement with bisection")
            return bisection_min_lambda, bisection_min_MAE
    return min_lambda, min_MAE

def MAE_bisection_optimization(initial_lambda_opt_step, train_kernel, train_vals, check_kernel, check_vals, bisec_scan_multiplier=8.0,
                    bisec_log_diff_tol=0.01, bisec_minimal_lambda=1e-11, bisec_maximal_lambda=1.0):
    data_args=(train_kernel, train_vals, check_kernel, check_vals)
    cur_der=initial_lambda_opt_step.err_der(*data_args, square_der=True)
    der_positive=(cur_der>0.0)
    if (der_positive and (bisec_scan_multiplier>1.0)) or ((not der_positive) and (bisec_scan_multiplier<1.0)):
        bisec_scan_multiplier**=-1
    bisection_interval=[]
    cur_lambda_val=initial_lambda_opt_step.lambda_val
    print("bisection_starting_lambda_step:", initial_lambda_opt_step)
    while (len(bisection_interval)==0):
        next_lambda_val=cur_lambda_val*bisec_scan_multiplier
        next_scan_step=Lambda_opt_step(next_lambda_val)
        next_der=next_scan_step.err_der(*data_args, square_der=True)
        next_MAE=next_scan_step.MAE(*data_args)
        print("scanning:next_step:", next_scan_step)
        next_der_positive=(next_der>0)
        if next_der_positive != der_positive:
            bisection_interval=[next_lambda_val, cur_lambda_val]
            bisection_interval.sort()
        else:
            if ((next_lambda_val<bisec_minimal_lambda) or (next_lambda_val>bisec_maximal_lambda)):
                return next_lambda_val, next_MAE
            cur_lambda_val=next_lambda_val
            cur_MAE=next_MAE
    # Do the bisection search.
    log_diff_tol_mult=math.exp(bisec_log_diff_tol)
    print('bisection interval:', bisection_interval)
    while (bisection_interval[1]>bisection_interval[0]*log_diff_tol_mult):
        middle_lambda=math.sqrt(bisection_interval[0]*bisection_interval[1])
        middle_step=Lambda_opt_step(middle_lambda)
        cur_der=middle_step.err_der(*data_args, square_der=True)
        print("bisection:middle_step:", middle_step)
        if (cur_der>0):
            updated_id=1
        else:
            updated_id=0
        bisection_interval[updated_id]=middle_lambda
    final_bisection_step=Lambda_opt_step(middle_lambda)
    final_bisection_step_MAE=final_bisection_step.MAE(*data_args)
    print("final_bisection_step:",final_bisection_step)
    if final_bisection_step_MAE is None:
        upper_interval_bound=bisection_interval[1]
        upper_interval_step=Lambda_opt_step(upper_interval_bound)
        upper_interval_MAE=upper_interval_step.MAE(*data_args)
        print("Final bisection result changed to upper bound:", upper_interval_bound, upper_interval_MAE)
        return upper_interval_bound, upper_interval_MAE
    else:
        return middle_lambda, final_bisection_step_MAE


#########
# Procedures for general gradient-based optimization.
#########
class Gradient_optimization_obj:
    def __init__(self, training_compounds, training_quants, check_compounds, check_quants, use_Gauss=False, use_MAE=True, reduced_hyperparam_func=None, sym_kernel_func=None, kernel_func=None, kernel_input_converter=None):
        self.init_kern_funcs(use_Gauss=use_Gauss, reduced_hyperparam_func=reduced_hyperparam_func, sym_kernel_func=sym_kernel_func,
                            kernel_func=kernel_func, kernel_input_converter=kernel_input_converter)

        self.training_compounds=GMO_sep_IBO_kern_input(training_compounds)
        self.check_compounds=GMO_sep_IBO_kern_input(check_compounds)

        self.training_quants=training_quants
        self.check_quants=check_quants

        self.use_MAE=use_MAE

    def init_kern_funcs(self, use_Gauss=False, reduced_hyperparam_func=None, sym_kernel_func=None, kernel_func=None, kernel_input_converter=None):

        if kernel_input_converter is None:
            self.reduced_hyperparam_func=reduced_hyperparam_func

        if kernel_func is None:
            if use_Gauss:
                self.def_kern_func=gauss_sep_IBO_kernel_conv
            else:
                self.def_kern_func=lin_sep_IBO_kernel_conv
        else:
            self.def_kern_func=kernel_func

        if sym_kernel_func is None:
            if use_Gauss:
                self.sym_kern_func=gauss_sep_IBO_sym_kernel_conv
            else:
                self.sym_kern_func=lin_sep_IBO_sym_kernel_conv
        else:
            self.sym_kern_func=sym_kernel_func

        if kernel_input_converter is None:
            self.kernel_input_converter=GMO_sep_IBO_kern_input
        else:
            self.kernel_input_converter=kernel_input_converter

    def reinitiate_basic_params(self, parameters):
        self.all_parameters=parameters
        self.num_params=len(self.all_parameters)
        self.lambda_val=parameters[0]
        self.inv_sq_width_params=parameters[1:]

    def recalculate_kernel_matrices(self):
        self.train_kernel=self.sym_kern_func(self.training_compounds, self.inv_sq_width_params)
        self.check_kernel=self.def_kern_func(self.check_compounds, self.training_compounds, self.inv_sq_width_params)

    def recalculate_kernel_mats_ders(self):
        train_kernel_wders=self.sym_kern_func(self.training_compounds, self.inv_sq_width_params, with_ders=True)
        check_kernel_wders=self.def_kern_func(self.check_compounds, self.training_compounds, self.inv_sq_width_params, with_ders=True)

        self.train_kernel=train_kernel_wders[:, :, 0]
        self.check_kernel=check_kernel_wders[:, :, 0]

        num_train_compounds=self.training_compounds.num_mols

        self.train_kernel_ders=one_diag_unity_tensor(num_train_compounds, self.num_params)
        self.check_kernel_ders=np.zeros((self.check_compounds.num_mols, num_train_compounds, self.num_params))

        self.train_kernel_ders[:, :, 1:]=train_kernel_wders[:, :, 1:]
        self.check_kernel_ders[:, :, 1:]=check_kernel_wders[:, :, 1:]
        if self.reduced_hyperparam_func is not None:
            self.train_kernel_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.train_kernel_ders, self.all_parameters)
            self.check_kernel_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.check_kernel_ders, self.all_parameters)

    def reinitiate_basic_components(self):
        
        modified_train_kernel=np.copy(self.train_kernel)
        modified_train_kernel[np.diag_indices_from(modified_train_kernel)]+=self.lambda_val
        try:
            self.train_cho_decomp=cho_factor(modified_train_kernel)
        except np.linalg.LinAlgError: # means mod_train_kernel is not invertible
            self.train_cho_decomp=None
        self.train_kern_invertible=(self.train_cho_decomp is not None)
        if self.train_kern_invertible:
            self.alphas=cho_solve(self.train_cho_decomp, self.training_quants)
            self.predictions=np.matmul(self.check_kernel, self.alphas)
            self.prediction_errors=self.predictions-self.check_quants

    def reinitiate_error_measures(self):
        if self.train_kern_invertible:
            self.cur_MAE=np.mean(np.abs(self.prediction_errors))
            self.cur_MSE=np.mean(self.prediction_errors**2)
        else:
            self.cur_MAE=None
            self.cur_MSE=None

    def der_predictions(self, train_der, check_der):
        output=np.matmul(train_der, self.alphas)
        output=cho_solve(self.train_cho_decomp, output)
        output=-np.matmul(self.check_kernel, output)
        output+=np.matmul(check_der, self.alphas)
        return output

    def reinitiate_error_measure_ders(self, lambda_der_only=False):
        if self.train_kern_invertible:
            if lambda_der_only:
                num_ders=1
            else:
                num_ders=self.check_kernel_ders.shape[2]
            self.cur_MSE_der=np.zeros((num_ders,))
            self.cur_MAE_der=np.zeros((num_ders,))
            for der_id in range(num_ders):
                cur_der_predictions=self.der_predictions(self.train_kernel_ders[:, :, der_id], self.check_kernel_ders[:, :, der_id])
                self.cur_MSE_der[der_id]=2*np.mean(self.prediction_errors*cur_der_predictions)   
                self.cur_MAE_der[der_id]=np.mean(cur_der_predictions*np.sign(self.prediction_errors))
        else:
            self.cur_MSE_der=non_invertible_default_log_der()
            self.cur_MAE_der=non_invertible_default_log_der()

    def error_measure(self, parameters):
        self.reinitiate_basic_params(parameters)
        self.recalculate_kernel_matrices()
        self.reinitiate_basic_components()
        self.reinitiate_error_measures()
        print("# Current parameter vales:", parameters)
        print("# Current MAE:", self.cur_MAE, "MSE:", self.cur_MSE)
        if self.use_MAE:
            return self.cur_MAE
        else:
            return self.cur_MSE

    def error_measure_ders(self, parameters, lambda_der_only=False):
        self.reinitiate_basic_params(parameters)
        self.recalculate_kernel_mats_ders()
        self.reinitiate_basic_components()
        self.reinitiate_error_measure_ders(lambda_der_only=lambda_der_only)

        print("# Current parameter values:", parameters)
        print("# Current MSE derivatives:", self.cur_MSE_der)
        print("# Current MAE derivatives:", self.cur_MAE_der)
        if self.use_MAE:
            return self.cur_MAE_der
        else:
            return self.cur_MSE_der

    def error_measure_wders(self, parameters, lambda_der_only=False):
        self.reinitiate_basic_params(parameters)
        self.recalculate_kernel_mats_ders()
        self.reinitiate_basic_components()
        self.reinitiate_error_measure_ders(lambda_der_only=lambda_der_only)
        self.reinitiate_error_measures()
        if self.use_MAE:
            return self.cur_MAE, self.cur_MAE_der
        else:
            return self.cur_MSE, self.cur_MSE_der


def non_invertible_default_log_der():
    return np.array([-1.0]) # we only have derivative for lambda value

class GOO_ensemble_subset(Gradient_optimization_obj):
    def __init__(self, training_indices, check_indices, all_quantities, use_MAE=True):
        self.training_indices=training_indices
        self.check_indices=check_indices

        self.training_quants=all_quantities[self.training_indices]
        self.check_quants=all_quantities[self.check_indices]
        
        self.use_MAE=use_MAE
    def recalculate_kernel_matrices(self, global_matrix=None, global_matrix_ders=None):
        if global_matrix is not None:
            self.train_kernel=global_matrix[self.training_indices, :][:, self.training_indices]
            self.check_kernel=global_matrix[self.check_indices, :][:, self.training_indices]
    def recalculate_kernel_mats_ders(self, global_matrix=None, global_matrix_ders=None):
        self.recalculate_kernel_matrices(global_matrix=global_matrix)
        if global_matrix_ders is not None:
            self.train_kernel_ders=global_matrix_ders[self.training_indices, :][:, self.training_indices][:]
            self.check_kernel_ders=global_matrix_ders[self.check_indices, :][:, self.training_indices][:]

# This class was introduced to enable multiple cross-validation.
class GOO_ensemble(Gradient_optimization_obj):
    def __init__(self, all_compounds, all_quantities, train_id_lists, check_id_lists, use_Gauss=False, use_MAE=True,
                        reduced_hyperparam_func=None, num_procs=None, num_threads=None, kernel_input_converter=None,
                        sym_kernel_func=None, num_kernel_params=None):

        self.init_kern_funcs(use_Gauss=use_Gauss, reduced_hyperparam_func=reduced_hyperparam_func,
                            sym_kernel_func=sym_kernel_func, kernel_func=None, kernel_input_converter=kernel_input_converter)

        self.all_compounds=self.kernel_input_converter(all_compounds)

        self.tot_num_points=len(all_compounds)


        self.goo_ensemble_subsets=[]
        for train_id_list, check_id_list in zip(train_id_lists, check_id_lists):
            self.goo_ensemble_subsets.append(GOO_ensemble_subset(train_id_list, check_id_list,
                                                        all_quantities, use_MAE=use_MAE))
        self.num_subsets=len(self.goo_ensemble_subsets)

        self.presaved_parameters=None

        self.num_procs=num_procs

        self.num_threads=num_threads

    def error_measure_wders(self, parameters, recalc_global_matrices=True, lambda_der_only=False, negligible_red_param_distance=None):
        if recalc_global_matrices:
            need_recalc=True
            if negligible_red_param_distance is not None:
                if self.presaved_parameters is not None:
                    need_recalc=(np.sqrt(np.sum((parameters-self.presaved_parameters)**2))>negligible_red_param_distance)
                if need_recalc:
                    self.presaved_parameters=np.copy(parameters)
            if need_recalc:
                self.recalculate_global_matrices(parameters)

        error_mean=0.0
        error_mean_ders=None

        error_erders_list=self.subset_error_measures_wders(parameters, lambda_der_only=lambda_der_only)

        for cur_error, cur_error_ders in error_erders_list:
            if cur_error is None:
                error_mean=0.0
                error_mean_ders=non_invertible_default_log_der()
                break
            else:
                error_mean+=cur_error
                if error_mean_ders is None:
                    error_mean_ders=np.copy(cur_error_ders)
                else:
                    error_mean_ders+=cur_error_ders
        error_mean/=self.num_subsets
        error_mean_ders/=self.num_subsets
        return error_mean, error_mean_ders

    def subset_error_measures_wders(self, parameters, lambda_der_only=False):
        if self.num_procs is None:
            return [goo_ensemble_subset.error_measure_wders(parameters, lambda_der_only=lambda_der_only) for goo_ensemble_subset in self.goo_ensemble_subsets]
        else:
            return embarassingly_parallel(single_subset_error_measure_wders, self.goo_ensemble_subsets, (parameters, lambda_der_only), num_threads=self.num_threads, num_procs=self.num_procs)

    def recalculate_global_matrices(self, parameters):
        global_kernel_wders=self.sym_kern_func(self.all_compounds, parameters[1:], with_ders=True)
        self.global_matrix=global_kernel_wders[:, :, 0]
        self.global_matrix_ders=one_diag_unity_tensor(self.tot_num_points, len(parameters))
        self.global_matrix_ders[:, :, 1:]=global_kernel_wders[:, :, 1:]

        if self.reduced_hyperparam_func is not None:
            self.global_matrix_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.global_matrix_ders, parameters)

        for subset_id in range(self.num_subsets):
            self.goo_ensemble_subsets[subset_id].recalculate_kernel_mats_ders(global_matrix=self.global_matrix, global_matrix_ders=self.global_matrix_ders)

# Auxiliary function for joblib parallelization.
def single_subset_error_measure_wders(subset, parameters, lambda_der_only):
    return subset.error_measure_wders(parameters, lambda_der_only=lambda_der_only)

def generate_random_GOO_ensemble(all_compounds, all_quantities, num_kfolds=16, training_set_ratio=0.5, use_Gauss=False, use_MAE=True, reduced_hyperparam_func=None, num_procs=None, num_threads=None):
    num_points=len(all_quantities)
    train_point_num=int(num_points*training_set_ratio)

    all_indices=list(range(num_points))

    train_id_lists=[]
    check_id_lists=[]

    for kfold_id in range(num_kfolds):
        train_id_list=random.sample(all_indices, train_point_num)
        train_id_list.sort()
        check_id_list=[]
        for train_interval_id in range(train_point_num+1):
            if train_interval_id==0:
                lower_bound=0
            else:
                lower_bound=train_id_list[train_interval_id-1]+1
            if train_interval_id==train_point_num:
                upper_bound=num_points
            else:
                upper_bound=train_id_list[train_interval_id]
            for index in range(lower_bound, upper_bound):
                check_id_list.append(index)
        train_id_lists.append(train_id_list)
        check_id_lists.append(check_id_list)

    return GOO_ensemble(all_compounds, all_quantities, train_id_lists, check_id_lists, use_Gauss=use_Gauss, use_MAE=use_MAE, reduced_hyperparam_func=reduced_hyperparam_func, num_procs=num_procs, num_threads=num_threads)
    

#   For going between the full hyperparameter set (lambda, global sigma, and other sigmas)
#   and a reduced hyperparameter set. The default class uses no reduction, just rescaling to logarithms.
class Reduced_hyperparam_func:
    def __init__(self, use_Gauss=False):
        self.num_full_params=None
        self.num_reduced_params=None
        self.use_Gauss=use_Gauss
    def initiate_param_nums(self, param_dim_arr):
        if ((self.num_full_params is None) and (self.num_reduced_params is None)):
            if isinstance(param_dim_arr, int):
                self.num_full_params=param_dim_arr
            else:
                self.num_full_params=len(param_dim_arr)
            self.num_reduced_params=self.num_full_params
    def reduced_params_to_full(self, reduced_parameters):
        return np.exp(reduced_parameters)
    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        return full_derivatives*full_parameters
    def full_params_to_reduced(self, full_parameters):
        return np.log(full_parameters)
    def initial_reduced_parameter_guess(self, init_lambda, base_inv_sqwidth_params):
        num_full_params=len(base_inv_sqwidth_params)+1
        self.initiate_param_nums(len(base_inv_sqwidth_params)+1)
        output=np.zeros((self.num_full_params,))
        output[0]=np.log(init_lambda)
        output[-len(base_inv_sqwidth_params):]=np.log(base_inv_sqwidth_params)
        return output
    def jacobian(self, parameters):
        self.initiate_param_nums(parameters)
        output=np.zeros((self.num_full_params, self.num_reduced_params))
        for param_id in range(self.num_full_params):
            cur_unity_vector=np.zeros((self.num_full_params,))
            cur_unity_vector[param_id]=1.0
            output[param_id, :]=self.full_derivatives_to_reduced(cur_unity_vector, parameters)[:]
        return output
    def transform_der_array_to_reduced(self, input_array, parameters):
        jacobian=self.jacobian(parameters)
        return np.matmul(input_array, jacobian)

# For using a simple rescaling coefficient for sigmas.
class Single_rescaling_rhf(Reduced_hyperparam_func):
    def __init__(self, base_inv_sqwidth_params, use_Gauss=False):
        self.base_inv_sqwidth_params=base_inv_sqwidth_params
        self.num_full_params=len(base_inv_sqwidth_params)+1
        self.num_reduced_params=2
        self.use_Gauss=use_Gauss
        self.sigmas_start_id=1
        if self.use_Gauss:
            self.num_full_params+=1
            self.num_reduced_params+=1
            self.sigmas_start_id+=1
    def reduced_params_to_full(self, reduced_parameters):
        output=np.zeros((self.num_full_params,))
        output[:self.sigmas_start_id]=np.exp(reduced_parameters[:self.sigmas_start_id])
        output[self.sigmas_start_id:]=self.base_inv_sqwidth_params*np.exp(reduced_parameters[-1])
        return output
    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.sigmas_start_id]=full_derivatives[:self.sigmas_start_id]*full_parameters[:self.sigmas_start_id]
        for sigma_id in range(self.sigmas_start_id, self.num_full_params):
            output[-1]+=full_derivatives[sigma_id]*full_parameters[sigma_id]
        return output
    def full_params_to_reduced(self, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.sigmas_start_id]=np.log(full_parameters[:self.sigmas_start_id])
        est_counter=0
        for sigma_id in range(self.sigmas_start_id, self.num_full_params):
            output[-1]+=np.log(full_parameters[sigma_id]/self.base_inv_sqwidth_params[sigma_id])
            est_counter+=1
        output[-1]/=est_counter
        return
    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        if self.use_Gauss:
            return np.array([np.log(init_lambda), 1.0, 1.0])
        else:
            return np.array([np.log(init_lambda), 1.0])

class Ang_mom_classified_rhf(Reduced_hyperparam_func):
    def __init__(self, rep_params, stddevs, use_Gauss=False):
        ang_mom_map=component_id_ang_mom_map(rep_params)
        self.num_simple_log_params=1
        if use_Gauss:
            self.num_simple_log_params+=1
        self.num_full_params=self.num_simple_log_params+len(ang_mom_map)

        prop_coeff_id_dict={}
        red_param_id_dict={}

        last_prop_coeff=0
        last_red_param=self.num_simple_log_params-1

        self.reduced_param_id_lists=[]
        self.sym_multipliers=[]
        self.prop_coeff_id=[]

        for simple_log_param_id in range(self.num_simple_log_params):
            self.reduced_param_id_lists.append([simple_log_param_id])
            self.prop_coeff_id.append(last_prop_coeff)
            self.sym_multipliers.append(1.0)

        for sigma_id, ang_mom_classifier in enumerate(ang_mom_map):
            ang_mom1=ang_mom_classifier[0]
            ang_mom2=ang_mom_classifier[1]
            coup_mat_id=ang_mom_classifier[2]
            same_atom=ang_mom_classifier[3]

            if (same_atom and (ang_mom1 != ang_mom2)):
                cur_sym_mult=2.0
            else:
                cur_sym_mult=1.0
            self.sym_multipliers.append(cur_sym_mult)

            coup_mat_tuple=(coup_mat_id, same_atom)
            if coup_mat_tuple in prop_coeff_id_dict:
                cur_prop_coeff_id=prop_coeff_id_dict[coup_mat_tuple]
            else:
                last_prop_coeff+=1
                cur_prop_coeff_id=last_prop_coeff
                prop_coeff_id_dict[coup_mat_tuple]=cur_prop_coeff_id

            self.prop_coeff_id.append(cur_prop_coeff_id)

            self.reduced_param_id_lists.append([])
            for cur_ang_mom in [ang_mom1, ang_mom2]:
                ang_mom_tuple=(cur_ang_mom, same_atom)
                if ang_mom_tuple in red_param_id_dict:
                    cur_red_param_id=red_param_id_dict[ang_mom_tuple]
                else:
                    last_red_param+=1
                    cur_red_param_id=last_red_param
                    red_param_id_dict[ang_mom_tuple]=cur_red_param_id
                self.reduced_param_id_lists[-1].append(cur_red_param_id)

        self.num_reduced_params=last_red_param+1

        self.coup_mat_prop_coeffs=np.zeros((last_prop_coeff+1,))
        self.coup_mat_prop_coeffs[:self.num_simple_log_params]=1.0
        for sigma_id, stddev in enumerate(stddevs):
            self.coup_mat_prop_coeffs[self.prop_coeff_id[sigma_id+self.num_simple_log_params]]+=stddev**2

        self.coup_mat_prop_coeffs=self.coup_mat_prop_coeffs**(-1)

    def reduced_params_to_full(self, reduced_parameters):
        output=np.repeat(1.0, self.num_full_params)
        for param_id in range(self.num_full_params):
            for red_param_id in self.reduced_param_id_lists[param_id]:
                output[param_id]*=np.sqrt(self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]]*self.sym_multipliers[param_id])*np.exp(reduced_parameters[red_param_id])
        return output

    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        for param_id, (param_der, param_val) in enumerate(zip(full_derivatives, full_parameters)):
            for red_param_id in self.reduced_param_id_lists[param_id]:
                output[red_param_id]+=param_der*param_val
        return output

    def full_params_to_reduced(self, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        for param_id, param_val in enumerate(full_parameters):
            red_param_id_list=self.reduced_param_id_lists[param_id]
            if (len(red_param_id_list)==2):
                if red_param_id_list[0] != red_param_id_list[1]:
                    continue
            output[red_param_id_list[0]]=np.log(param_val/self.sym_multipliers[param_id]/self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]])/len(red_param_id_list)
        return output

    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        output=np.zeros((self.num_reduced_params,))
        output[0]=np.log(init_lambda)
        return output


###
# End of reduced hyperparameter states.
###

class Optimizer_state:
    def __init__(self, error_measure, error_measure_red_ders, parameters, red_parameters):
        self.error_measure=error_measure
        self.error_measure_red_ders=error_measure_red_ders
        self.parameters=parameters
        self.red_parameters=red_parameters
    def extended_greater(self, other_state):
        if self.error_measure is None:
            return True
        else:
            if other_state.error_measure is None:
                return False
            else:
                return self.error_measure>other_state.error_measure
    def __gt__(self, other_state):
        return self.extended_greater(other_state)
    def __lt__(self, other_state):
        return (not self.extended_greater(other_state))
    def log_lambda_der(self):
        return self.error_measure_red_ders[0]
    def lambda_log_val(self):
        return self.red_parameters[0]
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "Optimizer_state object: Parameters: "+str(self.parameters)+", reduced parameters: "+str(self.red_parameters)+", error measure: "+str(self.error_measure)

class Optimizer_state_generator:
    def __init__(self, goo_ensemble):
        self.goo_ensemble=goo_ensemble
    def __call__(self, red_parameters, recalc_global_matrices=True, lambda_der_only=False):
        parameters=self.goo_ensemble.reduced_hyperparam_func.reduced_params_to_full(red_parameters)
        error_measure, error_measure_red_ders=self.goo_ensemble.error_measure_wders(parameters,
                                recalc_global_matrices=recalc_global_matrices, lambda_der_only=lambda_der_only)
        return Optimizer_state(error_measure, error_measure_red_ders, parameters, red_parameters)

class GOO_randomized_iterator:
    def __init__(self, opt_GOO_ensemble, initial_reduced_parameter_vals, lambda_opt_tolerance=0.1, step_magnitudes=None, noise_level_prop_coeffs=None,
                        lambda_max_num_scan_steps=10, default_step_magnitude=1.0, keep_init_lambda=False, bisec_lambda_opt=True):

        self.optimizer_state_generator=Optimizer_state_generator(opt_GOO_ensemble)

        self.cur_optimizer_state=self.optimizer_state_generator(initial_reduced_parameter_vals)

        print("Start params:", self.cur_optimizer_state.red_parameters, "starting error measure:", self.cur_optimizer_state.error_measure)

        self.lambda_opt_tolerance=lambda_opt_tolerance
        self.keep_init_lambda=keep_init_lambda

        self.num_reduced_params=self.optimizer_state_generator.goo_ensemble.reduced_hyperparam_func.num_reduced_params

        if step_magnitudes is None:
            self.step_magnitudes=np.repeat(default_step_magnitude, self.num_reduced_params)
        else:
            self.step_magnitudes=step_magnitudes

        if noise_level_prop_coeffs is None:
            self.noise_level_prop_coeffs=np.copy(self.step_magnitudes)
        else:
            self.noise_level_prop_coeffs=noise_level_prop_coeffs

        if keep_init_lambda:
            self.noise_level_prop_coeffs[0]=0.0

        self.noise_levels=np.zeros((self.num_reduced_params,))

        self.lambda_max_num_scan_steps=lambda_max_num_scan_steps
        self.lambda_opt_tolerance=lambda_opt_tolerance

        self.lambda_max_num_scan_steps=lambda_max_num_scan_steps

        self.bisec_lambda_opt=bisec_lambda_opt

        self.change_successful=None

    def iterate(self):
        cur_red_ders=self.cur_optimizer_state.error_measure_red_ders

        normalized_red_ders=np.copy(self.cur_optimizer_state.error_measure_red_ders)
        if self.keep_init_lambda:
            normalized_red_ders[0]=0.0

        normalized_red_ders=normalized_red_ders/np.sqrt(sum((normalized_red_ders/self.step_magnitudes)**2))

        print("Normalized reduced derivatives:", normalized_red_ders)

        trial_red_params=np.copy(self.cur_optimizer_state.red_parameters)
        for param_id in range(self.num_reduced_params):
            trial_red_params[param_id]+=np.random.normal()*self.noise_levels[param_id]-normalized_red_ders[param_id]

        trial_optimizer_state=self.optimizer_state_generator(trial_red_params)

        print("Trial params:", trial_red_params, ", trial error measure:", trial_optimizer_state.error_measure)

        self.change_successful=(self.cur_optimizer_state>trial_optimizer_state)

        if self.change_successful:
            self.cur_optimizer_state=copy.deepcopy(trial_optimizer_state)
            if ((not self.keep_init_lambda) and self.bisec_lambda_opt):
                self.bisection_lambda_optimization()
            self.noise_levels[:]=0.0
        else:
            self.noise_levels+=self.noise_level_prop_coeffs/np.sqrt(self.num_reduced_params)

    def bisection_lambda_optimization(self):
        print("Bisection optimization of lambda, starting position:", self.cur_optimizer_state.lambda_log_val(), self.cur_optimizer_state.error_measure)

        # Perform a scan to approximately locate the minimum.
        prev_iteration=self.cur_optimizer_state
        prev_lambda_der=prev_iteration.log_lambda_der()
        prev_iteration=copy.deepcopy(self.cur_optimizer_state)

        scan_additive=math.copysign(self.step_magnitudes[0], -prev_lambda_der)

        bisection_interval=None

        for init_scan_step in range(self.lambda_max_num_scan_steps):
            trial_red_params=np.copy(prev_iteration.red_parameters)
            trial_red_params[0]+=scan_additive

            trial_iteration=self.optimizer_state_generator(trial_red_params, recalc_global_matrices=False, lambda_der_only=True)
            trial_lambda_der=trial_iteration.log_lambda_der()

            print("Initial lambda scan:", trial_red_params[0], trial_lambda_der, trial_iteration.error_measure)

            if ((trial_lambda_der*prev_lambda_der)<0.0):
                bisection_interval=[trial_iteration, prev_iteration]
                bisection_interval.sort(key=lambda x: x.lambda_log_val())
                break
            else:
                if trial_iteration>prev_iteration: # something is wrong, we were supposed to be going down in MSE.
                    print("WARNING: Weird behavior during lambda value scan.")
                    self.cur_optimizer_state=self.optimizer_state_generator(prev_iteration.red_parameters, recalc_global_matrices=False)
                    return
                else:
                    prev_iteration=trial_iteration

        if bisection_interval is None:
            self.cur_optimizer_state=self.optimizer_state_generator(trial_iteration.red_parameters, recalc_global_matrices=False)
            return

        # Finalize locating the minumum via bisection.
        # Use bisection search to find the minimum. Note that we do not need to recalculate the kernel matrices.
        while (bisection_interval[1].lambda_log_val()>bisection_interval[0].lambda_log_val()+self.lambda_opt_tolerance):
            cur_log_lambda=(bisection_interval[0].lambda_log_val()+bisection_interval[1].lambda_log_val())/2

            middle_params=np.copy(self.cur_optimizer_state.red_parameters)
            middle_params[0]=cur_log_lambda

            middle_iteration=self.optimizer_state_generator(middle_params, recalc_global_matrices=False, lambda_der_only=True)

            middle_der=middle_iteration.log_lambda_der()

            print("Bisection lambda optimization, lambda logarithm:", cur_log_lambda, "derivative:", middle_der, "error measure:", middle_iteration.error_measure)

            for bisec_int_id in range(2):
                if (middle_der*bisection_interval[bisec_int_id].log_lambda_der()>0.0):
                    bisection_interval[bisec_int_id]=middle_iteration
            bisection_interval.sort(key=lambda x: x.lambda_log_val())

        self.cur_optimizer_state=self.optimizer_state_generator(min(bisection_interval).red_parameters, recalc_global_matrices=False)


hyperparam_red_funcs={"default": Reduced_hyperparam_func, "single_rescaling" : Single_rescaling_rhf}

def min_sep_IBO_random_walk_optimization(compound_list, quant_list, use_Gauss=False, init_lambda=1e-3, max_iterations=None,
                                    init_param_guess=None, hyperparam_red_type="default", max_stagnating_iterations=1,
                                    use_MAE=True, rep_params=None, num_kfolds=16, other_opt_goo_ensemble_kwargs={}, randomized_iterator_kwargs={}, iter_dump_name_add=None,
                                    additional_BFGS_iters=None, iter_dump_name_add_BFGS=None, negligible_red_param_distance=1e-9, num_procs=None, num_threads=None):


    avs, stddevs=oml_ensemble_avs_stddevs(compound_list)
    print("Found stddevs:", stddevs)
    base_inv_sqwidth_params=0.25/stddevs**2

    if hyperparam_red_type == "ang_mom_classified":
        red_hyperparam_func=Ang_mom_classified_rhf(rep_params, stddevs, use_Gauss=use_Gauss)
    else:
        if hyperparam_red_type=="default":
            red_hyperparam_func=hyperparam_red_funcs[hyperparam_red_type](use_Gauss=use_Gauss)
        else:
            red_hyperparam_func=hyperparam_red_funcs[hyperparam_red_type](base_inv_sqwidth_params, use_Gauss=use_Gauss)

    if init_param_guess is None:
        if hyperparam_red_type=="default":
            base_inv_sqwidth_params=np.append(1.0, base_inv_sqwidth_params)
        initial_reduced_parameter_vals=red_hyperparam_func.initial_reduced_parameter_guess(init_lambda, base_inv_sqwidth_params)
    else:
        initial_reduced_parameter_vals=red_hyperparam_func.full_params_to_reduced(init_param_guess)

    opt_GOO_ensemble=generate_random_GOO_ensemble(compound_list, quant_list, use_Gauss=use_Gauss, use_MAE=use_MAE, num_kfolds=num_kfolds,
                                                  reduced_hyperparam_func=red_hyperparam_func, **other_opt_goo_ensemble_kwargs,
                                                  num_procs=num_procs, num_threads=num_threads)

    randomized_iterator=GOO_randomized_iterator(opt_GOO_ensemble, initial_reduced_parameter_vals, **randomized_iterator_kwargs)
    num_stagnating_iterations=0
    num_iterations=0

    iterate_more=True

    while iterate_more:
        randomized_iterator.iterate()
        cur_opt_state=randomized_iterator.cur_optimizer_state
        num_iterations+=1

        if iter_dump_name_add is not None:
            cur_dump_name=iter_dump_name_add+"_"+str(num_iterations)+".pkl"
            dump2pkl(cur_opt_state, cur_dump_name)
            
        print("Parameters:", cur_opt_state.parameters, "error measure:", cur_opt_state.error_measure)
        if max_iterations is not None:
            if num_iterations>=max_iterations:
                iterate_more=False
        if max_stagnating_iterations is not None:
            if randomized_iterator.change_successful:
                num_stagnating_iterations=0
            else:
                num_stagnating_iterations+=1
                if num_stagnating_iterations>=max_stagnating_iterations:
                    iterate_more=False

    if additional_BFGS_iters is not None:
        error_measure_func=GOO_standalone_error_measure(opt_GOO_ensemble, red_hyperparam_func, negligible_red_param_distance, iter_dump_name_add=iter_dump_name_add_BFGS)
        if iter_dump_name_add_BFGS is None:
            iter_dump_name_add_BFGS_grad=None
        else:
            iter_dump_name_add_BFGS_grad=iter_dump_name_add_BFGS+"_grad"
        error_measure_ders_func=GOO_standalone_error_measure_ders(opt_GOO_ensemble, red_hyperparam_func, negligible_red_param_distance, iter_dump_name_add=iter_dump_name_add_BFGS_grad)
        finalized_result=minimize(error_measure_func, cur_opt_state.red_parameters, method='BFGS', jac=error_measure_ders_func, options={'disp': True, 'maxiter' : additional_BFGS_iters})
        print("BFGS corrected error measure:", finalized_result.fun)
        if finalized_result.fun < cur_opt_state.error_measure:
            finalized_params=red_hyperparam_func.reduced_params_to_full(finalized_result.x)
            return {"inv_sq_width_params" : finalized_params[1:], "lambda_val" : finalized_params[0], "error_measure" : finalized_result.fun}
    return {"inv_sq_width_params" : cur_opt_state.parameters[1:], "lambda_val" : cur_opt_state.parameters[0], "error_measure" :  cur_opt_state.error_measure}


######
#   Functions introduced to facilitate coupling with standard minimization protocols from scipy.
#####

class GOO_standalone_error_measure:
    def __init__(self, GOO_ensemble, reduced_hyperparam_func, negligible_red_param_distance, iter_dump_name_add=None):
        self.GOO=GOO_ensemble
        self.reduced_hyperparam_func=reduced_hyperparam_func
        self.iter_dump_name_add=iter_dump_name_add
        self.num_calls=0
        self.negligible_red_param_distance=negligible_red_param_distance
    def __call__(self, red_parameters):
        self.parameters=self.reduced_hyperparam_func.reduced_params_to_full(red_parameters)
        self.reinit_quants()
        self.dump_intermediate_result()
        return self.result()
    def reinit_quants(self):
        self.error_measure, self.error_measure_ders=self.GOO.error_measure_wders(self.parameters, negligible_red_param_distance=self.negligible_red_param_distance)
    def dump_intermediate_result(self):
        self.num_calls+=1
        cur_dump_name=self.iter_dump_name_add+"_"+str(self.num_calls)+".pkl"
        dump2pkl([self.parameters, self.result()], cur_dump_name)
    def result(self):
        return self.error_measure

class GOO_standalone_error_measure_ders(GOO_standalone_error_measure):
    def result(self):
        return self.error_measure_ders
    

#####
# Minor auxiliary functions.
#####

def one_diag_unity_tensor(dim12, dim3):
    output=np.zeros((dim12, dim12, dim3))
    for mol_id in range(dim12):
        output[mol_id, mol_id, 0]=1.0
    return output
