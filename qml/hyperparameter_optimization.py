# Implements several optimization techniques based on stochastic gradient descent for conventient hyperparameter optimization.

# TO-DO 1. may be better to use as a hyperparameter not the logarithm of lambda, but logarithm of the ratio of lambda 
# and average diagonal element of the training kernel matrix. Would be more covenient for setting an upper limit for the corresponding ratio.
# 2. Frequent rewrites resulted in a lot of "transpose" functions present. Perhaps get rid of some?

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import math, random, copy
from .oml_kernels import lin_sep_IBO_kernel_conv, lin_sep_IBO_sym_kernel_conv, GMO_sep_IBO_kern_input, oml_ensemble_avs_stddevs,\
                            gauss_sep_IBO_kernel_conv, gauss_sep_IBO_sym_kernel_conv, gen_GMO_kernel_input
from .oml_representations import component_id_ang_mom_map, scalar_rep_length
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
    def __init__(self, training_compounds, training_quants, check_compounds, check_quants, training_quants_ignore=None, check_quants_ignore=None, 
                            use_Gauss=False, use_MAE=True, reduced_hyperparam_func=None, sym_kernel_func=None, kernel_func=None,
                            kernel_input_converter=gen_GMO_kernel_input, quants_ignore_orderable=False, **kernel_additional_args):
        self.init_kern_funcs(use_Gauss=use_Gauss, reduced_hyperparam_func=reduced_hyperparam_func, sym_kernel_func=sym_kernel_func,
                            kernel_func=kernel_func, kernel_input_converter=kernel_input_converter)

        self.training_compounds, self.check_compounds=kernel_input_converter(training_compounds, check_compounds)
        self.kernel_additional_args=kernel_additional_args

        self.training_quants=training_quants
        self.check_quants=check_quants

        self.training_quants_ignore=training_quants_ignore
        self.check_quants_ignore=check_quants_ignore
        self.quants_ignore_orderable=quants_ignore_orderable

        self.use_MAE=use_MAE

    def init_kern_funcs(self, use_Gauss=False, reduced_hyperparam_func=None, sym_kernel_func=None, kernel_func=None, kernel_input_converter=None):

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
        self.sigmas=parameters[1:]

    def recalculate_kernel_matrices(self):
        self.train_kernel=self.sym_kern_func(self.training_compounds, self.sigmas, **self.kernel_additional_args)
        self.check_kernel=self.def_kern_func(self.check_compounds, self.training_compounds, self.sigmas, **self.kernel_additional_args)

    def recalculate_kernel_mats_ders(self):
        train_kernel_wders=self.sym_kern_func(self.training_compounds, self.sigmas, with_ders=True, **self.kernel_additional_args)
        check_kernel_wders=self.def_kern_func(self.check_compounds, self.training_compounds, self.sigmas, with_ders=True, **self.kernel_additional_args)

        self.train_kernel=train_kernel_wders[:, :, 0]
        self.check_kernel=check_kernel_wders[:, :, 0]

        num_train_compounds=self.train_kernel.shape[0]
        num_check_compounds=self.check_kernel.shape[0]

        self.train_kernel_ders=one_diag_unity_tensor(num_train_compounds, self.num_params)
        self.check_kernel_ders=np.zeros((num_check_compounds, num_train_compounds, self.num_params))

        self.train_kernel_ders[:, :, 1:]=train_kernel_wders[:, :, 1:]
        self.check_kernel_ders[:, :, 1:]=check_kernel_wders[:, :, 1:]
        if self.reduced_hyperparam_func is not None:
            self.train_kernel_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.train_kernel_ders, self.all_parameters)
            self.check_kernel_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.check_kernel_ders, self.all_parameters)

    def reinitiate_basic_components(self):
        
        modified_train_kernel=np.copy(self.train_kernel)
        modified_train_kernel[np.diag_indices_from(modified_train_kernel)]+=self.lambda_val
        try:
            self.train_cho_decomp=Cho_multi_factors(modified_train_kernel, indices_to_ignore=self.training_quants_ignore, ignored_orderable=self.quants_ignore_orderable)
        except np.linalg.LinAlgError: # means mod_train_kernel is not invertible
            self.train_cho_decomp=None
        self.train_kern_invertible=(self.train_cho_decomp is not None)
        if self.train_kern_invertible:
            self.talphas=self.train_cho_decomp.solve_with(self.training_quants).T
            self.predictions=np.matmul(self.check_kernel, self.talphas)
            self.prediction_errors=self.predictions-self.check_quants
            nullify_ignored(self.prediction_errors, self.check_quants_ignore)

    def reinitiate_error_measures(self):
        if self.train_kern_invertible:
            self.cur_MAE=np.mean(np.abs(self.prediction_errors))
            self.cur_MSE=np.mean(self.prediction_errors**2)
        else:
            self.cur_MAE=None
            self.cur_MSE=None

    def der_predictions(self, train_der, check_der):
        output=np.matmul(train_der, self.talphas)
        output=self.train_cho_decomp.solve_with(output).T
        output=-np.matmul(self.check_kernel, output)
        output+=np.matmul(check_der, self.talphas)
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
                nullify_ignored(cur_der_predictions, self.check_quants_ignore)
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
    def __init__(self, training_indices, check_indices, all_quantities, use_MAE=True, quants_ignore=None, quants_ignore_orderable=False):
        self.training_indices=training_indices
        self.check_indices=check_indices

        if (len(all_quantities.shape)==1):
            self.training_quants=all_quantities[self.training_indices]
            self.check_quants=all_quantities[self.check_indices]
        else:
            self.training_quants=all_quantities[self.training_indices, :]
            self.check_quants=all_quantities[self.check_indices, :]

        if quants_ignore is None:
            self.training_quants_ignore=None
            self.check_quants_ignore=None
        else:
            self.training_quants_ignore=quants_ignore[self.training_indices, :]
            self.check_quants_ignore=quants_ignore[self.check_indices, :]
        
        self.use_MAE=use_MAE
        self.quants_ignore_orderable=quants_ignore_orderable
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
    def __init__(self, all_compounds, all_quantities, train_id_lists, check_id_lists, quants_ignore=None, use_Gauss=False, use_MAE=True,
                        reduced_hyperparam_func=None, num_procs=None, num_threads=None, kernel_input_converter=None,
                        sym_kernel_func=None, num_kernel_params=None, quants_ignore_orderable=False, **kernel_additional_args):

        self.init_kern_funcs(use_Gauss=use_Gauss, reduced_hyperparam_func=reduced_hyperparam_func,
                            sym_kernel_func=sym_kernel_func, kernel_func=None, kernel_input_converter=kernel_input_converter)

        self.kernel_additional_args=kernel_additional_args

        self.all_compounds=self.kernel_input_converter(all_compounds)

        self.tot_num_points=len(all_compounds)

        self.goo_ensemble_subsets=[]
        for train_id_list, check_id_list in zip(train_id_lists, check_id_lists):
            self.goo_ensemble_subsets.append(GOO_ensemble_subset(train_id_list, check_id_list,
                                                        all_quantities, quants_ignore=quants_ignore, use_MAE=use_MAE, quants_ignore_orderable=quants_ignore_orderable))
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
        global_kernel_wders=self.sym_kern_func(self.all_compounds, parameters[1:], with_ders=True, **self.kernel_additional_args)
        self.global_matrix=global_kernel_wders[:, :, 0]
        print("# GOO_ensemble: Kernel recalculated, average diagonal element:", np.mean(self.global_matrix[np.diag_indices_from(self.global_matrix)]))
        self.global_matrix_ders=one_diag_unity_tensor(self.tot_num_points, len(parameters))
        self.global_matrix_ders[:, :, 1:]=global_kernel_wders[:, :, 1:]

        if self.reduced_hyperparam_func is not None:
            self.global_matrix_ders=self.reduced_hyperparam_func.transform_der_array_to_reduced(self.global_matrix_ders, parameters)

        for subset_id in range(self.num_subsets):
            self.goo_ensemble_subsets[subset_id].recalculate_kernel_mats_ders(global_matrix=self.global_matrix, global_matrix_ders=self.global_matrix_ders)

# Auxiliary function for joblib parallelization.
def single_subset_error_measure_wders(subset, parameters, lambda_der_only):
    return subset.error_measure_wders(parameters, lambda_der_only=lambda_der_only)

def generate_random_GOO_ensemble(all_compounds, all_quantities, quants_ignore=None, num_kfolds=16, training_set_ratio=0.5, use_Gauss=False, use_MAE=True,
        reduced_hyperparam_func=None, num_procs=None, num_threads=None, kernel_input_converter=None, sym_kernel_func=None, **other_kwargs):
    num_points=len(all_compounds)
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

    return GOO_ensemble(all_compounds, all_quantities, train_id_lists, check_id_lists, quants_ignore=quants_ignore, use_Gauss=use_Gauss, use_MAE=use_MAE,
                            reduced_hyperparam_func=reduced_hyperparam_func, num_procs=num_procs, num_threads=num_threads,
                            kernel_input_converter=kernel_input_converter, sym_kernel_func=sym_kernel_func, **other_kwargs)
    

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
        self.initiate_param_nums(full_parameters)
        return np.log(full_parameters)
    def initial_reduced_parameter_guess(self, init_lambda, sigmas):
        num_full_params=len(sigmas)+1
        self.initiate_param_nums(len(sigmas)+1)
        output=np.zeros((self.num_full_params,))
        output[0]=np.log(init_lambda)
        output[-len(sigmas):]=np.log(sigmas*np.sqrt(len(sigmas)))
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
    def str_output_dict(self, global_name, output_dict=None):
        str_output=global_name+"\n"
        if output_dict is not None:
            str_output=global_name+"\n"
            for str_id in output_dict:
                str_output+=str_id+": "+str(output_dict[str_id])+"\n"
        return str_output[:-1]

    def __str__(self):
        return self.str_output_dict("Default Reduced_hyperparam_func")
    def __repr__(self):
        return str(self)
        

# For using a simple rescaling coefficient for sigmas.
class Single_rescaling_rhf(Reduced_hyperparam_func):
    def __init__(self, use_Gauss=False, ang_mom_map=None, rep_params=None, use_global_mat_prop_coeffs=False, stddevs=None):
        if use_global_mat_prop_coeffs:
            coup_mat_prop_coeffs, prop_coeff_id=matrix_grouped_prop_coeffs(stddevs=stddevs,
                                ang_mom_map=ang_mom_map, rep_params=rep_params)
            coup_sigmas=[]
            for sigma_id, cur_prop_coeff_id in enumerate(prop_coeff_id):
                coup_sigmas.append(coup_mat_prop_coeffs[cur_prop_coeff_id])
            self.coup_sigmas=np.array(coup_sigmas)
        else:
            self.coup_sigmas=stddevs
        self.num_full_params=len(self.coup_sigmas)+1
        self.num_reduced_params=2
        self.use_Gauss=use_Gauss
        self.coup_sigmas_start_id=1
        if self.use_Gauss:
            self.num_full_params+=1
            self.num_reduced_params+=1
            self.coup_sigmas_start_id+=1
    def reduced_params_to_full(self, reduced_parameters):
        output=np.zeros((self.num_full_params,))
        output[:self.coup_sigmas_start_id]=np.exp(reduced_parameters[:self.coup_sigmas_start_id])
        output[self.coup_sigmas_start_id:]=self.coup_sigmas*np.exp(reduced_parameters[-1])
        return output
    def full_derivatives_to_reduced(self, full_derivatives, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.coup_sigmas_start_id]=full_derivatives[:self.coup_sigmas_start_id]*full_parameters[:self.coup_sigmas_start_id]
        for sigma_id in range(self.coup_sigmas_start_id, self.num_full_params):
            output[-1]+=full_derivatives[sigma_id]*full_parameters[sigma_id]
        return output
    def full_params_to_reduced(self, full_parameters):
        output=np.zeros((self.num_reduced_params,))
        output[:self.coup_sigmas_start_id]=np.log(full_parameters[:self.coup_sigmas_start_id])
        est_counter=0
        for sigma_id in range(self.coup_sigmas_start_id, self.num_full_params):
            output[-1]+=np.log(full_parameters[sigma_id]/self.coup_sigmas[sigma_id-self.coup_sigmas_start_id])
            est_counter+=1
        output[-1]/=est_counter
        return output
    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        init_resc_param_guess=np.log(self.num_full_params-2)
        if self.use_Gauss:
            return np.array([np.log(init_lambda), 0.0, init_resc_param_guess])
        else:
            return np.array([np.log(init_lambda), init_resc_param_guess])
    def __str__(self):
        vals_names={"num_full_params" : self.num_full_params,
                    "num_reduced_params" : self.num_reduced_params,
                    "coup_sigmas" : self.coup_sigmas}
        return self.str_output_dict("Single_rescaling_rhf", vals_names)
        

def matrix_grouped_prop_coeffs(rep_params=None, stddevs=None, ang_mom_map=None, forward_ones=0, return_sym_multipliers=False):
    if ang_mom_map is None:
        if rep_params is None:
            raise Exception
        else:
            ang_mom_map=component_id_ang_mom_map(rep_params)
    prop_coeff_id_dict={}

    sym_multipliers=[]
    prop_coeff_id=[]

    if forward_ones != 0:
        last_prop_coeff=0
        for forward_id in range(forward_ones):
            prop_coeff_id.append(last_prop_coeff)
            sym_multipliers.append(1.0)
    else:
        last_prop_coeff=-1

    for sigma_id, ang_mom_classifier in enumerate(ang_mom_map):
        ang_mom1=ang_mom_classifier[0]
        ang_mom2=ang_mom_classifier[1]
        coup_mat_id=ang_mom_classifier[2]
        same_atom=ang_mom_classifier[3]

        if (same_atom and (ang_mom1 != ang_mom2)):
            cur_sym_mult=2.0
        else:
            cur_sym_mult=1.0
        sym_multipliers.append(cur_sym_mult)

        coup_mat_tuple=(coup_mat_id, same_atom)
        try:
            cur_prop_coeff_id=prop_coeff_id_dict[coup_mat_tuple]
        except KeyError:
            last_prop_coeff+=1
            cur_prop_coeff_id=last_prop_coeff
            prop_coeff_id_dict[coup_mat_tuple]=cur_prop_coeff_id
        prop_coeff_id.append(cur_prop_coeff_id)

    coup_mat_prop_coeffs=np.zeros((last_prop_coeff+1,))
    norm_coeffs=np.zeros((last_prop_coeff+1,))
    if stddevs is None:
        coup_mat_prop_coeffs[:]=1.0
    else:
        if forward_ones == 0:
            coup_mat_true_start=0
        else:
            coup_mat_true_start=1
            coup_mat_prop_coeffs[0]=1.0
        for sigma_id, stddev in enumerate(stddevs):
            true_sigma_id=sigma_id+forward_ones
            cur_sym_coeff=sym_multipliers[true_sigma_id]
            cur_prop_coeff_id=prop_coeff_id[true_sigma_id]
            norm_coeffs[cur_prop_coeff_id]+=cur_sym_coeff
            coup_mat_prop_coeffs[cur_prop_coeff_id]+=stddev**2*cur_sym_coeff
        coup_mat_prop_coeffs[coup_mat_true_start:]=np.sqrt(coup_mat_prop_coeffs[coup_mat_true_start:]/norm_coeffs[coup_mat_true_start:])

    if return_sym_multipliers:
        return coup_mat_prop_coeffs, prop_coeff_id, sym_multipliers
    else:
        return coup_mat_prop_coeffs, prop_coeff_id


class Ang_mom_classified_rhf(Reduced_hyperparam_func):
    def __init__(self, rep_params=None, stddevs=None, ang_mom_map=None, use_Gauss=False):
        if ang_mom_map is None:
            if rep_params is None:
                raise Exception("No rep_params defined for Ang_mom_classified_rhf class.")
            else:
                ang_mom_map=component_id_ang_mom_map(rep_params)

        self.num_simple_log_params=1
        if use_Gauss:
            self.num_simple_log_params+=1
        self.num_full_params=self.num_simple_log_params+len(ang_mom_map)

        self.coup_mat_prop_coeffs, self.prop_coeff_id, self.sym_multipliers=matrix_grouped_prop_coeffs(ang_mom_map=ang_mom_map, stddevs=stddevs,
                                                                        forward_ones=self.num_simple_log_params, return_sym_multipliers=True)

        red_param_id_dict={}

        last_red_param=self.num_simple_log_params-1

        self.reduced_param_id_lists=[]

        for simple_log_param_id in range(self.num_simple_log_params):
            self.reduced_param_id_lists.append([simple_log_param_id])

        for sigma_id, ang_mom_classifier in enumerate(ang_mom_map):
            ang_mom1=ang_mom_classifier[0]
            ang_mom2=ang_mom_classifier[1]
            same_atom=ang_mom_classifier[3]

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

    def reduced_params_to_full(self, reduced_parameters):
        output=np.repeat(1.0, self.num_full_params)
        for param_id in range(self.num_full_params):
            for red_param_id in self.reduced_param_id_lists[param_id]:
                # TO-DO does this choice of sym_multipliers make sense?
                output[param_id]*=np.sqrt(self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]]/
                                    np.sqrt(self.sym_multipliers[param_id]))*np.exp(reduced_parameters[red_param_id])
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
            output[red_param_id_list[0]]=np.log(param_val*np.sqrt(self.sym_multipliers[param_id])/self.coup_mat_prop_coeffs[self.prop_coeff_id[param_id]])/len(red_param_id_list)
        return output

    def initial_reduced_parameter_guess(self, init_lambda, *other_args):
        output=np.zeros((self.num_reduced_params,))
        output[0]=np.log(init_lambda)
        return output

    def __str__(self):
        vals_names={"num_reduced_params" : self.num_reduced_params,
                    "reduced_param_id_lists" : self.reduced_param_id_lists,
                    "sym_multipliers" : self.sym_multipliers,
                    "coup_mat_prop_coeffs" : self.coup_mat_prop_coeffs,
                    "prop_coeff_id" : self.prop_coeff_id}
        return self.str_output_dict("Ang_mom_classified_rhf", vals_names)

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
                        lambda_max_num_scan_steps=256, default_step_magnitude=1.0, keep_init_lambda=False, bisec_lambda_opt=True, max_lambda_diag_el_ratio=None,
                        min_lambda_diag_el_ratio=1e-12):

        self.optimizer_state_generator=Optimizer_state_generator(opt_GOO_ensemble)

        self.cur_optimizer_state=self.optimizer_state_generator(initial_reduced_parameter_vals)

        print("Start params:", self.cur_optimizer_state.red_parameters, "starting error measure:", self.cur_optimizer_state.error_measure)

        self.lambda_opt_tolerance=lambda_opt_tolerance
        self.keep_init_lambda=keep_init_lambda
        self.negligible_lambda_log=-40.0

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
        
        self.max_lambda_diag_el_ratio=max_lambda_diag_el_ratio
        self.min_lambda_diag_el_ratio=min_lambda_diag_el_ratio

        # If initial lambda value is not large enough for the kernel matrix to be invertible,
        # use bisection to optimize it.
        if (not self.keep_init_lambda) and (self.max_lambda_diag_el_ratio is not None):
            while self.cur_optimizer_state.error_measure is None:
                if not self.keep_init_lambda:
                    self.bisection_lambda_optimization()

    def iterate(self):

        trial_red_params=self.generate_trial_red_params(self.keep_init_lambda)

        if self.lambda_outside_bounds(trial_red_params[0]):
            trial_red_params=self.generate_trial_red_params(True)

        trial_optimizer_state=self.optimizer_state_generator(trial_red_params)

        print("Trial params:", trial_red_params, ", trial error measure:", trial_optimizer_state.error_measure)

        self.change_successful=(self.cur_optimizer_state>trial_optimizer_state)

        if self.change_successful:
            self.cur_optimizer_state=copy.deepcopy(trial_optimizer_state)
            if ((not self.keep_init_lambda) and self.bisec_lambda_opt):
                self.bisection_lambda_optimization()
            self.noise_levels[:]=0.0
            if self.lambda_outside_bounds():
                self.change_lambda_until_normal()
        else:
            self.noise_levels+=self.noise_level_prop_coeffs/np.sqrt(self.num_reduced_params)

    def generate_trial_red_params(self, keep_init_lambda):
        normalized_red_ders=np.copy(self.cur_optimizer_state.error_measure_red_ders)
        if keep_init_lambda:
            normalized_red_ders[0]=0.0
        normalized_red_ders=normalized_red_ders/np.sqrt(sum((normalized_red_ders/self.step_magnitudes)**2))

        print("Normalized reduced derivatives:", normalized_red_ders)

        trial_red_params=np.copy(self.cur_optimizer_state.red_parameters)

        if keep_init_lambda:
            param_range_start=1
        else:
            param_range_start=0

        for param_id in range(param_range_start, self.num_reduced_params):
            trial_red_params[param_id]+=np.random.normal()*self.noise_levels[param_id]-normalized_red_ders[param_id]

        return trial_red_params

    def change_lambda_until_normal(self):
        scan_additive=np.abs(self.step_magnitudes[0])
        if self.lambda_outside_bounds(sign_determinant=scan_additive):
            scan_additive*=-1
        new_red_params=np.copy(self.cur_optimizer_state.red_parameters)

        while self.lambda_outside_bounds(new_red_params[0]):
            trial_red_params=np.copy(new_red_params)
            trial_red_params[0]+=scan_additive
            trial_iteration=self.optimizer_state_generator(trial_red_params, recalc_global_matrices=False, lambda_der_only=True)

            trial_error_measure=trial_iteration.error_measure
            print("Scanning lambda to normalize:", trial_red_params[0], trial_error_measure)

            if trial_error_measure is None:
                break
            else:
                new_red_params=np.copy(trial_red_params)

        self.recalc_cur_opt_state(new_red_params, recalc_global_matrices=False)

    def recalc_cur_opt_state(self, new_red_params, recalc_global_matrices=True, lambda_der_only=False):
        self.cur_optimizer_state=self.optimizer_state_generator(new_red_params, recalc_global_matrices=recalc_global_matrices,
                                        lambda_der_only=lambda_der_only)

    def lambda_outside_bounds(self, log_lambda_value=None, sign_determinant=None):
        if (self.max_lambda_diag_el_ratio is None) and (self.min_lambda_diag_el_ratio is None):
            return False
        else:
            if log_lambda_value is None:
                log_lambda_value=self.cur_optimizer_state.red_parameters[0]
            lambda_value=np.exp(log_lambda_value)
            train_kernel_mat=self.optimizer_state_generator.goo_ensemble.global_matrix
            av_kernel_diag_el=np.mean(train_kernel_mat[np.diag_indices_from(train_kernel_mat)])
            if self.max_lambda_diag_el_ratio is None:
                too_large=False
            else:
                too_large=(lambda_value>av_kernel_diag_el*self.max_lambda_diag_el_ratio)
            if self.min_lambda_diag_el_ratio is None:
                too_small=False
            else:
                too_small=(lambda_value<av_kernel_diag_el*self.min_lambda_diag_el_ratio)
            
            if sign_determinant is None:
                return (too_small or too_large)
            else:
                if sign_determinant>0.0:
                    return too_large
                else:
                    return too_small

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

            if (self.lambda_outside_bounds(log_lambda_value=trial_red_params[0], sign_determinant=scan_additive)):
                self.recalc_cur_opt_state(prev_iteration.red_parameters, recalc_global_matrices=False)
                return

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
                    self.recalc_cur_opt_state(prev_iteration.red_parameters, recalc_global_matrices=False)
                    return
                else:
                    prev_iteration=trial_iteration

        if bisection_interval is None:
            self.recalc_cur_opt_state(trial_iteration.red_parameters, recalc_global_matrices=False)
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

        self.recalc_cur_opt_state(min(bisection_interval).red_parameters, recalc_global_matrices=False)

list_supported_funcs=["default", "single_rescaling", "single_rescaling_global_mat_prop_coeffs", "ang_mom_classified"]

def min_sep_IBO_random_walk_optimization(compound_list, quant_list, quant_ignore_list=None, use_Gauss=False, quants_ignore_orderable=False, init_lambda=1e-3, max_iterations=256,
                                    init_param_guess=None, hyperparam_red_type="default", max_stagnating_iterations=1,
                                    use_MAE=True, rep_params=None, num_kfolds=16, other_opt_goo_ensemble_kwargs={}, randomized_iterator_kwargs={}, iter_dump_name_add=None,
                                    additional_BFGS_iters=None, iter_dump_name_add_BFGS=None, negligible_red_param_distance=1e-9, num_procs=None, num_threads=None,
                                    kernel_input_converter=None, sym_kernel_func=None, assumed_stddevs=None):

#   TO-DO make array of ones created for single rescaling of scalar representation length? Add function for oml_representation? that determines length of representation
    if hyperparam_red_type != "default":
        if rep_params is None:
            need_stddevs=True
        else:
            need_stddevs=(not rep_params.propagator_coup_mat)
        if need_stddevs:
            if assumed_stddevs is None:
                avs, stddevs=oml_ensemble_avs_stddevs(compound_list)
            else:
                stddevs=assumed_stddevs
            print("Found stddevs:", stddevs)
        else:
            stddevs=np.repeat(1.0, scalar_rep_length(compound_list[0]))

    if hyperparam_red_type not in list_supported_funcs:
        raise Exception
    if hyperparam_red_type=="default":
        red_hyperparam_func=Reduced_hyperparam_func(use_Gauss=use_Gauss)
    if hyperparam_red_type == "ang_mom_classified":
        red_hyperparam_func=Ang_mom_classified_rhf(rep_params=rep_params, stddevs=stddevs, use_Gauss=use_Gauss)
    if hyperparam_red_type=="single_rescaling":
        red_hyperparam_func=Single_rescaling_rhf(stddevs=stddevs, use_Gauss=use_Gauss,
                                            rep_params=rep_params, use_global_mat_prop_coeffs=False)
    if hyperparam_red_type=="single_rescaling_global_mat_prop_coeffs":
        red_hyperparam_func=Single_rescaling_rhf(stddevs=stddevs, use_Gauss=use_Gauss,
                                            rep_params=rep_params, use_global_mat_prop_coeffs=True)

    if init_param_guess is None:
        if hyperparam_red_type=="default":
            stddevs=np.append(1.0, stddevs)
        initial_reduced_parameter_vals=red_hyperparam_func.initial_reduced_parameter_guess(init_lambda, stddevs)
    else:
        initial_reduced_parameter_vals=red_hyperparam_func.full_params_to_reduced(init_param_guess)

    opt_GOO_ensemble=generate_random_GOO_ensemble(compound_list, quant_list, quants_ignore=quant_ignore_list, use_Gauss=use_Gauss, use_MAE=use_MAE, num_kfolds=num_kfolds,
                                                  reduced_hyperparam_func=red_hyperparam_func, **other_opt_goo_ensemble_kwargs, quants_ignore_orderable=quants_ignore_orderable,
                                                  num_procs=num_procs, num_threads=num_threads, kernel_input_converter=kernel_input_converter,
                                                  sym_kernel_func=sym_kernel_func)

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
            return {"sigmas" : finalized_params[1:], "lambda_val" : finalized_params[0], "error_measure" : finalized_result.fun}
    return {"sigmas" : cur_opt_state.parameters[1:], "lambda_val" : cur_opt_state.parameters[0], "error_measure" :  cur_opt_state.error_measure}


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
# Auxiliary functions.
#####

def one_diag_unity_tensor(dim12, dim3):
    output=np.zeros((dim12, dim12, dim3))
    for mol_id in range(dim12):
        output[mol_id, mol_id, 0]=1.0
    return output
    
# For dealing with several Cholesky decompositions at once.
def where2slice(indices_to_ignore):
    return np.where(np.logical_not(indices_to_ignore))[0]

def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(np.logical_not(cur_ignore_indices))]=0.0

class Cho_multi_factors:
    def __init__(self, train_kernel, indices_to_ignore=None, ignored_orderable=False):
        self.indices_to_ignore=indices_to_ignore
        self.ignored_orderable=ignored_orderable
        self.single_cho_decomp=(indices_to_ignore is None)
        if not self.single_cho_decomp:
            self.single_cho_decomp=(not self.indices_to_ignore.any())
        if self.single_cho_decomp:
            self.cho_factors=[cho_factor(train_kernel)]
        else:
            if self.ignored_orderable:
                ignored_nums=[]
                for i, cur_ignored in enumerate(self.indices_to_ignore):
                    ignored_nums.append((i, np.sum(cur_ignored)))
                ignored_nums.sort(key = lambda x : x[1])
                self.availability_order=np.array([i[0] for i in ignored_nums])
                self.avail_quant_nums=[self.indices_to_ignore.shape[0]-np.sum(cur_ignored) for cur_ignored in self.indices_to_ignore.T]
                self.cho_factors=[cho_factor(train_kernel[self.availability_order, :][:, self.availability_order])]
            else:
                self.cho_factors=[]
                for cur_ignore_ids in self.indices_to_ignore.T:
                    s=where2slice(cur_ignore_ids)
                    self.cho_factors.append(cho_factor(train_kernel[s, :][:, s]))
    def solve_with(self, rhs):
        if len(rhs.shape)==1:
            assert(self.single_cho_decomp)
            cycled_rhs=np.array([rhs])
        else:
            if not self.single_cho_decomp:
                if self.ignored_orderable:
                    assert(len(self.avail_quant_nums)==rhs.shape[1])
                else:
                    assert(len(self.cho_factors)==rhs.shape[1])
            cycled_rhs=rhs.T
        output=np.zeros(cycled_rhs.shape)
        for rhs_id, rhs_component in enumerate(cycled_rhs):
            if self.indices_to_ignore is None:
                included_indices=np.array(range(len(rhs_component)))
            else:
                if self.ignored_orderable:
                    included_indices=self.availability_order[:self.avail_quant_nums[rhs_id]]
                else:
                    included_indices=where2slice(self.indices_to_ignore[:, rhs_id])
            if self.single_cho_decomp:
                cur_decomp=self.cho_factors[0]
            else:
                if self.ignored_orderable:
                    cur_decomp=(self.cho_factors[0][0][:self.avail_quant_nums[rhs_id], :][:, :self.avail_quant_nums[rhs_id]], self.cho_factors[0][1])
                else:
                    cur_decomp=self.cho_factors[rhs_id]
            output[rhs_id, included_indices]=cho_solve(cur_decomp, rhs_component[included_indices])
        if len(rhs.shape)==1:
            return output[0]
        else:
            return output

