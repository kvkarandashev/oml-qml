import numpy as np
from scipy.linalg import cho_factor, cho_solve
import math, random, copy
from .oml_kernels import lin_sep_IBO_kernel_conv, lin_sep_IBO_sym_kernel_conv, GMO_sep_IBO_kern_input
from scipy.optimize import minimize

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



class Gradient_optimization_obj:
    def __init__(self, training_compounds, training_quants, check_compounds, check_quants, density_neglect=1e-9):
        self.training_compounds=GMO_sep_IBO_kern_input(training_compounds)
        self.num_params=self.training_compounds.max_num_scalar_reps+1

        self.check_compounds=GMO_sep_IBO_kern_input(check_compounds)

        self.training_quants=training_quants
        self.check_quants=check_quants
        self.density_neglect=density_neglect
    def MSE(self, parameters):
        self.reinitiate_matrices(parameters)
        return np.mean((self.predictions-self.check_quants)**2)

    def MSE_der(self, parameters):
        self.reinitiate_mats_ders(parameters)
        output=np.empty((self.num_params,), float)
        for param_id in range(self.num_params):
            cur_train_der=self.train_kernel_ders[:, :, param_id]
            cur_check_der=self.check_kernel_ders[:, :, param_id]
            output[param_id]=2*np.mean((self.predictions-self.check_quants)*self.der_predictions(cur_train_der, cur_check_der))
        return output

    def der_predictions(self, train_der, check_der):
        output=np.matmul(train_der, self.alphas)
        output=cho_solve(self.train_cho_decomp, output)
        output=-np.matmul(self.check_kernel, output)
        output+=np.matmul(check_der, self.alphas)
        return output

    def reinitiate_basic_params(self, parameters):
        self.lambda_val=parameters[0]
        self.inv_sq_width_params=parameters[1:]
        self.num_ders=len(parameters)

    def reinitiate_basic_components(self, train_kernel):
        train_kernel[np.diag_indices_from(train_kernel)]+=self.lambda_val
        self.train_cho_decomp=cho_factor(train_kernel)
        self.alphas=cho_solve(self.train_cho_decomp, self.training_quants)
        self.predictions=np.matmul(self.check_kernel, self.alphas)

    def reinitiate_matrices(self, parameters):
        self.reinitiate_basic_params(parameters)
        train_kernel=lin_sep_IBO_sym_kernel_conv(self.training_compounds, self.inv_sq_width_params, self.density_neglect)
        self.check_kernel=lin_sep_IBO_kernel_conv(self.check_compounds, self.training_compounds, self.inv_sq_width_params, self.density_neglect)
        self.reinitiate_basic_components(train_kernel)

    def reinitiate_mats_ders(self, parameters):
        self.reinitiate_basic_params(parameters)
        train_kernel_wders=lin_sep_IBO_sym_kernel_conv(self.training_compounds, self.inv_sq_width_params, self.density_neglect, with_ders=True)
        check_kernel_wders=lin_sep_IBO_kernel_conv(self.check_compounds, self.training_compounds, self.inv_sq_width_params, self.density_neglect, with_ders=True)

        train_kernel=train_kernel_wders[:, :, 0]
        self.check_kernel=check_kernel_wders[:, :, 0]
        num_train_compounds=self.training_compounds.num_mols
        self.train_kernel_ders=np.zeros((num_train_compounds, num_train_compounds, self.num_params))
        for mol_id in range(num_train_compounds):
            self.train_kernel_ders[mol_id, mol_id, 0]=1.0
        self.check_kernel_ders=np.zeros((self.check_compounds.num_mols, num_train_compounds, self.num_params))

        self.train_kernel_ders[:, :, 1:]=train_kernel_wders[:, :, 1:]
        self.check_kernel_ders[:, :, 1:]=check_kernel_wders[:, :, 1:]

        self.reinitiate_basic_components(train_kernel)

def min_sep_IBO_MSE_train_defined(training_compounds, training_quants, check_compounds, check_quants, init_lambda_guess, init_inv_sq_width_guess):
    go_obj=Gradient_optimization_oBj(training_compounds, training_quants, check_compounds, check_quants)
    init_param_guess=np.zeros((len(init_inv_sq_width_guess+1),))
    init_param_guess[0]=init_lambda_guess
    init_param_guess[1:]=np.array(init_inv_sq_width_guess)
    return minimize(Gradient_optimization_obj.MSE, init_param_guess, method='BFGS', jac=Gradient_optimization_obj.MSE_der, options={'disp': True})


def min_sep_IBO_MSE(compound_list, quant_list, init_lambda_guess=0.001, training_set_ratio=0.5, initial_guess_sigma_rescale=1.0):
    orb_sample=random_ibo_sample(compound_list, pair_reps=True)
    stddevs=oml_ensemble_widths_estimate(orb_sample)
    init_inv_sq_width_guess=0.25/(initial_guess_sigma_rescale*stddevs)**2

    comp_list_copy=copy.deepcopy(compound_list)
    quant_list_copy=copy.deepcopy(quant_list)
    comp_quant_list=list(zip(comp_list_copy, quant_list_copy))
    divisor=int(len(compound_list)*training_set_ratio)
    random.shuffle(comp_quant_list)
    comp_list_shuffled, quant_list_shuffled = zip(*comp_quant_list)

    return min_sep_IBO_MSE_train_defined(comp_list_shuffled[:divisor], quant_list_shuffled[:divisor], comp_list_shuffled[divisor:], quant_list_shuffled[divisor:], init_lambda_guess, init_inv_sq_width_guess)

