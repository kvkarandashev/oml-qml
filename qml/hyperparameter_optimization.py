import numpy as np
from scipy.linalg import cho_factor, cho_solve
from math import sqrt, exp

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
    log_diff_tol_mult=exp(bisec_log_diff_tol)
    print('bisection interval:', bisection_interval)
    while (bisection_interval[1]>bisection_interval[0]*log_diff_tol_mult):
        middle_lambda=sqrt(bisection_interval[0]*bisection_interval[1])
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
