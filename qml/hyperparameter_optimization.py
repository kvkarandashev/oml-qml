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
        self.der_MAE_val=None
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
    def der_MAE(self, train_kernel, train_vals, check_kernel, check_vals):
        if self.der_MAE_val is None:
            self.check_predicted_vals(train_kernel, train_vals, check_kernel)
            if self.cho_decomp_valid:
                der_predicted_vals=-np.dot(check_kernel, cho_solve(self.cho_decomp, self.alphas))
                self.der_MAE_val=np.mean(der_predicted_vals*np.sign(self.predicted_vals-check_vals))
            else:
                self.der_MAE_val=-1.0 # need to increase lambda.
        return self.der_MAE_val
    def __str__(self):
        output="optimization_step;lambda: "+str(self.lambda_val)
        if self.cho_decomp_valid is False:
            output+=" ,Cholesky failed"
        output+=kwdstr("MAE", self.MAE_val)
        output+=kwdstr("der_MAE", self.der_MAE_val)
        return output
    def __repr__(self):
        return str(self)

def kwdstr(keyword, value):
    if value is None:
        return ""
    else:
        return " ,"+keyword+": "+str(value)

def optimized_lambda_MAE(train_kernel, train_vals, check_kernel, check_vals, scan_multiplier=2.0,
                    initial_lambda_val=1e-6, log_diff_tol=0.01, minimal_lambda=1e-11, maximal_lambda=1.0):
    if scan_multiplier <= 1.0:
        raise Exception
    # Create the initial bisection interval.
    cur_scan_step=Lambda_opt_step(initial_lambda_val)
    cur_der=cur_scan_step.der_MAE(train_kernel, train_vals, check_kernel, check_vals)
    cur_MAE=cur_scan_step.MAE(train_kernel, train_vals, check_kernel, check_vals)
    der_positive=(cur_der>0)
    if der_positive:
        scan_multiplier**=-1
    bisection_interval=[]
    cur_lambda_val=initial_lambda_val
    print("starting_lambda_step:", cur_scan_step)
    while (len(bisection_interval)==0):
        next_lambda_val=cur_lambda_val*scan_multiplier
        next_scan_step=Lambda_opt_step(next_lambda_val)
        next_der=next_scan_step.der_MAE(train_kernel, train_vals, check_kernel, check_vals)
        next_MAE=next_scan_step.MAE(train_kernel, train_vals, check_kernel, check_vals)
        print("scanning:next_step:", next_scan_step)
        next_der_positive=(next_der>0)
        if next_der_positive != der_positive:
            if der_positive:
                bisection_interval=[next_lambda_val, cur_lambda_val]
            else:
                bisection_interval=[cur_lambda_val, next_lambda_val]
        else:
            if ((next_lambda_val<minimal_lambda) or (next_lambda_val>maximal_lambda)):
                return next_lambda_val, next_MAE
            if (cur_MAE is not None) and (next_MAE is not None):
                if cur_MAE<next_MAE:
                    print("WARNING: bisection aborted, derivative is noisy.")
                    return cur_lambda_val, cur_MAE
            cur_lambda_val=next_lambda_val
            cur_MAE=next_MAE
            cur_scan_step=next_scan_step
    # Do the bisection search.
    log_diff_tol_mult=exp(log_diff_tol)
    while (bisection_interval[1]>bisection_interval[0]*log_diff_tol_mult):
        middle_lambda=sqrt(bisection_interval[0]*bisection_interval[1])
        middle_step=Lambda_opt_step(middle_lambda)
        cur_der=middle_step.der_MAE(train_kernel, train_vals, check_kernel, check_vals)
        print("bisection:middle_step:", middle_step)
        if (cur_der>0):
            updated_id=1
        else:
            updated_id=0
        bisection_interval[updated_id]=middle_lambda
    final_bisection_step=Lambda_opt_step(middle_lambda)
    final_bisection_step_MAE=final_bisection_step.MAE(train_kernel, train_vals, check_kernel, check_vals)
    print("final_bisection_step:",final_bisection_step)
    return middle_lambda, final_bisection_step_MAE
    
