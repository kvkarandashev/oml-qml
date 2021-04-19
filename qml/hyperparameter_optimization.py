import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve as scipy_cho_solve
from math import sqrt, exp
from .math import cho_solve as fcho_solve

### Some useful np-based routines. Analogous to some from qml.math,
### introduced for cases when parallelization of qml.math does not work
### properly.
def np_svd_solve(a,b,rcond=0.0):
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
#    w = np.linalg.solve(np.diag(s),c[:len(s)])
    w=np.zeros(len(s))
    for i, (cur_c, cur_s) in enumerate(zip(c[:len(s)], s)):
        if (abs(cur_s>rcond)):
            w[i]=cur_c/cur_s
    x = np.dot(v.T,w)
    return x

def np_eigh_posd_solve(mat, vec, rcond=1e-9):
    eigenvals, eigenvecs=np.linalg.eigh(mat)
    vec_transformed=np.dot(vec, eigenvecs)
    for eigenval_id, eigenval in enumerate(eigenvals):
        if eigenval > rcond:
            vec_transformed[eigenval_id]/=eigenval
        else:
            vec_transformed[eigenval_id]=0
    return np.dot(eigenvecs, vec_transformed)


def np_cho_solve(mat, vec):
    c, low=cho_factor(mat)
    return scipy_cho_solve((c, low), vec)

def np_cho_solve_wcheck(mat, vec, eigh_rcond=1e-9):
    try:
        return np_cho_solve(mat, vec)
    except np.linalg.LinAlgError:
        print("WARNING: Cholesky failed.")
        return np_eigh_posd_solve(mat, vec, rcond=eigh_rcond)

cho_solve_implementations={'qml' : fcho_solve, 'scipy' : np_cho_solve}

def cho_solve(mat, vec, cho_impl='qml'):
    return cho_solve_implementations[cho_impl](mat, vec)

class Lambda_opt_step:
    def __init__(self, lambda_val=None, cho_impl='qml'):
        self.lambda_val=lambda_val
        self.alphas=None
        self.mod_train_kernel=None
        self.cho_impl=cho_impl
        self.MAE_val=None
        self.der_MAE_val=None
        self.predicted_vals=None
    def check_mod_train_kernel(self, train_kernel):
        if self.mod_train_kernel is None:
            if self.lambda_val is None:
                self.mod_train_kernel=train_kernel
            else:
                self.mod_train_kernel=np.copy(train_kernel)
                self.mod_train_kernel[np.diag_indices_from(train_kernel)]+=self.lambda_val
    def check_alphas(self, train_kernel, train_vals):
        if self.alphas is None:
            self.check_mod_train_kernel(train_kernel)
            self.alphas=cho_solve(self.mod_train_kernel, train_vals, cho_impl=self.cho_impl)
    def check_predicted_vals(self, train_kernel, train_vals, check_kernel):
        if self.predicted_vals is None:
            self.check_alphas(train_kernel, train_vals)
            self.predicted_vals=np.dot(check_kernel, self.alphas)
    def MAE(self, train_kernel, train_vals, check_kernel, check_vals):
        if self.MAE_val is None:
            self.check_predicted_vals(train_kernel, train_vals, check_kernel)
            self.MAE_val=np.mean(np.abs(self.predicted_vals-check_vals))
        return self.MAE_val
    def der_MAE(self, train_kernel, train_vals, check_kernel, check_vals):
        if self.der_MAE_val is None:
            self.check_predicted_vals(train_kernel, train_vals, check_kernel)
            der_predicted_vals=-np.dot(check_kernel, cho_solve(self.mod_train_kernel, self.alphas, cho_impl=self.cho_impl))
            self.der_MAE_val=np.mean(der_predicted_vals*np.sign(self.predicted_vals-check_vals))
        return self.der_MAE_val
    def __str__(self):
        output="optimization_step;lambda: "+str(self.lambda_val)
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
                    initial_lambda_val=1e-6, log_diff_tol=0.01, minimal_lambda=1e-11, maximal_lambda=1.0, cho_impl='qml'):
    # Create the initial bisection interval.
    cur_scan_step=Lambda_opt_step(initial_lambda_val, cho_impl=cho_impl)
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
        next_scan_step=Lambda_opt_step(next_lambda_val, cho_impl=cho_impl)
        next_der=next_scan_step.der_MAE(train_kernel, train_vals, check_kernel, check_vals)
        next_MAE=next_scan_step.MAE(train_kernel, train_vals, check_kernel, check_vals)
        print("scanning:next_step:", next_scan_step)
        next_der_positive=(next_der>0)
        if next_der_positive != der_positive:
            if der_positive:
                bisection_interval=[next_lambda_val, cur_lambda_val]
            else:
                bisection_interval=[cur_lambda_val, next_lambda_val]
            scan_endpoint_min_MAE=min(cur_scan_step, next_scan_step, key=lambda x: x.MAE_val)
        else:
            if ((next_lambda_val<minimal_lambda) or (next_lambda_val>maximal_lambda)):
                return next_lambda_val
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
        middle_step=Lambda_opt_step(middle_lambda, cho_impl=cho_impl)
        cur_der=middle_step.der_MAE(train_kernel, train_vals, check_kernel, check_vals)
        print("bisection:middle_step:", middle_step)
        if (cur_der>0):
            updated_id=1
        else:
            updated_id=0
        bisection_interval[updated_id]=middle_lambda
    final_bisection_step=Lambda_opt_step(middle_lambda, cho_impl=cho_impl)
    final_bisection_step_MAE=final_bisection_step.MAE(train_kernel, train_vals, check_kernel, check_vals)
    print("final_bisection_step:",final_bisection_step)
    output_step=min(final_bisection_step, scan_endpoint_min_MAE, key=lambda x: x.MAE_val)
    return output_step.lambda_val, output_step.MAE_val
    
