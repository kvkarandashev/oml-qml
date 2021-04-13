import numpy as np
from scipy.linalg import cho_factor, cho_solve
from math import sqrt, exp

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


def np_cho_solve(mat, vec, eigh_rcond=1e-9):
    try:
        c, low=cho_factor(mat)
        return cho_solve((c, low), vec)
    except np.linalg.LinAlgError:
        print("WARNING: Cholesky failed.")
        return np_eigh_posd_solve(mat, vec, rcond=eigh_rcond)


#   Functions for optimizing hyperparameters.

#   Optimize lambda.
def optimize_lambda(train_kernel, train_vals, check_kernel, check_vals, initial_lambda_val=1e-6, log_diff_tol=0.01):
    # Create the initial bisection interval.
    cur_der=MAE_der(train_kernel, train_vals, check_kernel, check_vals, lambda_val=initial_lambda_val)
    der_positive=(cur_der>0)
    if der_positive:
        multiplier=0.5
    else:
        multiplier=2.0
    bisection_interval=[]
    cur_lambda_val=initial_lambda_val
    while len(bisection_interval)==0:
        next_lambda_val=cur_lambda_val*multiplier
        next_der=MAE_der(train_kernel, train_vals, check_kernel, check_vals, lambda_val=next_lambda_val)
        next_der_positive=(next_der>0)
        if next_der_positive != der_positive:
            if der_positive:
                bisection_interval=[next_lambda_val, cur_lambda_val]
            else:
                bisection_interval=[cur_lambda_val, next_lambda_val]
        else:
            cur_lambda_val=next_lambda_val
    # Do the bisection search.
    log_diff_tol_mult=exp(log_diff_tol)
    while (bisection_interval[1]>bisection_interval[0]*log_diff_tol_mult):
        middle_lambda=sqrt(bisection_interval[0]*bisection_interval[1])
        cur_der=MAE_der(train_kernel, train_vals, check_kernel, check_vals, lambda_val=middle_lambda)
        if (cur_der>0):
            updated_id=1
        else:
            updated_id=0
        bisection_interval[updated_id]=middle_lambda
    return middle_lambda

def MAE_der(train_kernel, train_vals, check_kernel, check_vals, lambda_val=None, return_predicted=False, return_alphas=False, **additional_kwargs):
    der_pred_vals, alphas=der_predicted_vals(train_kernel, train_vals, check_kernel, lambda_val=lambda_val, return_alphas=True, **additional_kwargs)
    predicted_vals=np.dot(check_kernel, alphas)
    total_der=np.mean(der_pred_vals*np.sign(predicted_vals-check_vals))
    if return_predicted:
        if return_alphas:
            return total_der, predicted_vals, alphas
        else:
            return total_der, predicted_vals
    else:
        return total_der

def der_predicted_vals(train_kernel, train_vals, check_kernel, lambda_val=None, train_kernel_deriv=None, check_kernel_deriv=None,
                        return_alphas=False):
    if lambda_val is None:
        tk_wlambda=train_kernel
    else:
        tk_wlambda=np.copy(train_kernel)
        tk_wlambda[np.diag_indices_from(tk_wlambda)]+=lambda_val
    alphas=np_cho_solve(tk_wlambda, train_vals)
    if train_kernel_deriv is None:
        cho_rhs=alphas
    else:
        cho_rhs=np.dot(train_kernel_deriv, alphas)
    der_predicted_vals=-np.dot(check_kernel, np_cho_solve(tk_wlambda, cho_rhs))
    if check_kernel_deriv is not None:
        der_predicted_vals+=np.dot(check_kernel_deriv, alphas)
    if return_alphas:
        return der_predicted_vals, alphas
    else:
        return der_predicted_vals
        
