import numpy as np
from .factive_learning import fmetadynamics_active_learning_order, ffeature_distance_learning_order

learning_order_functions={"metadynamics" : fmetadynamics_active_learning_order, "feature_distance" : ffeature_distance_learning_order}

def active_learning_order(sym_kernel_mat, starting_indices=None, num_to_generate=None, active_learning_method="metadynamics", lambda_val=None,
                            covariance_relative_tolerance=0.2, orthog_sqnorm_tol=0.0):
    assert(len(sym_kernel_mat.shape)==2)
    num_samples=sym_kernel_mat.shape[0]
    if num_to_generate is None:
        num_to_generate=num_samples
    output_indices=np.zeros((num_to_generate,), dtype=np.int32)
    if starting_indices is None:
        initial_ordered_size=0
    else:
        initial_ordered_size=len(starting_indices)
        assert(num_samples>initial_ordered_size)
        output_indices[:initial_ordered_size]=np.array(starting_indices)[:initial_ordered_size]
        output_indices+=1 # because we'll be using it in Fortran
        output_indices[:initial_ordered_size].sort()

    assert(num_samples==sym_kernel_mat.shape[1])    
    
    used_kernel_mat=np.copy(sym_kernel_mat)
    if lambda_val is not None:
        used_kernel_mat[np.diag_indices_from(used_kernel_mat)]+=lambda_val
    norm_coeffs=1.0/np.sqrt(used_kernel_mat[np.diag_indices_from(used_kernel_mat)])
    for i in range(num_samples):
        used_kernel_mat[i, :]*=norm_coeffs
        used_kernel_mat[:, i]*=norm_coeffs
    learning_order_functions[active_learning_method](used_kernel_mat.T, num_samples, initial_ordered_size,
                                num_to_generate, covariance_relative_tolerance, orthog_sqnorm_tol, output_indices)
    return output_indices
    
