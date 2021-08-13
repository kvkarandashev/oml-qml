import numpy as np
from .factive_learning import fmetadynamics_active_learning_order

def metadynamics_active_learning_order(sym_kernel_mat, initial_ordered_size=0, num_to_generate=None):
    assert(len(sym_kernel_mat.shape)==2)
    num_samples=sym_kernel_mat.shape[0]
    assert(num_samples==sym_kernel_mat.shape[1])
    assert(num_samples>initial_ordered_size)
    if num_to_generate is None:
        num_to_generate=num_samples
    output_indices=np.zeros((num_to_generate,), dtype=np.int32)
    fmetadynamics_active_learning_order(sym_kernel_mat.T, num_samples, initial_ordered_size, num_to_generate, output_indices)
    return output_indices
