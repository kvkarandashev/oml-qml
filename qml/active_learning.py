import numpy as np
from .factive_learning import fmetadynamics_active_learning_order, ffeature_distance_learning_order, flinear_dependent_entries
from numba import njit, prange
from numba.types import bool_

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
        assert(num_samples>=initial_ordered_size)
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
    

@njit(fastmath=True)
def all_indices_except(to_include):
    num_left=0
    for el in to_include:
        if not el:
            num_left+=1
    output=np.zeros((num_left,), dtype=np.int32)
    arr_pos=0
    for el_id, el in enumerate(to_include):
        if not el:
            print("Skipped: ", el_id)
            output[arr_pos]=el_id
            arr_pos+=1
    return output[:arr_pos]

@njit(fastmath=True, parallel=True)
def numba_linear_dependent_entries(train_kernel, residue_tol_coeff):
    num_elements=train_kernel.shape[0]

    sqnorm_residue=np.zeros(num_elements)
    residue_tolerance=np.zeros(num_elements)

    for i in prange(num_elements):
        sqnorm=train_kernel[i, i]
        sqnorm_residue[i]=sqnorm
        residue_tolerance[i]=sqnorm*residue_tol_coeff

    cur_orth_id=0

    to_include=np.ones(num_elements, dtype=bool_)

    orthonormalized_vectors=np.eye(num_elements)

    for cur_orth_id in range(num_elements):
        if not to_include[cur_orth_id]:
            continue
        # Normalize the vector.
        cur_norm=np.sqrt(sqnorm_residue[cur_orth_id])
        for i in prange(cur_orth_id+1):
            orthonormalized_vectors[cur_orth_id, i]/=cur_norm
        # Subtract projections of the normalized vector from all currently not orthonormalized vectors.
        # Also check that their residue is above the corresponding threshold.
        for i in prange(cur_orth_id+1, num_elements):
            if not to_include[i]:
                continue
            cur_product=0.0
            for j in range(cur_orth_id+1):
                if to_include[j]:
                    cur_product+=train_kernel[i, j]*orthonormalized_vectors[cur_orth_id, j]
            sqnorm_residue[i]-=cur_product**2
            if sqnorm_residue[i]<residue_tolerance[i]:
                to_include[i]=False
            else:
                for j in range(cur_orth_id+1):
                    orthonormalized_vectors[i, j]-=cur_product*orthonormalized_vectors[cur_orth_id, j]
        cur_orth_id+=1
    return all_indices_except(to_include)

class KernelUnstable(Exception):
    pass

def linear_dependent_entries(train_kernel, residue_tol_coeff, use_Fortran=True, lambda_val=0.0,
                                    return_orthonormalized=False, ascending_residue_order=True):
    if use_Fortran:
        num_elements=train_kernel.shape[0]
        output_indices=np.zeros(num_elements, dtype=np.int32)
        orthonormalized_vectors=np.zeros((num_elements, num_elements))
        flinear_dependent_entries(train_kernel, orthonormalized_vectors.T, num_elements,
                    residue_tol_coeff, lambda_val, ascending_residue_order, output_indices)
        if output_indices[0]==-2:
            raise KernelUnstable
        final_output=[]
        for i in range(num_elements):
            if output_indices[i]==-1:
                final_output=output_indices[:i]
                break
        if return_orthonormalized:
            return final_output, orthonormalized_vectors
        else:
            return final_output
    else:
        return numba_linear_dependent_entries(train_kernel, residue_tol_coeff)


def solve_Gram_Schmidt(sym_mat, vec, residue_tol_coeff, lambda_val=0.0, ascending_residue_order=False):
    ignored_indices, orthonormalized_vectors=linear_dependent_entries(sym_mat, residue_tol_coeff, use_Fortran=True,
                    lambda_val=lambda_val, return_orthonormalized=True, ascending_residue_order=ascending_residue_order)
    return ignored_indices, np.matmul(orthonormalized_vectors.T, np.matmul(orthonormalized_vectors, vec))


# For distance-based cutting off redundant entries from the kernel matrix.
@njit(fastmath=True, parallel=True)
def kernel2sqdist(train_kernel):
    num_train=train_kernel.shape[0]
    sqdist_mat=np.zeros((num_train, num_train))
    for i in prange(num_train):
        for j in range(num_train):
            sqdist_mat[i,j]=train_kernel[i,i]+train_kernel[j,j]-2*train_kernel[i,j]
    return sqdist_mat

@njit(fastmath=True)
def min_id_sqdist(sqdist_row, to_include, entry_id):
    cur_min_sqdist=0.0
    cur_min_sqdist_id=0
    minimal_sqdist_init=False
    num_train=sqdist_row.shape[0]

    for j in range(num_train):
        if entry_id != j:
            cur_sqdist=sqdist_row[j]
            if (((cur_sqdist<cur_min_sqdist) or (not minimal_sqdist_init)) and to_include[j]):
                minimal_sqdist_init=True
                cur_min_sqdist=cur_sqdist
                cur_min_sqdist_id=j
    return cur_min_sqdist_id, cur_min_sqdist


@njit(fastmath=True, parallel=True)
def numba_rep_sqdist_mat(rep_arr):
    num_vecs=rep_arr.shape[0]
    sqdist_mat=np.zeros((num_vecs, num_vecs))
    for i in prange(num_vecs):
        for j in range(i):
            sqdist_mat[i, j]=np.sum(np.square(rep_arr[i]-rep_arr[j]))
            sqdist_mat[j, i]=sqdist_mat[i, j]
    return sqdist_mat

@njit(fastmath=True, parallel=True)
def numba_sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries):
    num_train=sqdist_mat.shape[0]

    minimal_distance_ids=np.zeros(num_train, dtype=np.int32)
    minimal_distances=np.zeros(num_train)
    to_include=np.ones(num_train, dtype=bool_)

    for i in prange(num_train):
        minimal_distance_ids[i], minimal_distances[i]=min_id_sqdist(sqdist_mat[i], to_include, i)

    num_ignored=0

    while True:
        cur_min_id, cur_min_sqdist=min_id_sqdist(minimal_distances, to_include, -1)
        if (cur_min_sqdist > min_sqdist) and (min_sqdist > 0.0):
            break
        if np.random.random()>0.5:
            new_ignored=cur_min_id
        else:
            new_ignored=minimal_distance_ids[cur_min_id]

        to_include[new_ignored]=False
        num_ignored+=1
        if num_ignored==1:
            print("Smallest ignored distance:", cur_min_sqdist)
        if num_ignored==num_cut_closest_entries:
            print("Largest ignored distance:", cur_min_sqdist)
            break
        for i in prange(num_train):
            if to_include[i]:
                if (minimal_distance_ids[i]==new_ignored):
                    minimal_distance_ids[i], minimal_distances[i]=min_id_sqdist(sqdist_mat[i], to_include, i)

    return all_indices_except(to_include)



@njit(fastmath=True, parallel=True)
def numba_kernel_exclude_nearest(train_kernel, min_sqdist, num_cut_closest_entries):
    sqdist_mat=kernel2sqdist(train_kernel)
    return numba_sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries)


