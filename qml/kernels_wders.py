# MIT License
#
# Copyright (c) 2016 Anders Steen Christensen, Felix A. Faber, Lars A. Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

from .kernels import gaussian_kernel, laplacian_kernel
from .fkernels_wders import fgaussian_pos_sum_restr_kernel

def merge_save_indices(*arrays):
    output_array=[]
    output_slices=[]
    cur_lower_bound=0
    for array in arrays:
        cur_upper_bound=cur_lower_bound+len(array)
        output_slices.append((cur_lower_bound,cur_upper_bound))
        cur_lower_bound=cur_upper_bound
        output_array+=array
    return output_array, output_slices

def merged_representation_arrays(total_compound_array, indices):
    output=[]
    for index_tuple in indices:
        output.append(np.array([total_compound_array[comp_id].representation for comp_id in range(*index_tuple)]))
    if len(output)==1:
        return output[0]
    else:
        return output

def SLATM_kernel_input(*compound_arrays):
    from .representations import get_slatm_mbtypes
    combined_array, part_slices=merge_save_indices(*compound_arrays)
    nuclear_charge_list=[]
    for comp in combined_array:
        nuclear_charge_list.append(comp.nuclear_charges)
    mbtypes=get_slatm_mbtypes(nuclear_charge_list)
    for compound_id in range(len(combined_array)):
        combined_array[compound_id].generate_slatm(mbtypes)
    return merged_representation_arrays(combined_array, part_slices)

def CM_kernel_input(*compound_arrays, sorting="row-norm"):
    combined_array, part_slices=merge_save_indices(*compound_arrays)
    max_size=0
    for compound_obj in combined_array:
        max_size=max(max_size, len(compound_obj.atomtypes))
    for compound_id in range(len(combined_array)):
        combined_array[compound_id].generate_coulomb_matrix(size=max_size, sorting=sorting)
    return merged_representation_arrays(combined_array, part_slices)

def kernel_from_converted(A, B, sigma, use_Gauss=True, with_ders=False):
    if use_Gauss:
        kernel=gaussian_kernel(A, B, sigma)
    else:
        kernel=laplacian_kernel(A, B, sigma)
    if with_ders:
        output=np.empty((*kernel.shape, 2))
        output[:, :, 0]=kernel
        output[:, :, 1]=-np.log(kernel)/sigma
        if use_Gauss:
            output[:, :, 1]*=2
        return output
    else:
        return kernel

def CM_kernel(A, B, sigma, use_Gauss=True, with_ders=False):
    Ac, Bc=CM_kernel_input(A, B)
    return kernel_from_converted(Ac, Bc, sigma, use_Gauss=use_Gauss, with_ders=with_ders)

def SLATM_kernel(A, B, sigma, use_Gauss=True, with_ders=False):
    Ac, Bc=SLATM_kernel_input(A, B)
    return kernel_from_converted(Ac, Bc, sigma, use_Gauss=use_Gauss, with_ders=with_ders)


# Some auxiliary functions for more convenient scripting in hyperparameter_optimization module.

def gaussian_sym_kernel_conv_wders(A, sigma_arr, with_ders=False):
    return  kernel_from_converted(A, A, sigma_arr[0], use_Gauss=True, with_ders=with_ders)

def laplacian_sym_kernel_conv_wders(A, sigma_arr, with_ders=False):
    return  kernel_from_converted(A, A, sigma_arr[0], use_Gauss=False, with_ders=with_ders)

def gaussian_kernel_conv_wders(A, B, sigma_arr, with_ders=False):
    return  kernel_from_converted(A, B, sigma_arr[0], use_Gauss=True, with_ders=width_ders)

def laplacian_kernel_conv_wders(A, B, sigma_arr, with_ders=False):
    return  kernel_from_converted(A, B, sigma_arr[0], use_Gauss=False, with_ders=width_ders)


# Kernel for representation vectors constrained by being normalized and undefined for negative values.
# (e.g. mixed masses/mass fractions).
def gaussian_pos_sum_restr_kernel(A, B, sigmas, with_ders=False):
    nA=len(A)
    nB=len(B)
    dimf=len(sigmas)

    A_conv=np.array(A)
    B_conv=np.array(B)

    kern_el_dim=1
    if with_ders:
        kern_el_dim+=dimf

    kernel=np.zeros((nA, nB, kern_el_dim))

    fgaussian_pos_sum_restr_kernel(A_conv.T, B_conv.T, sigmas, nA, nB, dimf, kern_el_dim, kernel.T)

    if with_ders:
        return kernel
    else:
        return kernel[:, :, 0]

