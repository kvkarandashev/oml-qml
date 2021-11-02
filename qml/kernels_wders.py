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
from .fkernels_wders import fgaussian_pos_sum_restr_kernel, fgaussian_pos_restr_kernel, fgaussian_pos_restr_sym_kernel, fgaussian_pos_restr_input_init

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

def dummy_kernel_input(*rep_arr_lists):
    output=[]
    for rep_arr_list in rep_arr_lists:
        output.append(np.array(rep_arr_list))
    if len(output)==1:
        return output[0]
    else:
        return output

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
def gaussian_pos_sum_restr_sym_kernel_wders(A, sigmas, with_ders=False):
    return gaussian_pos_sum_restr_kernel_wders(A, A, sigmas, with_ders=with_ders)

def gaussian_pos_sum_restr_kernel_wders(A, B, sigmas, use_Gauss=None, with_ders=False):
    nA=A.shape[0]
    nB=B.shape[0]
    dimf=sigmas.shape[0]

    kern_el_dim=1
    if with_ders:
        kern_el_dim+=dimf

    kernel=np.zeros((nA, nB, kern_el_dim))

    fgaussian_pos_sum_restr_kernel(A.T, B.T, sigmas, nA, nB, dimf, kern_el_dim, kernel.T)

    if with_ders:
        return kernel
    else:
        return kernel[:, :, 0]


# Kernel for representation vectors constrained by being undefined for negative values.


class gaussian_pos_restr_kernel_comp_converter():
    def __init__(self, base_converter):
        self.base_converter=base_converter
    def __call__(self, comp_lists):
        merged_rep_arrs=self.base_converter(comp_lists)
        if isinstance(merged_rep_arrs, list):
            return [Pos_restr_kernel_input(mra) for mra in merged_rep_arrs]
        else:
            return Pos_restr_kernel_input(merged_rep_arrs)

class Pos_restr_kernel_input:
    def __init__(self, merged_rep_vecs, sigmas, with_ders=False):
        assert len(merged_rep_vecs.shape)==2
        self.nvecs=merged_rep_vecs.shape[0]
        self.dimf=merged_rep_vecs.shape[1]
        self.nsigmas=len(sigmas)
        self.sigmas=sigmas
        self.lin_kern_el_dim=1
        self.with_ders=with_ders
        if self.with_ders:
            self.lin_kern_el_dim+=self.nsigmas-1
        self.resc_rep_wsqrt=np.zeros((self.nvecs, self.dimf, 2))
        self.self_prods=np.zeros((self.nvecs, self.lin_kern_el_dim))
        fgaussian_pos_restr_input_init(merged_rep_vecs.T, self.sigmas, self.nsigmas, self.nvecs, self.dimf,
                            self.lin_kern_el_dim, self.with_ders, self.resc_rep_wsqrt.T, self.self_prods.T)

def gaussian_pos_restr_sym_kernel_conv_wders(Ac, sigmas, with_ders=False, use_Gauss=None):
    nsigmas=len(sigmas)
    assert ((Ac.dimf+1==nsigmas) or (nsigmas==2))
    assert (Ac.with_ders==with_ders)

    kern_el_dim=1
    if with_ders:
        kern_el_dim+=nsigmas

    kernel=np.zeros((Ac.nvecs, Ac.nvecs, kern_el_dim))

    fgaussian_pos_restr_sym_kernel(Ac.resc_rep_wsqrt.T, Ac.self_prods.T, Ac.sigmas,
                Ac.nvecs, Ac.dimf, nsigmas, kern_el_dim, kernel.T)

    if with_ders:
        return kernel
    else:
        return kernel[:, :, 0]
    

def gaussian_pos_restr_kernel_conv_wders(Ac, Bc, sigmas, with_ders=False, use_Gauss=None):
    nsigmas=len(sigmas)
    assert ((Ac.dimf+1==nsigmas) or (nsigmas==2))
    assert (Bc.dimf==Ac.dimf)
    assert (Ac.with_ders==with_ders)
    assert (Bc.with_ders==with_ders)

    kern_el_dim=1
    if with_ders:
        kern_el_dim+=nsigmas

    kernel=np.zeros((Ac.nvecs, Bc.nvecs, kern_el_dim))

    fgaussian_pos_restr_kernel(Ac.resc_rep_wsqrt.T, Bc.resc_rep_wsqrt.T, Ac.self_prods.T, Bc.self_prods.T, Ac.sigmas,
                Ac.nvecs, Bc.nvecs, Ac.dimf, nsigmas, kern_el_dim, kernel.T)

    if with_ders:
        return kernel
    else:
        return kernel[:, :, 0]


def gaussian_pos_restr_sym_kernel_wders(A, sigmas, with_ders=False, use_Gauss=None):
    Ac=Pos_restr_kernel_input(A, sigmas, with_ders=with_ders)
    return gaussian_pos_restr_sym_kernel_conv_wders(Ac, sigmas, with_ders=with_ders, use_Gauss=use_Gauss)

def gaussian_pos_restr_kernel_wders(A, B, sigmas, use_Gauss=None, with_ders=False):
    Ac=Pos_restr_kernel_input(A, sigmas, with_ders=with_ders)
    Bc=Pos_restr_kernel_input(B, sigmas, with_ders=with_ders)
    return gaussian_pos_restr_kernel_conv_wders(Ac, Bc, sigmas, with_ders=with_ders, use_Gauss=use_Gauss)





