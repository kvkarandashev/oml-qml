# MIT License
#
# Copyright (c) 2016-2017 Anders Steen Christensen, Felix Faber, Lars Andersen Bratholm
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
import jax.numpy as jnp
import jax.config as jconfig
from jax import jit, vmap
from .python_parallelization import embarassingly_parallel
import math

from .foml_kernels import fgmo_kernel, flinear_base_kernel_mat





#   Randomly select some OML_compound objects and combine their orbital representations (type OML_ibo_rep) into a list.
def random_ibo_sample(oml_comp_array, num_sampled_mols=None, pair_reps=True):
    sample_mols=random_sample_length_checked(oml_comp_array, num_sampled_mols)
    sample_orbs=[]
    for mol in sample_mols:
        if pair_reps:
            for slater in mol.comps:
                sample_orbs+=slater.orb_reps
        else:
            sample_orbs+=mol.orb_reps
    return sample_orbs

def random_sample_length_checked(list_in, num_rand_samp):
    import random
    if num_rand_samp is None:
        return list_in
    if (len(list_in)<=num_rand_samp) or (num_rand_samp==0):
        return list_in
    else:
        return random.sample(list_in, num_rand_samp)

#   Estimate the average square deviation of scalar representations of atomic contributions to IBOs
#   in the orb_rep_array. Used to estimate reasonable hyperparameter values. 
def oml_ensemble_widths_estimate(orb_rep_array, var_cutoff_val=0.0):
    vec_length=len(orb_rep_array[0].ibo_atom_reps[0].scalar_reps)
    av_vals=jnp.zeros(vec_length)
    av2_vals=jnp.zeros(vec_length)
    norm_prefac=0.0
    for orb_rep in orb_rep_array:
        for atom_rep in orb_rep.ibo_atom_reps:
            weight_factor=orb_rep.rho*atom_rep.rho
            av_vals+=atom_rep.scalar_reps*weight_factor
            av2_vals+=atom_rep.scalar_reps**2*weight_factor
            norm_prefac+=weight_factor
    av_vals/=norm_prefac
    av2_vals/=norm_prefac
    return jnp.array([sqrt_sign_checked(av2_val-av_val**2, var_cutoff_val=var_cutoff_val) for av_val, av2_val in zip(av_vals, av2_vals)])

def sqrt_sign_checked(val, var_cutoff_val=0.0):
    try:
        if (val<var_cutoff_val):
            print("WARNING: oml_ensemble_widths_estimate found a variation to be negligible.")
            return 1.0
        else:
            return math.sqrt(val)
    except ValueError:
        print("WARNING: oml_ensemble_widths_estimate returned a zero variance value.")
        return 1.0

#Related to the GMO kernel
class GMO_kernel_params:
    def __init__(self, width_params=None, final_sigma=1.0, normalize_lb_kernel=False, parallel=False, use_Fortran=True, use_Gaussian_kernel=False, pair_reps=True):
        self.width_params=width_params
        if self.width_params is not None:
            self.width_params=jnp.array(self.width_params)
            self.inv_sq_width_params=self.width_params**(-2)
        else:
            self.inv_sq_width_params=None
        self.final_sigma=final_sigma
        self.use_Fortran=use_Fortran
        self.normalize_lb_kernel=normalize_lb_kernel
        self.parallel=parallel
        self.use_Gaussian_kernel=use_Gaussian_kernel
        self.pair_reps=pair_reps
    def update_width(self, width_params):
        self.width_params=jnp.array(width_params)
        self.inv_sq_width_params=self.width_params**(-2)

#   TO-DO rename scalar reps!!!

class GMO_kernel_input:
    def __init__(self, oml_compound_array=None):
        if oml_compound_array is None:
            self.num_mols=None
            self.max_tot_num_ibo_atom_reps=None
            self.rhos=None
            self.ibo_atom_sreps=None
        else:
            self.num_mols=len(oml_compound_array)
            self.max_tot_num_ibo_atom_reps=0
            for oml_comp in oml_compound_array:
                self.max_tot_num_ibo_atom_reps=max(self.max_tot_num_ibo_atom_reps, count_ibo_atom_reps(oml_comp))
            self.max_num_scalar_reps=len(oml_compound_array[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)
            self.rhos=np.zeros((self.num_mols, self.max_tot_num_ibo_atom_reps))
            self.ibo_atom_sreps=np.zeros((self.num_mols, self.max_tot_num_ibo_atom_reps, self.max_num_scalar_reps))
            for ind_comp, oml_comp in enumerate(oml_compound_array):
                aibo_counter=0
                for ibo in oml_comp.orb_reps:
                    for ibo_atom_rep in ibo.ibo_atom_reps:
                        self.rhos[ind_comp, aibo_counter]=ibo.rho*ibo_atom_rep.rho
                        self.ibo_atom_sreps[ind_comp, aibo_counter, :]=ibo_atom_rep.scalar_reps
                        aibo_counter+=1
            self.ibo_atom_sreps=jnp.array(self.ibo_atom_sreps)
            self.rhos=jnp.array(self.rhos)

def count_ibo_atom_reps(oml_comp):
    output=0
    for ibo in oml_comp.orb_reps:
        for ibo_atom_rep in ibo.ibo_atom_reps:
            output+=1
    return output


def generate_GMO_kernel(A, B, kernel_params, sym_kernel_mat=False):
    if kernel_params.pair_reps:
        Ac=pair_GMO_kernel_input(A)
        Bc=pair_GMO_kernel_input(B)
    else:
        Ac=GMO_kernel_input(A)
        Bc=GMO_kernel_input(B)
    if kernel_params.use_Fortran:
        kernel_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
        if kernel_params.use_Gaussian_kernel:
            fgmo_kernel(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel,
                    sym_kernel_mat, kernel_mat)
        else:
            flinear_base_kernel_mat(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, sym_kernel_mat, kernel_mat)
    else:
        jconfig.update("jax_enable_x64", True)
        if kernel_params.parallel:
            return jnp.array(embarassingly_parallel(GMO_kernel_row, zip(Ac.rhos, Ac.ibo_atom_sreps), Bc, kernel_params))
        else:
            return jit(jax_gen_GMO_kernel, static_argnums=(6,7))(Ac.rhos, Ac.ibo_atom_sreps, Bc.rhos,
                        Bc.ibo_atom_sreps, kernel_params.inv_sq_width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel,
                        kernel_params.use_Gaussian_kernel)
    return kernel_mat

def GMO_kernel_row(A_tuple, Bc, kernel_params):
    return jit(jax_GMO_kernel_row, static_argnums=(6,7))(A_tuple[0], A_tuple[1], Bc.rhos, Bc.ibo_atom_sreps, kernel_params.inv_sq_width_params,
                kernel_params.final_sigma, kernel_params.normalize_lb_kernel, kernel_params.use_Gaussian_kernel)

def jax_gen_GMO_kernel(A_rhos, A_sreps, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel, use_Gaussian_kernel):
    return vmap(jax_GMO_kernel_row, in_axes=(0, 0, None, None, None, None, None, None))(A_rhos, A_sreps, B_rhos, B_sreps, inv_sq_width_params,
                                                                                    final_sigma, normalize_lb_kernel, use_Gaussian_kernel)

#   TO-DO Would vmap accelerate this?
def jax_GMO_kernel_row(A_rho, A_srep, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel, use_Gaussian_kernel):
    return vmap(jax_GMO_kernel_element, in_axes=(None, None, 0, 0, None, None, None, None)) (A_rho, A_srep, B_rhos, B_sreps,
                                                inv_sq_width_params, final_sigma, normalize_lb_kernel, use_Gaussian_kernel)

def jax_GMO_kernel_element(A_rho, A_srep, B_rho, B_srep, inv_sq_width_params, final_sigma, normalize_lb_kernel, use_Gaussian_kernel):
    AB=jax_GMO_lb_kernel_element(A_rho, A_srep, B_rho, B_srep, inv_sq_width_params)
    if use_Gaussian_kernel:
        AA=jax_GMO_lb_sq_norm(A_rho, A_srep, inv_sq_width_params)
        BB=jax_GMO_lb_sq_norm(B_rho, B_srep, inv_sq_width_params)
        if normalize_lb_kernel:
            sq_dist=2*(1.0-AB/jnp.sqrt(AA*BB))
        else:
            sq_dist=AA+BB-2*AB
        return jnp.exp(-sq_dist/2/final_sigma**2)
    else:
        return AB


def jax_GMO_lb_kernel_element(A_rho, A_srep, B_rho, B_srep, inv_sq_width_params):
    lb_kernel=0.0
    #TO-DO A good way to replace this with jax.lax.scan?
    Gauss_overlap_matrix=vmap(jax_GMO_Gauss_overlap_row, in_axes=(0, None, None)) (A_srep, B_srep, inv_sq_width_params)
    return jnp.dot(A_rho, jnp.matmul(Gauss_overlap_matrix, B_rho))

def jax_GMO_Gauss_overlap_row(A_aibo_srep, B_aibo_sreps, inv_sq_width_params):
    return vmap(jax_GMO_Gauss_overlap, in_axes=(None, 0, None))(A_aibo_srep, B_aibo_sreps, inv_sq_width_params)

def jax_GMO_Gauss_overlap(A_aibo_srep, B_aibo_srep, inv_sq_width_params):
    return jnp.exp(-jnp.dot(inv_sq_width_params, (A_aibo_srep-B_aibo_srep)**2)/4)


def jax_GMO_lb_sq_norm(rho, srep, inv_sq_width_params):
    return jax_GMO_lb_kernel_element(rho, srep, rho, srep, inv_sq_width_params)

    
# For pairs of Slater determinants.
def pair_GMO_kernel_input(pair_oml_compound_array):
    comp_arr1=[]
    comp_arr2=[]
    for pair_compound in pair_oml_compound_array:
        comp_arr1.append(pair_compound.comps[0])
        comp_arr2.append(pair_compound.comps[1])
    conv_arr1=GMO_kernel_input(comp_arr1)
    conv_arr2=GMO_kernel_input(comp_arr2)
    final_result=GMO_kernel_input()
    final_result.ibo_atom_sreps=merge_along_2nd_dim(conv_arr1.ibo_atom_sreps, conv_arr2.ibo_atom_sreps)
    final_result.rhos=merge_along_2nd_dim(conv_arr1.rhos, conv_arr2.rhos, first_negative=True)
    final_result.num_mols=len(pair_oml_compound_array)
    final_result.max_tot_num_ibo_atom_reps=conv_arr1.max_tot_num_ibo_atom_reps+conv_arr2.max_tot_num_ibo_atom_reps
    final_result.max_num_scalar_reps=conv_arr1.max_num_scalar_reps
    return final_result

def merge_along_2nd_dim(arr1, arr2, first_negative=False):
    output=[]
    for mol1, mol2 in zip(arr1, arr2):
        part1=np.array(mol1)
        part2=np.array(mol2)
        if first_negative:
            part1*=-1
        output.append(np.concatenate((part1, part2)))
    return jnp.array(output)

#   Estimate the average square deviation of scalar representations of atomic contributions to IBOs
#   in the orb_rep_array. Used to estimate reasonable hyperparameter values. 
def oml_pair_ensemble_widths_estimate(orb_rep_array):
    vec_length=len(orb_rep_array[0].ibo_atom_reps[0].scalar_reps)
    av_vals=jnp.zeros(vec_length)
    av2_vals=jnp.zeros(vec_length)
    norm_prefac=0.0
    for orb_rep in orb_rep_array:
        for atom_rep in orb_rep.ibo_atom_reps:
            weight_factor=orb_rep.rho*atom_rep.rho
            av_vals+=atom_rep.scalar_reps*weight_factor
            av2_vals+=atom_rep.scalar_reps**2*weight_factor
            norm_prefac+=weight_factor
    av_vals/=norm_prefac
    av2_vals/=norm_prefac
    return jnp.array([sqrt_sign_checked(av2_val-av_val**2) for av_val, av2_val in zip(av_vals, av2_vals)])


