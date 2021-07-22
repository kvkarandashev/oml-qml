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
import math, itertools, copy
from numba import njit, prange


from .foml_kernels import fgmo_kernel, flinear_base_kernel_mat, fgmo_sq_dist, fgmo_sep_ibo_kernel, fibo_fr_kernel, fgmo_sep_ibo_sym_kernel, fgmo_sep_ibo_sqdist_sums_nums





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
    sum_vals, sum2_vals, norm_prefac=sum12_from_orbs(orb_rep_array)
    return sum12_to_RMSE(sum_vals, sum2_vals, norm_prefac, var_cutoff_val=var_cutoff_val)

def sum12_from_orbs(orb_rep_array):
    vec_length=len(orb_rep_array[0].ibo_atom_reps[0].scalar_reps)
    sum_vals=jnp.zeros(vec_length)
    sum2_vals=jnp.zeros(vec_length)
    norm_prefac=0.0
    for orb_rep in orb_rep_array:
        for atom_rep in orb_rep.ibo_atom_reps:
            weight_factor=orb_rep.rho*atom_rep.rho
            sum_vals+=atom_rep.scalar_reps*weight_factor
            sum2_vals+=atom_rep.scalar_reps**2*weight_factor
            norm_prefac+=weight_factor
    return sum_vals, sum2_vals, norm_prefac

def sum12_to_RMSE(sum_vals, sum2_vals, norm_prefac, var_cutoff_val=0.0):
    av_vals=sum_vals/norm_prefac
    av2_vals=sum2_vals/norm_prefac
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
    def __init__(self, width_params=None, final_sigma=1.0, normalize_lb_kernel=False, parallel=False, use_Fortran=True, use_Gaussian_kernel=False,
                    pair_reps=True, density_neglect=1e-9):
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
        self.density_neglect=np.double(density_neglect)
    def update_width(self, width_params):
        self.width_params=jnp.array(width_params)
        self.inv_sq_width_params=self.width_params**(-2)

#   TO-DO rename scalar reps!!!

class GMO_kernel_input:
    def __init__(self, oml_compound_array=None, **other_kwargs):
        if oml_compound_array is None:
            self.num_mols=None
            self.max_tot_num_ibo_atom_reps=None
            self.max_num_scala_reps=None
            self.rhos=None
            self.ibo_atom_sreps=None
        else:
            self.num_mols=len(oml_compound_array)
            self.max_tot_num_ibo_atom_reps=0
            for oml_comp in oml_compound_array:
                self.max_tot_num_ibo_atom_reps=max(self.max_tot_num_ibo_atom_reps, count_ibo_atom_reps(oml_comp, **other_kwargs))
            self.max_num_scalar_reps=len(list(iterated_orb_reps(oml_compound_array[0], **other_kwargs))[0].ibo_atom_reps[0].scalar_reps)
            self.rhos=np.zeros((self.num_mols, self.max_tot_num_ibo_atom_reps))
            self.ibo_atom_sreps=np.zeros((self.num_mols, self.max_tot_num_ibo_atom_reps, self.max_num_scalar_reps))
            for ind_comp, oml_comp in enumerate(oml_compound_array):
                cur_rho_sreps = sorted_rhos_ibo_atom_sreps(oml_comp, **other_kwargs)
                aibo_counter=0
                for rho_srep in cur_rho_sreps:
                    self.rhos[ind_comp, aibo_counter]=rho_srep[0]
                    self.ibo_atom_sreps[ind_comp, aibo_counter, :]=rho_srep[1][:]
                    aibo_counter+=1
            self.ibo_atom_sreps=jnp.array(self.ibo_atom_sreps)
            self.rhos=jnp.array(self.rhos)

def orb_rep_rho_list(oml_comp, pair_reps=False):
    output=[]
    for ibo in iterated_orb_reps(oml_comp, pair_reps=pair_reps):
        output.append([ibo.rho, ibo])
    if pair_reps:
        for i in range(len(oml_comp.comps[0].orb_reps)):
            output[i][0]*=-1
    return output

def sorted_rhos_ibo_atom_sreps(oml_comp, pair_reps=False, **other_kwargs):
    output=[]
    for ibo in iterated_orb_reps(oml_comp, pair_reps=pair_reps, **other_kwargs):
        for ibo_atom_rep in ibo.ibo_atom_reps:
            output.append([ibo.rho*ibo_atom_rep.rho, ibo_atom_rep.scalar_reps])
    if pair_reps:
        for i in range(count_ibo_atom_reps(oml_comp.comps[0])):
            output[i][0]*=-1
    output.sort(key=lambda x: abs(x[0]), reverse=True)
    return output

def count_ibo_atom_reps(oml_comp, **other_kwargs):
    output=0
    for ibo in iterated_orb_reps(oml_comp, **other_kwargs):
        for ibo_atom_rep in ibo.ibo_atom_reps:
            output+=1
    return output

def iterated_orb_reps(oml_comp, pair_reps=False, single_ibo_list=False):
    if pair_reps:
        return itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps)
    else:
        if single_ibo_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps

def generate_GMO_kernel(A, B, kernel_params, sym_kernel_mat=False):
    if isinstance(kernel_params.pair_reps, list):
        pair_reps_A=kernel_params.pair_reps[0]
        pair_reps_B=kernel_params.pair_reps[1]
    else:
        pair_reps_A=kernel_params.pair_reps
        pair_reps_B=kernel_params.pair_reps
    Ac=GMO_kernel_input(oml_compound_array=A, pair_reps=pair_reps_A)
    Bc=GMO_kernel_input(oml_compound_array=B, pair_reps=pair_reps_B)
    if kernel_params.use_Fortran:
        kernel_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
        if kernel_params.use_Gaussian_kernel:
            fgmo_kernel(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, kernel_params.final_sigma, kernel_params.density_neglect,
                    kernel_params.normalize_lb_kernel, sym_kernel_mat, kernel_mat)
        else:
            flinear_base_kernel_mat(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, kernel_params.density_neglect, sym_kernel_mat, kernel_mat)
    else:
        #TO-DO use density_neglect here too?
        jconfig.update("jax_enable_x64", True)
        if kernel_params.parallel:
            return jnp.array(embarassingly_parallel(GMO_kernel_row, zip(Ac.rhos, Ac.ibo_atom_sreps), Bc, kernel_params))
        else:
            return jit(jax_gen_GMO_kernel, static_argnums=(6,7))(Ac.rhos, Ac.ibo_atom_sreps, Bc.rhos,
                        Bc.ibo_atom_sreps, kernel_params.inv_sq_width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel,
                        kernel_params.use_Gaussian_kernel)
    return kernel_mat

class GMO_sep_IBO_kern_input:
    def __init__(self, oml_compound_array=None, pair_reps=False):
        if pair_reps:
            self.max_num_scalar_reps=len(oml_compound_array[0].comps[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)
        else:
            self.max_num_scalar_reps=len(oml_compound_array[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)

        self.num_mols=len(oml_compound_array)
        self.ibo_nums=np.zeros((self.num_mols,), dtype=int)
        self.max_num_ibos=0
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            self.ibo_nums[comp_id]=len(rho_orb_list)
        self.max_num_ibos=max(self.ibo_nums)
        self.ibo_atom_nums=np.zeros((self.num_mols, self.max_num_ibos), dtype=int)
        self.ibo_rhos=np.zeros((self.num_mols, self.max_num_ibos))
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            for orb_id, [orb_rho, orb_rep] in enumerate(rho_orb_list):
                self.ibo_atom_nums[comp_id, orb_id]=len(orb_rep.ibo_atom_reps)
                self.ibo_rhos[comp_id, orb_id]=orb_rho

        self.max_num_ibo_atom_reps=np.amax(self.ibo_atom_nums)

        self.ibo_arep_rhos=np.zeros((self.num_mols, self.max_num_ibos, self.max_num_ibo_atom_reps))
        self.ibo_atom_sreps=np.zeros((self.num_mols, self.max_num_ibos, self.max_num_ibo_atom_reps, self.max_num_scalar_reps))
        for ind_comp, oml_comp in enumerate(oml_compound_array):
            for ind_ibo, [ibo_rho, ibo_rep] in enumerate(orb_rep_rho_list(oml_comp, pair_reps=pair_reps)):
                for ind_ibo_arep, ibo_arep in enumerate(ibo_rep.ibo_atom_reps):
                    self.ibo_arep_rhos[ind_comp, ind_ibo, ind_ibo_arep]=ibo_arep.rho
                    self.ibo_atom_sreps[ind_comp, ind_ibo, ind_ibo_arep, :]=ibo_arep.scalar_reps[:]

def GMO_sep_IBO_kernel(A, B, kernel_params):
    if isinstance(kernel_params.pair_reps, list):
        pair_reps_A=kernel_params.pair_reps[0]
        pair_reps_B=kernel_params.pair_reps[1]
    else:
        pair_reps_A=kernel_params.pair_reps
        pair_reps_B=kernel_params.pair_reps
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, pair_reps=pair_reps_A)
    Bc=GMO_sep_IBO_kern_input(oml_compound_array=B, pair_reps=pair_reps_B)
    kernel_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
    fgmo_sep_ibo_kernel(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.ibo_arep_rhos.T, Ac.ibo_rhos.T,
                    Ac.ibo_atom_nums.T, Ac.ibo_nums,
                    Ac.max_num_ibo_atom_reps, Ac.max_num_ibos, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.ibo_arep_rhos.T, Bc.ibo_rhos.T,
                    Bc.ibo_atom_nums.T, Bc.ibo_nums,
                    Bc.max_num_ibo_atom_reps, Bc.max_num_ibos, Bc.num_mols,
                    kernel_params.width_params, kernel_params.final_sigma,
                    kernel_mat)
    return kernel_mat

def GMO_sep_IBO_sym_kernel(A, kernel_params):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, pair_reps=kernel_params.pair_reps)
    kernel_mat = np.empty((Ac.num_mols, Ac.num_mols), order='F')
    fgmo_sep_ibo_sym_kernel(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.ibo_arep_rhos.T, Ac.ibo_rhos.T,
                    Ac.ibo_atom_nums.T, Ac.ibo_nums,
                    Ac.max_num_ibo_atom_reps, Ac.max_num_ibos, Ac.num_mols,
                    kernel_params.width_params, kernel_params.final_sigma,
                    kernel_mat)
    return kernel_mat

#   Create matrices containing sum over square distances between IBOs and the number of such pairs.
#   Mainly introduced for convenient sanity check of hyperparameter optimization results.
def GMO_sep_IBO_sqdist_sums_nums(A, kernel_params):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, pair_reps=kernel_params.pair_reps)
    sqdist_sums = np.empty((Ac.num_mols, Ac.num_mols), order='F')
    sqdist_nums = np.empty((Ac.num_mols, Ac.num_mols), dtype=int, order='F')
    for iA1, ibo_num1 in enumerate(Ac.ibo_nums):
        for iA2, ibo_num2 in enumerate(Ac.ibo_nums):
            sqdist_nums[iA1, iA2]=ibo_num1*ibo_num2
    fgmo_sep_ibo_sqdist_sums_nums(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.ibo_arep_rhos.T,
                    Ac.ibo_atom_nums.T, Ac.ibo_nums,
                    Ac.max_num_ibo_atom_reps, Ac.max_num_ibos, Ac.num_mols,
                    kernel_params.width_params, sqdist_sums)
    return sqdist_sums, sqdist_nums

def GMO_sqdist_mat(A, B, kernel_params, sym_kernel_mat=False):
    Ac=GMO_kernel_input(A, kernel_params.pair_reps)
    Bc=GMO_kernel_input(B, kernel_params.pair_reps)
    sqdist_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
    fgmo_sq_dist(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, kernel_params.density_neglect,
                    kernel_params.normalize_lb_kernel, sym_kernel_mat, sqdist_mat)
    return sqdist_mat

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

############
#   For IBO-FR procedures.
############

class IBOFR_kernel_params:
    def __init__(self, vec_rep_mult=None, pair_reps=False, density_neglect=1e-7):
        self.vec_rep_mult=vec_rep_mult
        self.pair_reps=pair_reps
        self.density_neglect=density_neglect

def rho_iterator(oml_comp):
    return np.repeat(oml_comp.ibo_occ, len(oml_comp.orb_reps))

def rho_orb_iterator(oml_comp, pair_reps):
    if pair_reps:
        return zip(itertools.chain(rho_iterator(oml_comp.comps[1]), -rho_iterator(oml_comp.comps[0])),
                    itertools.chain(oml_comp.comps[1].orb_reps, oml_comp.comps[0].orb_reps))
    else:
        return zip(rho_iterator(oml_comp), oml_comp.orb_reps)

# Introduced for convenience.
def ibofr_smoothed_mult(val, rep_params):
    output=np.repeat(val, 2*rep_params.num_prop_times)
    for i in range(rep_params.num_prop_times):
        multiplier=math.cos(math.pi*(i+1)/2/(rep_params.num_prop_times+1))**2
        output[2*i:2*(i+1)]*=multiplier
    return output

class IBOFR_kernel_input:
    def __init__(self, compound_list, vec_rep_mult, pair_reps=False):
        self.num_mols=len(compound_list)
        self.max_num_ibos=0
        for compound in compound_list:
            self.max_num_ibos=max(self.max_num_ibos, len(compound.orb_reps))
        self.vec_length=len(compound_list[0].orb_reps[0])
        self.ibo_scaled_vecs=np.zeros((self.vec_length, self.max_num_ibos, self.num_mols), order='F')
        self.rhos=np.zeros((self.max_num_ibos, self.num_mols), order='F')
        for comp_id, compound in enumerate(compound_list):
            for ibo_id, (rho_val, ibo_rep) in enumerate(rho_orb_iterator(compound, pair_reps)):
                self.ibo_scaled_vecs[:, ibo_id, comp_id]=ibo_rep[:]
                self.rhos[ibo_id, comp_id]=rho_val
        for vec_comp_id, cur_mult in enumerate(vec_rep_mult):
            self.ibo_scaled_vecs[vec_comp_id]*=cur_mult

def gen_ibofr_kernel(A, B, kernel_params, sym_kernel_mat=False):
    Ac=IBOFR_kernel_input(A, kernel_params.vec_rep_mult, pair_reps=kernel_params.pair_reps)
    Bc=IBOFR_kernel_input(B, kernel_params.vec_rep_mult, pair_reps=kernel_params.pair_reps)
    kernel_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
    fibo_fr_kernel(Ac.vec_length,
                    Ac.ibo_scaled_vecs, Ac.rhos, Ac.max_num_ibos, Ac.num_mols,
                    Bc.ibo_scaled_vecs, Bc.rhos, Bc.max_num_ibos, Bc.num_mols,
                    kernel_params.density_neglect, sym_kernel_mat, kernel_mat)
    return kernel_mat

def ibofr_rep_avs(compound_list, pair_reps=False):
    vec_length=len(compound_list[0].orb_reps[0])
    av_vals=np.zeros(vec_length)
    av2_vals=np.zeros(vec_length)
    norm_prefac=0.0
    for oml_comp in compound_list:
        for ibo_id, (rho_val, ibo_rep) in enumerate(rho_orb_iterator(oml_comp, pair_reps)):
            norm_prefac+=abs(rho_val)
            av_vals+=ibo_rep
            av2_vals+=ibo_rep**2
    av_vals/=norm_prefac
    av2_vals/=norm_prefac
    return av_vals, av2_vals


def ibofr_rep_stddevs(compound_list, pair_reps=False):
    av_vals, av2_vals=ibofr_rep_avs(compound_list, pair_reps=pair_reps)
    return np.array([sqrt_sign_checked(av2_val-av_val**2) for av_val, av2_val in zip(av_vals, av2_vals)])


#########
# END OF IBO-FR.
#########

### For linear kernel with separable IBOs.

# Orbital-orbital covariance.
@njit(fastmath=True)
def orb_areps_sqdiffs(orb_areps_A, arep_A_id, orb_areps_B, arep_B_id):
    return (orb_areps_A[arep_A_id, :]-orb_areps_B[arep_B_id, :])**2

@njit(fastmath=True)
def orb_orb_cov(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, max_num_ibo_atom_reps_A, max_num_ibo_atom_reps_B, inv_sq_width_params, density_neglect, with_ders=False):
    if with_ders:
        output=np.zeros((inv_sq_width_params.shape[0]+1,))
    else:
        output=np.zeros((1,))
    for arep_A_id in range(max_num_ibo_atom_reps_A):
        rho_A=arep_rhos_A[arep_A_id]
        if rho_A<density_neglect:
            break
        for arep_B_id in range(max_num_ibo_atom_reps_B):
            rho_B=arep_rhos_B[arep_B_id]
            if rho_B<density_neglect:
                break
            sqdiffs=orb_areps_sqdiffs(orb_areps_A, arep_A_id, orb_areps_B, arep_B_id)
            orb_orb_cov_comp=rho_A*rho_B*np.exp(-np.sum(inv_sq_width_params*sqdiffs))
            output[0]+=orb_orb_cov_comp
            if with_ders:
                output[1:]-=orb_orb_cov_comp*sqdiffs
    return output

# Self-covariance (used for normalizing the orbitals).
@njit(fastmath=True)
def orb_sqrt_self_cov(orb_areps, arep_rhos, max_num_ibo_atom_reps, inv_sq_width_params, density_neglect):
    return np.sqrt(orb_orb_cov(orb_areps, orb_areps, arep_rhos, arep_rhos, max_num_ibo_atom_reps, max_num_ibo_atom_reps, inv_sq_width_params, density_neglect)[0])


# For renormalizing orbital arep rho coefficients.
@njit(fastmath=True, parallel=True)
def lin_sep_kern_renormalized_arep_rhos(inp_arr_orb_areps, inp_arr_arep_rhos, inp_arr_ibo_rhos, inv_sq_width_params, density_neglect):
    num_mols=inp_arr_orb_areps.shape[0]
    max_num_ibos=inp_arr_orb_areps.shape[1]
    max_num_ibo_atom_reps=inp_arr_orb_areps.shape[2]
    output=np.copy(inp_arr_arep_rhos)
    for mol_id in prange(num_mols):
        for ibo_id in range(max_num_ibos):
            if np.abs(inp_arr_ibo_rhos[mol_id, ibo_id])<density_neglect:
                break
            output[mol_id, ibo_id, :]/=orb_sqrt_self_cov(inp_arr_orb_areps[mol_id, ibo_id, :, :], inp_arr_arep_rhos[mol_id, ibo_id, :], max_num_ibo_atom_reps, inv_sq_width_params, density_neglect)
    return output

@njit(fastmath=True, parallel=True)
def numba_lin_sep_IBO_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                inv_sq_width_params, density_neglect, sym_kernel, with_ders):

    A_num_mols=A_orb_areps.shape[0]
    A_num_ibos=A_orb_areps.shape[1]
    A_num_ibo_atom_reps=A_orb_areps.shape[2]

    B_num_mols=B_orb_areps.shape[0]
    B_num_ibos=B_orb_areps.shape[1]
    B_num_ibo_atom_reps=B_orb_areps.shape[2]

    if with_ders:
        num_ders=inv_sq_width_params.shape[0]

        A_sp_log_ders=self_product_log_ders(A_orb_areps, A_arep_rhos, A_orb_rhos, inv_sq_width_params, density_neglect)
        B_sp_log_ders=self_product_log_ders(B_orb_areps, B_arep_rhos, B_orb_rhos, inv_sq_width_params, density_neglect)
    else:
        num_ders=0

    cur_ibo_contribution=np.zeros((num_ders+1,))

    Kernel=np.zeros((A_num_mols, B_num_mols, num_ders+1))

    for A_mol_id in prange(A_num_mols):
        if sym_kernel:
            upper_B_mol_range=A_mol_id+1
        else:
            upper_B_mol_range=B_num_mols
        for B_mol_id in range(upper_B_mol_range):
            for A_ibo_id in range(A_num_ibos):
                rho_A=A_orb_rhos[A_mol_id, A_ibo_id]
                if np.abs(rho_A)<density_neglect:
                    break
                for B_ibo_id in range(B_num_ibos):
                    rho_B=B_orb_rhos[B_mol_id, B_ibo_id]
                    if np.abs(rho_B)<density_neglect:
                        break
                    cur_ibo_contribution=orb_orb_cov(A_orb_areps[A_mol_id, A_ibo_id, :, :], B_orb_areps[B_mol_id, B_ibo_id, :, :], A_arep_rhos[A_mol_id, A_ibo_id, :], B_arep_rhos[B_mol_id, B_ibo_id, :], A_num_ibo_atom_reps,
                                                                B_num_ibo_atom_reps, inv_sq_width_params, density_neglect, with_ders=with_ders)
                    if with_ders:
                        cur_ibo_contribution[1:]-=cur_ibo_contribution[0]*(A_sp_log_ders[A_mol_id, A_ibo_id]+B_sp_log_ders[B_mol_id, B_ibo_id])/2
                    Kernel[A_mol_id, B_mol_id,:]+=rho_A*rho_B*cur_ibo_contribution
            if sym_kernel:
                if A_mol_id != B_mol_id:
                    Kernel[B_mol_id, A_mol_id, :]=Kernel[A_mol_id, B_mol_id, :]
    return Kernel

def lin_sep_IBO_kernel_conv(Ac, Bc, inv_sq_width_params, density_neglect, preserve_converted_arrays=True, with_ders=False):
    if preserve_converted_arrays:
        Ac_renormed=copy.deepcopy(Ac)
        Bc_renormed=copy.deepcopy(Bc)
    else:
        Ac_renormed=Ac
        Bc_renormed=Bc
    Ac_renormed.ibo_arep_rhos=lin_sep_kern_renormalized_arep_rhos(Ac_renormed.ibo_atom_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos, inv_sq_width_params, density_neglect)
    sym_kernel=(Bc is None)
    if sym_kernel:
        Bc_renormed=Ac_renormed
    else:
        Bc_renormed.ibo_arep_rhos=lin_sep_kern_renormalized_arep_rhos(Bc_renormed.ibo_atom_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos, inv_sq_width_params, density_neglect)
    final_kernel=numba_lin_sep_IBO_kernel(Ac_renormed.ibo_atom_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                    Bc_renormed.ibo_atom_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos,
                                    inv_sq_width_params, density_neglect, sym_kernel, with_ders)
    if with_ders:
        return final_kernel
    else:
        return final_kernel[:, :, 0]

def is_pair_reps(comp_arr):
    return (hasattr(comp_arr[0], 'comps'))

def lin_sep_IBO_kernel(A, B, inv_sq_width_params, density_neglect, with_ders=False):
    pair_reps_A=is_pair_reps(A)
    pair_reps_B=is_pair_reps(B)
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, pair_reps=pair_reps_A)
    Bc=GMO_sep_IBO_kern_input(oml_compound_array=B, pair_reps=pair_reps_B)
    return lin_sep_IBO_kernel_conv(Ac, Bc, inv_sq_width_params, density_neglect, preserve_converted_arrays=False, with_ders=with_ders)

def lin_sep_IBO_sym_kernel_conv(Ac, inv_sq_width_params, density_neglect, with_ders=False, preserve_converted_arrays=True):
    return lin_sep_IBO_kernel_conv(Ac, None, inv_sq_width_params, density_neglect, preserve_converted_arrays=preserve_converted_arrays, with_ders=with_ders)

def lin_sep_IBO_sym_kernel(A, inv_sq_width_params, density_neglect, with_ders=False):
    pair_reps_A=is_pair_reps(A)
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, pair_reps=pair_reps_A)
    return lin_sep_IBO_sym_kernel_conv(Ac, inv_sq_width_params, density_neglect, preserve_converted_arrays=False, with_ders=with_ders)

### Generate log derivatives.
@njit(fastmath=True)
def orb_self_cov_wders(orb_areps, arep_rhos, max_num_ibo_atom_reps, inv_sq_width_params, density_neglect):
    return orb_orb_cov(orb_areps, orb_areps, arep_rhos, arep_rhos, max_num_ibo_atom_reps, max_num_ibo_atom_reps, inv_sq_width_params, density_neglect, with_ders=True)

@njit(fastmath=True, parallel=True)
def self_product_log_ders(orb_areps, arep_rhos, ibo_rhos, inv_sq_width_params, density_neglect):
    num_mols=orb_areps.shape[0]
    num_ibos=orb_areps.shape[1]
    num_ibo_atom_reps=orb_areps.shape[2]

    num_ders=inv_sq_width_params.shape[0]

    log_ders=np.zeros((num_mols, num_ibos, num_ders))

    cur_prod_self_der=np.zeros((num_ders+1,))
    for mol_id in prange(num_mols):
        for ibo_id in range(num_ibos):
            if np.abs(ibo_rhos[mol_id, ibo_id])<density_neglect:
                break
            cur_prod_self_der=orb_self_cov_wders(orb_areps[mol_id, ibo_id, :, :], arep_rhos[mol_id, ibo_id, :], num_ibo_atom_reps, inv_sq_width_params, density_neglect)
            log_ders[mol_id, ibo_id, :]=cur_prod_self_der[1:]/cur_prod_self_der[0]
    return log_ders

