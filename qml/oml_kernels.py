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
from jax import jit, vmap
from .python_parallelization import embarassingly_parallel
import math

from .foml_kernels import fgmo_kernel





#   Randomly select some OML_compound objects and combine their orbital representations (type OML_ibo_rep) into a list.
def random_ibo_sample(oml_comp_array, num_sampled_mols=None):
    sample_mols=random_sample_length_checked(oml_comp_array, num_sampled_mols)
    sample_orbs=[]
    for mol in sample_mols:
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
def oml_ensemble_widths_estimate(orb_rep_array):
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

def sqrt_sign_checked(val):
    try:
        return math.sqrt(val)
    except:
        print("WARNING: oml_ensemble_widths_estimate returned a zero variance value.")
        return 0.0

#Related to the GMO kernel
class GMO_kernel_params:
    def __init__(self, width_params=None, final_sigma=1.0, normalize_lb_kernel=True, parallel=False, use_Fortran=True):
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
    def update_width(self, width_params):
        self.width_params=jnp.array(width_params)
        self.inv_sq_width_params=self.width_params**(-2)

#   TO-DO rename scalar reps!!!

class GMO_kernel_input:
    def __init__(self, oml_compound_array):
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


def generate_GMO_kernel(A, B, kernel_params):
    Ac=GMO_kernel_input(A)
    Bc=GMO_kernel_input(B)
    if kernel_params.use_Fortran:
        kernel_mat = np.empty((Ac.num_mols, Bc.num_mols), order='F')
        fgmo_kernel(Ac.max_num_scalar_reps,
                    Ac.ibo_atom_sreps.T, Ac.rhos.T, Ac.max_tot_num_ibo_atom_reps, Ac.num_mols,
                    Bc.ibo_atom_sreps.T, Bc.rhos.T, Bc.max_tot_num_ibo_atom_reps, Bc.num_mols,
                    kernel_params.width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel, kernel_mat)
    else:
        if kernel_params.parallel:
            return jnp.array(embarassingly_parallel(GMO_kernel_row, zip(Ac.rhos, Ac.ibo_atom_sreps), Bc, kernel_params))
        else:
            return jit(jax_gen_GMO_kernel, static_argnums=(6,))(Ac.rhos, Ac.ibo_atom_sreps, Bc.rhos,
                        Bc.ibo_atom_sreps, kernel_params.inv_sq_width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel)
    return kernel_mat

def GMO_kernel_row(A_tuple, Bc, kernel_params):
    return jit(jax_GMO_kernel_row, static_argnums=(5,))(A_tuple[0], A_tuple[1], Bc.rhos, Bc.sreps, kernel_params.inv_sq_width_params, kernel_params.final_sigma, kernel_params.normalize_lb_kernel)

def jax_gen_GMO_kernel(A_rhos, A_sreps, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel):
    return vmap(jax_GMO_kernel_row, in_axes=(0, 0, None, None, None, None, None))(A_rhos, A_sreps, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel)

#   TO-DO Would vmap accelerate this?
def jax_GMO_kernel_row(A_rho, A_srep, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel):
    return vmap(jax_GMO_kernel_element, in_axes=(None, None, 0, 0, None, None, None)) (A_rho, A_srep, B_rhos, B_sreps, inv_sq_width_params, final_sigma, normalize_lb_kernel)

def jax_GMO_kernel_element(A_rho, A_srep, B_rho, B_srep, inv_sq_width_params, final_sigma, normalize_lb_kernel):
    A_tuple=(A_rho, A_srep)
    B_tuple=(B_rho, B_srep)
    AB=jax_GMO_lb_kernel_element(A_tuple, B_tuple, inv_sq_width_params)
    AA=jax_GMO_lb_kernel_element(A_tuple, A_tuple, inv_sq_width_params)
    BB=jax_GMO_lb_kernel_element(B_tuple, B_tuple, inv_sq_width_params)
    if normalize_lb_kernel:
        sq_dist=2*(1.0-AB/jnp.sqrt(AA*BB))
    else:
        sq_dist/=AA+BB-2*AB
    return jnp.exp(-sq_dist/2/final_sigma**2)
        
def jax_GMO_lb_kernel_element(A_tuple, B_tuple, inv_sq_width_params):
    lb_kernel=0.0
    for A_rho, A_srep in zip(*A_tuple):
        for B_rho, B_srep in zip(*B_tuple):
            lb_kernel+=A_rho*B_rho*jnp.exp(-jnp.dot(inv_sq_width_params, (A_srep-B_srep)**2)/4)
    return lb_kernel

# For Fortran implementation of the GMO kernel.

def count_ibo_atom_reps(oml_comp):
    output=0
    for ibo in oml_comp.orb_reps:
        for ibo_atom_rep in ibo.ibo_atom_reps:
            output+=1
    return output

