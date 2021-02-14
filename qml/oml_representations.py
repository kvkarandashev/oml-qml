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
from jax import jit
import math

from .foml_representations import fgen_ibo_atom_scalar_rep, fgen_fock_ft_coup_mats
#   TO-DO get rid of next line
from .oml_kernels import generate_GMO_kernel
from qml.math import cho_invert


class OML_rep_params:
#   Parameters defining how IBO's are represented and biased.
#   tol_orb_cutoff      - consider AO's coefficient zero if it's smaller than tol_orb_cutoff.
#   en_bias_coeff       - coefficient for exponential biasing of energy.
#   ibo_atom_rho_comp   - compose IBO representation out of AO's centered on minimal ammount of atoms that would (approximately) account for at least
#                         ibo_atom_rho_comp of the electronic density.
#   l_max               - maximal value of angular momentum.
    def __init__(self, tol_orb_cutoff=1.0e-6, en_bias_coeff=None, en_degen_tol=0.01, ibo_rho_comp=None,
                    ibo_atom_rho_comp=None, max_angular_momentum=3, use_Fortran=True, mult_coup_mat=False,
                    mult_coup_mat_level=1, fock_based_coup_mat=False, num_fbcm_times=1, fbcm_delta_t=1.0,
                    fbcm_pseudo_orbs=False, ibo_spectral_representation=False, energy_rho_comp=1.0):
        self.tol_orb_cutoff=tol_orb_cutoff
        self.ibo_atom_rho_comp=ibo_atom_rho_comp
        self.max_angular_momentum=max_angular_momentum
        self.use_Fortran=use_Fortran
        self.fock_based_coup_mat=fock_based_coup_mat
        self.num_fbcm_times=num_fbcm_times
        self.fbcm_delta_t=fbcm_delta_t
        self.fbcm_pseudo_orbs=fbcm_pseudo_orbs
        self.ibo_spectral_representation=ibo_spectral_representation
        self.energy_rho_comp=energy_rho_comp
    def __str__(self):
        str_out="ibo_atom_rho_comp:"+str(self.ibo_atom_rho_comp)+",max_ang_mom:"+str(self.max_angular_momentum)
        if self.fock_based_coup_mat:
            str_out+=",fbcm_delta_t:"+str(self.fbcm_delta_t)+",num_fbcm_times:"+str(self.num_fbcm_times)
        return str_out

def gen_fock_based_coup_mats(rep_params, hf_orb_coeffs, hf_orb_energies):
    num_orbs=len(hf_orb_energies)
    inv_hf_orb_coeffs=np.linalg.inv(hf_orb_coeffs)
    if rep_params.use_Fortran:
        output=np.zeros((num_orbs, num_orbs, rep_params.num_fbcm_times*2), order='F')
        fgen_fock_ft_coup_mats(inv_hf_orb_coeffs, hf_orb_energies, rep_params.fbcm_delta_t, num_orbs, rep_params.num_fbcm_times, output)
        return tuple(np.transpose(output))
    else:
        output=()
        for freq_counter in range(rep_params.num_fbcm_times):
            prop_time=(freq_counter+1)*rep_params.fbcm_delta_t
            for trigon_func in [math.cos, math.sin]:
                new_mat=np.zeros((num_orbs, num_orbs))
                for i, en in enumerate(hf_orb_energies):
                    new_mat[i][i]=trigon_func(prop_time*en)
                new_mat=np.matmul(inv_hf_orb_coeffs.T, np.matmul(new_mat, inv_hf_orb_coeffs))
                output=(*output, new_mat)
        return output

#   Representation of contribution of a single atom to an IBO.

class OML_ibo_atom_rep:
    def __init__(self, atom_id, atom_list, coeffs, rep_params, atom_ao_ranges, angular_momenta, ovlp_mat, coup_mats):
        if rep_params.use_Fortran:
            self.scalar_reps=np.zeros(scalar_rep_length(rep_params, coup_mats))
            rho_arr=np.zeros(1)
            fgen_ibo_atom_scalar_rep(atom_id, atom_list, coeffs, np.transpose(atom_ao_ranges), angular_momenta, ovlp_mat, np.transpose(coup_mats),
                                        len(atom_list), len(coup_mats), rep_params.max_angular_momentum,
                                        len(coeffs), len(atom_ao_ranges), self.scalar_reps, rho_arr)
            self.rho=rho_arr[0]
        else:
            self.scalar_reps, self.rho=ang_mom_descr(atom_id, atom_ao_ranges, coeffs, angular_momenta, ovlp_mat, rep_params.max_angular_momentum)
            couplings=[]
            for coup_id, mat in enumerate(coup_mats):
                for same_atom in [True, False]:
                    for ang_mom1 in avail_ang_mom(rep_params):
                        for ang_mom2 in avail_ang_mom(rep_params):
                            if same_atom and (ang_mom1 > ang_mom2):
                                continue
                            if same_atom:
                                cur_coupling=ibo_atom_atom_coupling(atom_id, atom_id, ang_mom1, ang_mom2,
                                                            atom_ao_ranges, coeffs, mat, angular_momenta)
                            else:
                                cur_coupling=0.0
                                for other_atom_id in atom_list:
                                    if other_atom_id != atom_id:
                                        cur_coupling+=ibo_atom_atom_coupling(atom_id, other_atom_id, ang_mom1, ang_mom2,
                                                        atom_ao_ranges, coeffs, mat, angular_momenta)
                            couplings.append(cur_coupling)
            self.scalar_reps=np.append(self.scalar_reps, couplings)
        self.scalar_reps/=self.rho
    def __str__(self):
        return "OML_ibo_atom_rep,rho:"+str(self.rho)
    def __repr__(self):
        return str(self)

def ang_mom_descr(atom_id, atom_ao_ranges, coeffs, angular_momenta, ovlp_mat, max_angular_momentum):
    ang_mom_distr=np.zeros(max_angular_momentum)
    for ang_mom_counter in range(max_angular_momentum):
        ang_mom=ang_mom_counter+1
        ang_mom_distr[ang_mom_counter]=ibo_atom_atom_coupling(atom_id, atom_id, ang_mom, ang_mom, atom_ao_ranges, coeffs, ovlp_mat, angular_momenta)
    rho=ibo_atom_atom_coupling(atom_id, atom_id, 0, 0, atom_ao_ranges, coeffs, ovlp_mat, angular_momenta)+sum(ang_mom_distr)
    return ang_mom_distr, rho

def avail_ang_mom(rep_params):
    return range(num_ang_mom(rep_params))

def scalar_rep_length(rep_params, coup_mats):
    nam=num_ang_mom(rep_params)
    return nam-1+len(coup_mats)*(nam**2+(nam*(nam+1))//2)

def num_ang_mom(rep_params):
    return rep_params.max_angular_momentum+1

#   TO-DO How to use JIT here? Applying it to parts of arrays does not look straightforward.
def ibo_atom_atom_coupling(atom_id1, atom_id2, ang_mom1, ang_mom2, atom_ao_ranges, coeffs, matrix, angular_momenta):
    coupling=0.0
    for aid1 in range(atom_ao_ranges[atom_id1, 0], atom_ao_ranges[atom_id1, 1]):
        if angular_momenta[aid1]==ang_mom1:
            for aid2 in range(atom_ao_ranges[atom_id2, 0], atom_ao_ranges[atom_id2, 1]):
                if angular_momenta[aid2]==ang_mom2:
                    coupling+=coeffs[aid1]*coeffs[aid2]*matrix[aid1, aid2]
    return coupling


# Internal format for processing AOs created by pySCF.
orb_ang_mom={"s" : 0, "p" : 1, "d" : 2, "f" : 3, "g" : 4, "h" : 5, "i" : 6}

class AO:
    def __init__(self, ao_label):
        info=ao_label.split()
        self.ao_type=info[2]
        self.atom_id=int(info[0])
        for char in self.ao_type:
            try:
                int(char)
            except ValueError:
                self.angular=orb_ang_mom[char]
                break
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return ":atom_id:"+str(self.atom_id)+":ao_type:"+self.ao_type+":ang_momentum:"+str(self.angular)

def generate_ao_arr(mol):
    return [AO(ao_label) for ao_label in mol.ao_labels()]

def generate_atom_ao_ranges(mol):
    ao_sliced_with_shells=mol.aoslice_by_atom()
    output=[]
    for atom_data in ao_sliced_with_shells:
        output.append(atom_data[2:4])
    return np.array(output)

#   Generate an array of IBO representations.
def generate_ibo_rep_array(ibo_mat, rep_params, aos, atom_ao_ranges, ovlp_mat, *coupling_mats):
    # It's important that ovlp_mat appears first in this array.
    atom_ids=[ao.atom_id for ao in aos]
    angular_momenta=[ao.angular for ao in aos]
    ibo_tmat=ibo_mat.T
    return [OML_ibo_rep(ibo_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats) for ibo_coeffs in ibo_tmat]
    
def generate_ibo_spectral_rep_array(ibo_mat, rep_params, overlap_func, mo_coeff, mo_occ, mo_energy, reference_energy):
    return weighted_array([OML_ibo_spectral_rep(ibo_coeffs, rep_params, overlap_func, mo_coeff, mo_occ, mo_energy, reference_energy) for ibo_coeffs in ibo_mat.T])
    
class OML_ibo_spectral_line:
    def __init__(self, ibo_coeffs, reference_energy, mo_en, mo_c, overlap_func):
        cur_coeff=overlap_func(ibo_coeffs, mo_c)
        self.pos_sign=(cur_coeff >= 0.0)
        self.rho=cur_coeff**2
        self.relative_energy=mo_en-reference_energy
    def covariance_measure(self, other_spectral_line, kernel_params):
        return -((self.relative_energy-other_spectral_line.relative_energy)/kernel_params.width_params)**2
    def covariance_finalized(self, measure_value, kernel_params):
        return math.exp(measure_value/2)

#   Representation of an IBO from its energy spectrum.
class OML_ibo_spectral_rep:
    def __init__(self, ibo_coeffs, rep_params, overlap_func, mo_coeff, mo_occ, mo_energy, reference_energy):
        from .oml_representations import weighted_array
        self.rho=0.0
        self.full_coeffs=ibo_coeffs
        self.energy_spectrum=weighted_array([])
        for mo_en, mo_c, occ in zip(mo_energy, mo_coeff.T, mo_occ):
            if occ>1.0:
                self.energy_spectrum.append(OML_ibo_spectral_line(self.full_coeffs, reference_energy, mo_en, mo_c, overlap_func))
        self.energy_spectrum.normalize_sort_rhos()
        # Try to decrease the number of atomic representations, leaving only the most relevant ones.
        self.energy_spectrum.cutoff_minor_weights(remaining_rho=rep_params.energy_rho_comp)

#   Representation of an IBO from atomic contributions.
class OML_ibo_rep:
    def __init__(self, ibo_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coup_mats):
        from .oml_representations import weighted_array
        self.rho=0.0
        self.full_coeffs=ibo_coeffs
        atom_list=[]
        prev_atom=-1
        for atom_id, ao_coeff in zip(atom_ids, self.full_coeffs):
            if abs(ao_coeff)>rep_params.tol_orb_cutoff:
                if prev_atom != atom_id:
                    atom_list.append(atom_id)
                    prev_atom=atom_id
        # Each of the resulting groups of AOs is represented with OML_ibo_atom_rep object.
        self.ibo_atom_reps=weighted_array([OML_ibo_atom_rep(atom_id, atom_list, self.full_coeffs, rep_params, atom_ao_ranges, angular_momenta, ovlp_mat, coup_mats)
                                for atom_id in atom_list])
        self.ibo_atom_reps.normalize_sort_rhos()
        # Try to decrease the number of atomic representations, leaving only the most relevant ones.
        self.ibo_atom_reps.cutoff_minor_weights(remaining_rho=rep_params.ibo_atom_rho_comp)

class weighted_array(list):
    def normalize_rhos(self, normalization_constant=None):
        if normalization_constant is None:
            normalization_constant=sum(el.rho for el in self)
        for i in range(len(self)):
            self[i].rho/=normalization_constant
    def sort_rhos(self):
        self.sort(key=lambda x: x.rho, reverse=True)
    def normalize_sort_rhos(self):
        self.normalize_rhos()
        self.sort_rhos()
    def cutoff_minor_weights(self, remaining_rho=None):
        if remaining_rho is not None:
            rho_new_sum=0.0
            delete_from_end=0
            for counter, el in enumerate(self):
                rho_new_sum+=el.rho
                if rho_new_sum>remaining_rho:
                    delete_from_end=counter+1-len(self)
                    break
            if delete_from_end != 0:
                del(self[delete_from_end:])
                self.normalize_rhos(rho_new_sum)


