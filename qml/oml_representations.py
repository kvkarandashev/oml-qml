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

from .foml_representations import fgen_ibo_atom_scalar_rep, fgen_ft_coup_mats, fang_mom_descr


class OML_rep_params:
#   Parameters defining how IBO's are represented and biased.
#   tol_orb_cutoff      - consider AO's coefficient zero if it's smaller than tol_orb_cutoff.
#   ibo_atom_rho_comp   - compose IBO representation out of AO's centered on minimal ammount of atoms that would (approximately) account for at least
#                         ibo_atom_rho_comp of the electronic density.
#   l_max               - maximal value of angular momentum.
    def __init__(self, tol_orb_cutoff=1.0e-6,  ibo_atom_rho_comp=None, max_angular_momentum=3, use_Fortran=True,
                    fock_based_coup_mat=False, num_prop_times=1, prop_delta_t=1.0, fbcm_exclude_Fock=False,
                    fbcm_pseudo_orbs=False, ibo_fidelity_rep=False, norm_by_nelec=False):
        self.tol_orb_cutoff=tol_orb_cutoff
        self.ibo_atom_rho_comp=ibo_atom_rho_comp
        self.max_angular_momentum=max_angular_momentum
        self.use_Fortran=use_Fortran
        self.fock_based_coup_mat=fock_based_coup_mat
        self.num_prop_times=num_prop_times
        self.prop_delta_t=prop_delta_t
        self.fbcm_pseudo_orbs=fbcm_pseudo_orbs
        self.norm_by_nelec=norm_by_nelec
        self.fbcm_exclude_Fock=fbcm_exclude_Fock

        self.ibo_fidelity_rep=ibo_fidelity_rep
    def __str__(self):
        if self.ibo_fidelity_rep:
            str_out="IBOFR,num_prop_times:"+str(self.num_prop_times)+",prop_delta_dt"+str(self.prop_delta_t)
        else:
            str_out="ibo_atom_rho_comp:"+str(self.ibo_atom_rho_comp)+",max_ang_mom:"+str(self.max_angular_momentum)
            if self.fock_based_coup_mat:
                str_out+=",prop_delta_t:"+str(self.prop_delta_t)+",num_prop_times:"+str(self.num_prop_times)
        return str_out

def gen_fock_based_coup_mats(rep_params, hf_orb_coeffs, hf_orb_energies):
    num_orbs=len(hf_orb_energies)
    inv_hf_orb_coeffs=np.linalg.inv(hf_orb_coeffs)
    if rep_params.use_Fortran:
        output=np.zeros((num_orbs, num_orbs, rep_params.num_prop_times*2), order='F')
        fgen_ft_coup_mats(inv_hf_orb_coeffs, hf_orb_energies, rep_params.prop_delta_t, num_orbs, rep_params.num_prop_times, output)
        return tuple(np.transpose(output))
    else:
        output=()
        for timestep_counter in range(rep_params.num_prop_times):
            prop_time=(timestep_counter+1)*rep_params.prop_delta_t
            for trigon_func in [math.cos, math.sin]:
                new_mat=np.zeros((num_orbs, num_orbs))
                for i, en in enumerate(hf_orb_energies):
                    new_mat[i][i]=trigon_func(prop_time*en)
                new_mat=np.matmul(inv_hf_orb_coeffs.T, np.matmul(new_mat, inv_hf_orb_coeffs))
                output=(*output, new_mat)
        return output

#   Representation of contribution of a single atom to an IBO.

class OML_ibo_atom_rep:
    def __init__(self, atom_ao_range, coeffs, rep_params, angular_momenta, ovlp_mat):
        if rep_params.use_Fortran:
            self.atom_ao_range=atom_ao_range
            self.scalar_reps=np.zeros(rep_params.max_angular_momentum)
            rho_arr=np.zeros(1)
            fang_mom_descr(atom_ao_range, coeffs, angular_momenta, ovlp_mat, rep_params.max_angular_momentum, len(coeffs), self.scalar_reps, rho_arr)
            self.rho=rho_arr[0]
            self.pre_renorm_rho=self.rho
        else:
            self.scalar_reps, self.rho=ang_mom_descr(atom_id, atom_ao_ranges, coeffs, angular_momenta, ovlp_mat, rep_params.max_angular_momentum)
    def completed_scalar_reps(self, coeffs, rep_params, angular_momenta, coup_mats):
        if rep_params.use_Fortran:
            couplings=np.zeros(scalar_coup_length(rep_params, coup_mats))
            fgen_ibo_atom_scalar_rep(self.atom_ao_range, coeffs, angular_momenta, np.transpose(coup_mats),
                                        len(coup_mats), rep_params.max_angular_momentum, len(coeffs), couplings)
        else:
        #   TO-DO rewrite in JAX???
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
        self.scalar_reps/=self.pre_renorm_rho
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

def scalar_coup_length(rep_params, coup_mats):
    nam=num_ang_mom(rep_params)
    return len(coup_mats)*(nam**2+(nam*(nam+1))//2)

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
    def __init__(self, ao_label, atom_id=None):
        if atom_id is None:
            info=ao_label.split()
            self.ao_type=info[2]
            self.atom_id=int(info[0])
        else:
            self.atom_id=atom_id
            self.ao_type=ao_label
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
        self.ibo_atom_reps=weighted_array([OML_ibo_atom_rep(atom_ao_range, self.full_coeffs, rep_params, angular_momenta, ovlp_mat)
                                for atom_ao_range in atom_ao_ranges])
        self.ibo_atom_reps.normalize_sort_rhos()
        # Try to decrease the number of atomic representations, leaving only the most relevant ones.
        self.ibo_atom_reps.cutoff_minor_weights(remaining_rho=rep_params.ibo_atom_rho_comp)
        for ibo_arep_counter in range(len(self.ibo_atom_reps)):
            self.ibo_atom_reps[ibo_arep_counter].completed_scalar_reps(self.full_coeffs, rep_params, angular_momenta, coup_mats)

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
                self[-1].rho+=remaining_rho-rho_new_sum
                self.normalize_rhos()

# Related to IBO fidelity representation.
def generate_ibo_fidelity_rep(oml_compound, rep_params):
    nspins=len(oml_compound.ibo_mat)
    naos=len(oml_compound.aos)
    num_orbs=0
    for ispin in range(nspins):
        num_orbs+=len(oml_compound.ibo_mat[ispin].T)
    num_prop_types=1
    tot_mat_num=num_prop_types*rep_params.num_prop_times*2
    orb_reps=np.zeros((num_orbs, tot_mat_num))
    prop_mats=np.zeros((naos, naos, tot_mat_num), order='F')
    prp_wovlp=np.zeros((tot_mat_num, naos, naos), order='F')
    cur_orb_id=0
    for ispin in range(nspins):
        if nspins==1:
            vhf=oml_compound.j_mat[0]-0.5*oml_compound.k_mat[0]
        else:
            vhf=oml_compound.j_mat[0]+oml_compound.j_mat[1]-oml_compound.k_mat[ispin]
        pseudo_ens, pseudo_orbs=np.linalg.eigh(vhf)
        fgen_ft_coup_mats(pseudo_orbs.T, pseudo_ens, rep_params.prop_delta_t, naos, rep_params.num_prop_times, prop_mats)
        for mat_id, mat in enumerate(prop_mats.T):
            prp_wovlp[mat_id]=np.matmul(oml_compound.ovlp_mat, mat)
        for orb_id, orb_coeffs in enumerate(oml_compound.ibo_mat[ispin].T):
            for prop_time_id, prop_mat in enumerate(prop_mats.T):
                orb_reps[cur_orb_id, prop_time_id]=np.dot(orb_coeffs,np.matmul(prop_mat,orb_coeffs))
            cur_orb_id+=1
    return orb_reps
