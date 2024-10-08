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
import math, copy

try:
    from .foml_representations import fgen_ibo_atom_scalar_rep, fgen_ft_coup_mats, fang_mom_descr, fgen_ibo_global_couplings
except:
    print("Fortran orbital representation routines not found.")

from .numba_oml_representations import gen_orb_atom_scalar_rep, ang_mom_descr
from .numba_oml_representations import ang_mom_descr as numba_ang_mom_descr

from .aux_abinit_classes import AO

class OML_rep_params:
#   Parameters defining how IBO's are represented and biased.
#   tol_orb_cutoff      - consider AO's coefficient zero if it's smaller than tol_orb_cutoff.
#   ibo_atom_rho_comp   - compose IBO representation out of AO's centered on minimal ammount of atoms that would (approximately) account for at least
#                         ibo_atom_rho_comp of the electronic density.
#   l_max               - maximal value of angular momentum.
    def __init__(self, tol_orb_cutoff=0.0,  ibo_atom_rho_comp=None, max_angular_momentum=3, use_Fortran=True,
                    propagator_coup_mat=False, num_prop_times=1, prop_delta_t=1.0,
                    ibo_fidelity_rep=False, add_global_ibo_couplings=False, atom_sorted_pseudo_ibos=False,
                    ofd_coup_mats=False, orb_en_adj=False, ofd_extra_inversions=True):
        self.tol_orb_cutoff=tol_orb_cutoff
        self.ibo_atom_rho_comp=ibo_atom_rho_comp
        self.max_angular_momentum=max_angular_momentum
        self.use_Fortran=use_Fortran
        self.propagator_coup_mat=propagator_coup_mat
        self.num_prop_times=num_prop_times
        self.prop_delta_t=prop_delta_t
        self.add_global_ibo_couplings=add_global_ibo_couplings

        self.atom_sorted_pseudo_ibos=atom_sorted_pseudo_ibos
        self.ofd_coup_mats=ofd_coup_mats
        self.orb_en_adj=orb_en_adj
        self.ofd_extra_inversions=ofd_extra_inversions

        self.ibo_fidelity_rep=ibo_fidelity_rep
    def __str__(self):
        if self.ibo_fidelity_rep:
            str_out="IBOFR,num_prop_times:"+str(self.num_prop_times)+",prop_delta_dt"+str(self.prop_delta_t)
        else:
            str_out="ibo_atom_rho_comp:"+str(self.ibo_atom_rho_comp)+",max_ang_mom:"+str(self.max_angular_momentum)
            if self.propagator_coup_mat:
                str_out+=",prop_delta_t:"+str(self.prop_delta_t)+",num_prop_times:"+str(self.num_prop_times)
        return str_out

def gen_propagator_based_coup_mats(rep_params, hf_orb_coeffs, hf_orb_energies, ovlp_mat):
    naos, num_orbs=hf_orb_coeffs.shape
    inv_hf_orb_coeffs=np.matmul(hf_orb_coeffs.T, ovlp_mat)
    if rep_params.use_Fortran:
        output=np.zeros((naos, naos, rep_params.num_prop_times*2), order='F')
        fgen_ft_coup_mats(inv_hf_orb_coeffs, hf_orb_energies, rep_params.prop_delta_t, naos, num_orbs, rep_params.num_prop_times, output)
        return tuple(np.transpose(output))
    else:
        output=()
        for timestep_counter in range(rep_params.num_prop_times):
            prop_time=(timestep_counter+1)*rep_params.prop_delta_t
            for trigon_func in [math.cos, math.sin]:
                new_mat=np.zeros(num_orbs, dtype=float)
                prop_coeffs=np.array([trigon_func(prop_time*en) for en in hf_orb_energies])
                new_mat=np.matmul(inv_hf_orb_coeffs.T*prop_coeffs, inv_hf_orb_coeffs)
                output=(*output, new_mat)
        return output

def gen_odf_based_coup_mats(rep_params, mo_coeff, mo_energy, mo_occ, ovlp_mat):
    reconstr_mats_kwargs={"mo_energy" : mo_energy, "mo_occ" : mo_occ}
    coupling_matrices=()
    inv_mo_coeff=np.matmul(ovlp_mat, mo_coeff)
    coeff_mats=[inv_mo_coeff, mo_coeff]
    if rep_params.ofd_extra_inversions:
        mat_types_list=[["ovlp", "Fock", "density"] for i in range(2)]
    else:
        mat_types_list=[["ovlp", "Fock"], ["density"]]
    for coeff_mat_id, mat_types in enumerate(mat_types_list):
        added_mats=reconstr_mats(coeff_mats[coeff_mat_id], **reconstr_mats_kwargs,
                                        mat_types=mat_types)
        if len(mat_types)==1:
            coupling_matrices=(*coupling_matrices, added_mats)
        else:
            coupling_matrices=(*coupling_matrices, *added_mats)
    return coupling_matrices

def reconstr_mat(coeff_mat, mo_energy=None, mo_occ=None, mat_type="ovlp"):
    norbs=coeff_mat.shape[1]
    mo_arr=np.ones(norbs, dtype=float)
    for orb_id in range(norbs):
        if mat_type=="density":
            if mo_occ[orb_id]<0.5:
                mo_arr[orb_id]=0.0
        if mat_type=="Fock":
            mo_arr[orb_id]=mo_energy[orb_id]
    return np.matmul(coeff_mat*mo_arr, coeff_mat.T)


def reconstr_mats(mo_coeffs, mo_energy=None, mo_occ=None, mat_types=["ovlp"]):
    output=()
    for mat_type in mat_types:
        output=(*output, reconstr_mat(mo_coeffs, mo_energy=mo_energy, mo_occ=mo_occ, mat_type=mat_type))
    if len(mat_types)==1:
        return output[0]
    else:
        return output

#   Representation of contribution of a single atom to an IBO.

class OML_ibo_atom_rep:
    def __init__(self, atom_ao_range, coeffs, rep_params, angular_momenta, ovlp_mat):
        self.atom_ao_range=atom_ao_range
        self.scalar_reps=np.zeros(rep_params.max_angular_momentum)
        rho_arr=np.zeros(1)
        if rep_params.use_Fortran:
            fang_mom_descr(atom_ao_range, coeffs, angular_momenta, ovlp_mat, rep_params.max_angular_momentum, len(coeffs), self.scalar_reps, rho_arr)
        else:
            numba_ang_mom_descr(ovlp_mat, coeffs, np.array(angular_momenta), self.atom_ao_range, rep_params.max_angular_momentum, self.scalar_reps, rho_arr)
        self.rho=rho_arr[0]
        self.pre_renorm_rho=self.rho
        self.atom_id=None
    def completed_scalar_reps(self, coeffs, rep_params, angular_momenta, coup_mats):
        couplings=np.zeros(scalar_coup_length(rep_params, coup_mats))
        if rep_params.use_Fortran:
            fgen_ibo_atom_scalar_rep(self.atom_ao_range, coeffs, angular_momenta, np.transpose(coup_mats),
                                        len(coup_mats), rep_params.max_angular_momentum, len(coeffs), couplings)
        else:
            #TO-DO: make coup_mats NP-array in the first place?
            gen_orb_atom_scalar_rep(np.array(coup_mats), coeffs, np.array(angular_momenta), self.atom_ao_range, rep_params.max_angular_momentum, couplings)
        self.scalar_reps=np.append(self.scalar_reps, couplings)
        self.scalar_reps/=self.pre_renorm_rho
        if (rep_params.propagator_coup_mat or rep_params.ofd_coup_mats):
            # The parts corresponding to the angular momentum distribution are duplicated, remove:
            self.scalar_reps=self.scalar_reps[rep_params.max_angular_momentum:]
    def energy_readjustment(self, energy_shift, rep_params):
        nam=num_ang_mom(rep_params)
        coup_mat_comp_num=(nam*(3*nam+1))//2
        
        self.scalar_reps[coup_mat_comp_num:2*coup_mat_comp_num]-=self.scalar_reps[:coup_mat_comp_num]*energy_shift
    def __str__(self):
        return "OML_ibo_atom_rep,rho:"+str(self.rho)
    def __repr__(self):
        return str(self)

def ang_mom_descr(atom_ao_range, coeffs, angular_momenta, ovlp_mat, max_angular_momentum):
    ang_mom_distr=np.zeros(max_angular_momentum)
    for ang_mom_counter in range(max_angular_momentum):
        ang_mom=ang_mom_counter+1
        ang_mom_distr[ang_mom_counter]=ibo_atom_atom_coupling(atom_ao_range, ang_mom, ang_mom, coeffs, angular_momenta, ovlp_mat)
    rho=ibo_atom_atom_coupling(atom_ao_range, 0, 0, coeffs, angular_momenta, ovlp_mat)+sum(ang_mom_distr)
    return ang_mom_distr, rho

def avail_ang_mom(rep_params):
    return range(num_ang_mom(rep_params))

def scalar_coup_length(rep_params, coup_mats):
    nam=num_ang_mom(rep_params)
    return len(coup_mats)*(nam**2+(nam*(nam+1))//2)

def num_ang_mom(rep_params):
    return rep_params.max_angular_momentum+1

def ibo_atom_atom_coupling(atom_ao_range, ang_mom1, ang_mom2, coeffs, angular_momenta, matrix, same_atom=True):
    coupling=0.0
    cur_ao_list=list(range(*atom_ao_range))
    if same_atom:
        other_ao_list=cur_ao_list
    else:
        other_ao_list=list(range(atom_ao_range[0]))+list(range(atom_ao_range[1], len(angular_momenta)))

    for aid1 in cur_ao_list:
        if angular_momenta[aid1]==ang_mom1:
            for aid2 in other_ao_list:
                if angular_momenta[aid2]==ang_mom2:
                    coupling+=coeffs[aid1]*coeffs[aid2]*matrix[aid1, aid2]
    return coupling

#   Auxiliary functions.
def scalar_rep_length(oml_comp):
    try: # if that's a Slater pair list
        first_comp=oml_comp.comps[0]
    except AttributeError: # if that's a compound list
        first_comp=oml_comp
    return len(first_comp.orb_reps[0].ibo_atom_reps[0].scalar_reps)

# Generates a representation of what different components of atomic components of IBOs correspond to.
# TO-DO check whether it's the same for Fortran and Python implementations?
def component_id_ang_mom_map(rep_params):
    output=[]
    if rep_params.propagator_coup_mat:
        # Real and imaginary propagator components for each propagation time plus the overlap matrix.
        num_coup_matrices=rep_params.num_prop_times*2+1
    else:
        # Components corresponding to the angular momentum distribution.
        if rep_params.ofd_coup_mats:
            num_coup_matrices=3
            if rep_params.ofd_extra_inversions:
                num_coup_matrices*=2
        else:
            num_coup_matrices=3 # F, J, and K; or overlap, F, and density
    if not (rep_params.propagator_coup_mat or rep_params.ofd_coup_mats):
        for ang_mom in range(1, num_ang_mom(rep_params)):
            output.append([ang_mom, ang_mom, -1, True])
    for coup_mat_id in range(num_coup_matrices):
        for same_atom in [True, False]:
            for ang_mom1 in range(num_ang_mom(rep_params)):
                for ang_mom2 in range(num_ang_mom(rep_params)):
                    if not (same_atom and (ang_mom1>ang_mom2)):
                        output.append([ang_mom1, ang_mom2, coup_mat_id, same_atom])
    return output



def generate_atom_ao_ranges(mol):
    ao_sliced_with_shells=mol.aoslice_by_atom()
    output=[]
    for atom_data in ao_sliced_with_shells:
        output.append(atom_data[2:4])
    return np.array(output)

def placeholder_ibo_rep(rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats):
    placeholder_ibo_coeffs=np.zeros(len(ovlp_mat))
    placeholder_ibo_coeffs[0]=1.0
    placeholder_rep=OML_ibo_rep(placeholder_ibo_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats)
    placeholder_rep.virtual=True
    return placeholder_rep

#   Generate an array of IBO representations.
def generate_ibo_rep_array(ibo_mat, rep_params, aos, atom_ao_ranges, ovlp_mat, *coupling_mats):
    atom_ids=[ao.atom_id for ao in aos]
    angular_momenta=[ao.angular for ao in aos]
    if len(ibo_mat)==0:
        return [placeholder_ibo_rep(rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats)]
    # It's important that ovlp_mat appears first in this array.
    ibo_tmat=ibo_mat.T
    output=[OML_ibo_rep_from_coeffs(ibo_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats) for ibo_coeffs in ibo_tmat]
    # TO-DO add definition of add_global_couplings?
    if rep_params.add_global_ibo_couplings:
        for ibo_id in range(len(output)):
            output[ibo_id].add_global_couplings(ibo_tmat, ibo_id, rep_params, atom_ids, angular_momenta, coupling_mats)
    return output

def gen_atom_sorted_pseudo_ibos(ibo_rep_arr):
    backup_placeholder_arr=[]
    atom_sorted_areps={}
    ibo_occ=None
    for ibo_rep in ibo_rep_arr:
        if ibo_rep.virtual:
            backup_placeholder_arr.append(ibo_rep)
        else:
            if ibo_occ is None:
                ibo_occ=ibo_rep.rho
            for ibo_atom_rep in ibo_rep.ibo_atom_reps:
                cur_atom_id=ibo_atom_rep.atom_id
                if cur_atom_id in atom_sorted_areps:
                    atom_sorted_areps[cur_atom_id].append(ibo_atom_rep)
                else:
                    atom_sorted_areps[cur_atom_id]=[ibo_atom_rep]
    output=[]
    for atom_id in atom_sorted_areps:
        output.append(OML_ibo_rep(None, atom_sorted_areps[atom_id]))
        output[-1].rho=ibo_occ
    if len(output)==0:
        return backup_placeholder_arr
    else:
        return output

#   Representation of an IBO from atomic contributions.
class OML_ibo_rep:
    def __init__(self, ibo_coeffs, ibo_atom_reps):
        self.rho=0.0
        self.full_coeffs=ibo_coeffs
        self.virtual=False
        self.ibo_atom_reps=ibo_atom_reps
    def add_global_ibo_couplings(self, all_ibo_coeffs, ibo_id, rep_params, atom_ids, angular_momenta, coup_mats):
        global_ibo_couplings=np.zeros(len(coup_mats)*((max_angular_momentum+1)*(3*max_angular_momentum+4))/2)
        fgen_ibo_global_couplings(all_ibo_coeffs, ibo_id, angular_momenta, rep_params.max_angular_momentum,
                    len(angular_momenta), coup_mats, coup_mats.shape[2], global_ibo_coupling)
        # TO-DO recheck????
        for ibo_arep_counter in range(len(self.ibo_atom_reps)):
            self.ibo_atom_reps[ibo_arep_counter].scalar_reps=np.append(self.ibo_atom_reps[ibo_arep_counter].scalar_reps,
                                        global_ibo_couplings)
    def orbital_energy_readjustment(self, Fock_mat, rep_params):
        energy_shift=np.dot(self.full_coeffs, np.matmul(Fock_mat, self.full_coeffs))
        for ibo_arep_counter in range(len(self.ibo_atom_reps)):
            self.ibo_atom_reps[ibo_arep_counter].energy_readjustment(energy_shift, rep_params)


def OML_ibo_rep_from_coeffs(ibo_coeffs, rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coup_mats):
    atom_list=[]
    prev_atom=-1
    for atom_id, ao_coeff in zip(atom_ids, ibo_coeffs):
        if abs(ao_coeff)>rep_params.tol_orb_cutoff:
            if prev_atom != atom_id:
                atom_list.append(atom_id)
                prev_atom=atom_id
    # Each of the resulting groups of AOs is represented with OML_ibo_atom_rep object.
    ibo_atom_reps=[]
    for atom_id, atom_ao_range in enumerate(atom_ao_ranges):
        cur_ibo_atom_rep=OML_ibo_atom_rep(atom_ao_range, ibo_coeffs, rep_params, angular_momenta, ovlp_mat)
        cur_ibo_atom_rep.atom_id=atom_id
        ibo_atom_reps.append(cur_ibo_atom_rep)
    ibo_atom_reps=weighted_array(ibo_atom_reps)
    
    ibo_atom_reps.normalize_sort_rhos()
    # Try to decrease the number of atomic representations, leaving only the most relevant ones.
    ibo_atom_reps.cutoff_minor_weights(remaining_rho=rep_params.ibo_atom_rho_comp)
    for ibo_arep_counter in range(len(ibo_atom_reps)):
        ibo_atom_reps[ibo_arep_counter].completed_scalar_reps(ibo_coeffs, rep_params, angular_momenta, coup_mats)
    return OML_ibo_rep(ibo_coeffs, ibo_atom_reps)

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
        if (remaining_rho is not None) and (len(self)>1):
            ignored_rhos=0.0
            for remaining_length in range(len(self),0,-1):
                upper_cutoff=self[remaining_length-1].rho
                cut_rho=upper_cutoff*remaining_length+ignored_rhos
                if cut_rho>(1.0-remaining_rho):
                    density_cut=(1.0-remaining_rho-ignored_rhos)/remaining_length
                    break
                else:
                    ignored_rhos+=upper_cutoff
            del(self[remaining_length:])
            for el_id in range(remaining_length):
                self[el_id].rho=max(0.0, self[el_id].rho-density_cut) # max was introduced in case there is some weird numerical noise.
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
        # TO-DO is there a way to get true orbitals instead???
        pseudo_ens, pseudo_orbs=np.linalg.eigh(vhf)
        fgen_ft_coup_mats(pseudo_orbs.T, pseudo_ens, rep_params.prop_delta_t, naos, rep_params.num_prop_times, prop_mats)
        for mat_id, mat in enumerate(prop_mats.T):
            prp_wovlp[mat_id]=np.matmul(oml_compound.ovlp_mat, mat)
        for orb_id, orb_coeffs in enumerate(oml_compound.ibo_mat[ispin].T):
            for prop_time_id, prop_mat in enumerate(prop_mats.T):
                orb_reps[cur_orb_id, prop_time_id]=np.dot(orb_coeffs,np.matmul(prop_mat,orb_coeffs))
            cur_orb_id+=1
    return orb_reps



