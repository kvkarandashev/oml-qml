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

from .compound import Compound
from .oml_representations import generate_ibo_rep_array, gen_fock_based_coup_mats, weighted_array, generate_ibo_spectral_rep_array
import jax.numpy as jnp
import numpy as np
import math
import pickle
from pyscf import lo, gto, scf, dft
from os.path import isfile

mf_creator={"HF" : scf.RHF, "UHF" : scf.UHF, "KS" : dft.RKS, "UKS" : dft.UKS}

is_restricted={"HF" : True, "UHF" : False, "KS" : True, "UKS" : False}

is_HF={"HF" : True, "UHF" : True, "KS" : False, "UKS" : False}

is_KS={"HF" : False, "UHF" : False, "KS" : True, "UKS" : True}

neglect_orb_occ=0.1


class pySCFNotConvergedError(Exception):
    pass

class OML_compound(Compound):
    """ 'Orbital Machine Learning (OML) compound' class is used to store data normally
        part of the compound class along with results of ab initio calculations done with
        the pySCF package.
        
        xyz             - xyz file use to create base Compound object.
        mats_savefile   - the file where results of the ab initio calculations are stored;
                          if it is the name is specified then the results would be imported from the file if it exists
                          or saved to the file otherwise.
        calc_type       - type of the calculation (for now only HF with IBO localization and the default basis set are supported).
    """
    def __init__(self, xyz = None, mats_savefile = None, calc_type="HF", basis="sto-3g", used_orb_type="standard_IBO", use_Huckel=False, optimize_geometry=False, charge=0,
            dft_xc='lda,vwn', dft_nlc=''):
        super().__init__(xyz=xyz)

        self.calc_type=calc_type
        self.charge=charge
        self.mats_savefile=mats_savefile
        self.basis=basis
        self.used_orb_type=used_orb_type
        self.use_Huckel=use_Huckel
        self.optimize_geometry=optimize_geometry
        if is_KS[self.calc_type]:
            self.dft_xc=dft_xc
            self.dft_nlc=dft_nlc
        if self.mats_savefile is None:
            self.mats_created=False
            self.pyscf_chkfile_avail=False
        else:
            if self.mats_savefile.endswith(".pkl"):
                self.pyscf_chkfile=self.mats_savefile[:-3]+"chkfile"
            else:
                savefile_prename=self.mats_savefile+"."+self.calc_type+"."+self.basis
                if self.use_Huckel:
                    savefile_prename+="."+"Huckel"
                if self.optimize_geometry:
                    savefile_prename+="."+"geom_opt"
                if self.charge != 0:
                    savefile_prename+="."+"charge_"+str(self.charge)
                if is_KS[self.calc_type]:
                    savefile_prename+=".xc_"+self.dft_xc+".nlc_"+str(self.dft_nlc)
                self.pyscf_chkfile=savefile_prename+".chkfile"
                self.mats_savefile=savefile_prename+"."+self.used_orb_type+".pkl"
            self.mats_created=isfile(self.mats_savefile)
            self.pyscf_chkfile_avail=isfile(self.pyscf_chkfile)
        if self.mats_created:
            # Import ab initio results from the savefile.
            savefile = open(self.mats_savefile, "rb")
            precalc_mats = pickle.load(savefile)
            savefile.close()
            precalc_vals=np.load(self.mats_savefile, allow_pickle=True)
            self.mo_coeff=precalc_vals["mo_coeff"]
            self.mo_occ=precalc_vals["mo_occ"]
            self.mo_energy=precalc_vals["mo_energy"]
            self.aos=precalc_vals["aos"]
            self.atom_ao_ranges=precalc_vals["atom_ao_ranges"]
            self.e_tot=precalc_vals["e_tot"]

            self.ovlp_mat=precalc_vals["ovlp_mat"]
            self.iao_mat=precalc_vals["iao_mat"]
            self.ibo_mat=precalc_vals["ibo_mat"]

            if is_HF[self.calc_type]:
                self.j_mat=precalc_vals["j_mat"]
                self.k_mat=precalc_vals["k_mat"]
                self.fock_mat=precalc_vals["fock_mat"]
            if self.optimize_geometry:
                self.opt_coords=precalc_vals["opt_coords"]
        else:
            self.mo_coeff=None
            self.mo_occ=None
            self.mo_energy=None
            self.aos=None
            self.atom_ao_ranges=None
            self.e_tot=None

            self.ovlp_mat=None
            self.iao_mat=None
            self.ibo_mat=None
            if is_HF[self.calc_type]:
                self.j_mat=None
                self.k_mat=None
                self.fock_mat=None
            if self.optimize_geometry:
                self.opt_coords=None
        self.orb_reps=[]
    def run_calcs(self, pyscf_calc_params=None):
        """ Runs the ab initio calculations if they are necessary.

            pyscf_calc_params   - object of OML_pyscf_calc_params class containing parameters of the pySCF calculation. (To be made more useful.)
        """
        if not self.mats_created:
            # Run the pySCF calculations.
            from qml.oml_representations import generate_ao_arr, generate_atom_ao_ranges
            pyscf_mol=self.generate_pyscf_mol()
            mf=self.generate_pyscf_mf(pyscf_mol)
            if self.optimize_geometry:
                from pyscf.geomopt.geometric_solver import optimize
                pyscf_mol=optimize(mf)
                self.opt_coords=pyscf_mol.atom_coords(unit='Ang')
                mf=self.generate_pyscf_mf(pyscf_mol)
            ### Special operations.
            if self.used_orb_type != "standard_IBO":
                mf.mo_occ=self.alter_mo_occ(mf.mo_occ)
            ###
            self.mo_coeff=self.adjust_spin_mat_dimen(mf.mo_coeff)
            self.mo_occ=self.adjust_spin_mat_dimen(mf.mo_occ)
            self.mo_energy=self.adjust_spin_mat_dimen(mf.mo_energy)
            self.aos=generate_ao_arr(pyscf_mol)
            self.atom_ao_ranges=generate_atom_ao_ranges(pyscf_mol)
            self.e_tot=mf.e_tot

            occ_orb_arrs=self.occ_orbs()
            if is_HF[self.calc_type]:
                self.j_mat=self.adjust_spin_mat_dimen(mf.get_j())
                self.k_mat=self.adjust_spin_mat_dimen(mf.get_k())
                self.fock_mat=self.adjust_spin_mat_dimen(mf.get_fock())
                self.ovlp_mat=jnp.array(mf.get_ovlp())
            else:
                self.ovlp_mat=generate_ovlp_mat(pyscf_mol)
            self.iao_mat=[]
            self.ibo_mat=[]
            for occ_orb_arr in occ_orb_arrs:
                self.iao_mat.append(jnp.array(lo.iao.iao(pyscf_mol, occ_orb_arr)))
                if pyscf_calc_params is None:
                   self.ibo_mat.append(jnp.array(lo.ibo.ibo(pyscf_mol, occ_orb_arr, max_iter=5000)))
                else:
                   self.ibo_mat.append(jnp.array(lo.ibo.ibo(pyscf_mol, occ_orb_arr, max_iter=pyscf_calc_params.ibo_max_iter,grad_tol=pyscf_calc_params.ibo_grad_tol)))
            self.mats_created=True
            if self.mats_savefile is not None:
                #TO-DO Check ways for doing it in a less ugly way.
                saved_data={"mo_coeff" : self.mo_coeff, "mo_occ" : self.mo_occ, "mo_energy" : self.mo_energy, "aos" : self.aos,
                                    "ovlp_mat" : self.ovlp_mat, "iao_mat" : self.iao_mat, "ibo_mat" : self.ibo_mat, "atom_ao_ranges" : self.atom_ao_ranges,
                                    "e_tot" : self.e_tot}
                if is_HF[self.calc_type]:
                    saved_data={**saved_data, "j_mat" : self.j_mat, "k_mat" : self.k_mat, "fock_mat" : self.fock_mat}
                if self.optimize_geometry:
                    saved_data["opt_coords"]=self.opt_coords
                savefile = open(self.mats_savefile, "wb")
                pickle.dump(saved_data, savefile)
                savefile.close()
    def generate_orb_reps(self, rep_params):
        """ Generates orbital representation.

            rep_params  - object of oml_representations.OML_rep_params class containing parameters of the orbital representation.
        """
        if not self.mats_created:
            self.run_calcs()
        if is_restricted[self.calc_type]:
            num_spins=1
        else:
            num_spins=2
        self.orb_reps=[]
        for spin in range(num_spins):
            #   Generate the array of orbital representations.
            if rep_params.fock_based_coup_mat:
                if (rep_params.fbcm_pseudo_orbs and is_HF[self.calc_type]):
                    pseudo_ens, pseudo_orbs=jnp.linalg.eigh(self.fock_mat[spin])
                    coupling_matrices=gen_fock_based_coup_mats(rep_params, pseudo_orbs, pseudo_ens)
                else:
                    coupling_matrices=gen_fock_based_coup_mats(rep_params, self.mo_coeff[spin], self.mo_energy[spin])
                if is_HF[self.calc_type]:
                    cur_fock_mat=self.fock_mat[spin]
                else:
                    cur_fock_mat=reconstruct_effective_Hamiltonian(self.mo_coeff[spin], self.mo_energy[spin])
                coupling_matrices=(*coupling_matrices, cur_fock_mat)
            else:
                coupling_matrices=(self.fock_mat[spin], self.j_mat[spin]/orb_occ_prop_coeff(self), self.k_mat[spin]/orb_occ_prop_coeff(self))
            self.orb_reps+=generate_ibo_rep_array(self.ibo_mat[spin], rep_params, self.aos, self.atom_ao_ranges, self.ovlp_mat, *coupling_matrices)
        ibo_occ=orb_occ_prop_coeff(self)
        for orb_rep_counter in range(len(self.orb_reps)):
            self.orb_reps[orb_rep_counter].rho=ibo_occ
#                if rep_params.ibo_spectral_representation: # It would be very funny if this representation proves to be useful.
#                    self.orb_reps=generate_ibo_spectral_rep_array(sibo_mat, rep_params, self.orb_overlap, self.mo_coeff, self.mo_occ, self.mo_energy, self.HOMO_en())
#                else:
    #   Find maximal value of angular momentum for AOs of current molecule.
    def find_max_angular_momentum(self):
        if not self.mats_created:
            self.run_calcs()
        max_angular_momentum=0
        for ao in self.aos:
            max_angular_momentum=max(max_angular_momentum, ao.angular)
        return max_angular_momentum
    def orb_overlap(self, coeffs1, coeffs2):
        return jnp.dot(coeffs1, jnp.matmul(self.ovlp_mat, coeffs2))
    def generate_pyscf_mol(self):
        # Convert between the Mole class used in pySCF and the Compound class used in the rest of QML.
        mol=gto.Mole()
        # atom_coords should be in Angstrom.
        mol.atom=[ [atom_type, atom_coords] for atom_type, atom_coords in zip(self.atomtypes, self.coordinates)]
        mol.charge=self.charge
        mol.spin=self.charge%2
        mol.basis=self.basis
        mol.build()
        return mol
    def generate_pyscf_mf(self, pyscf_mol):
        mf=mf_creator[self.calc_type](pyscf_mol)
        if is_KS[self.calc_type]:
            mf.xc=self.dft_xc
            mf.nlc=self.dft_nlc
        mf.chkfile=self.pyscf_chkfile
        if self.use_Huckel:
            mf.init_guess='huckel'
            mf.max_cycle=-1
        else:
            mf.max_cycle=5000
            if self.pyscf_chkfile_avail:
                mf.init_guess='chkfile'
        mf.run()
        if not (mf.converged or self.use_Huckel):
            print("WARNING: A SCF calculation failed to converge at ", mf.max_cycle, " cycles.")
            raise pySCFNotConvergedError
#        subprocess.run(["rm", '-f', chkfile])
        return mf

    def alter_mo_occ(self, mo_occ):
        if is_restricted[self.calc_type]:
            true_mo_occ=mo_occ
        else:
            true_mo_occ=mo_occ[0]
        for i, orb_occ in enumerate(true_mo_occ):
            if orb_occ < neglect_orb_occ:
                if is_restricted[self.calc_type]:
                    if add_LUMO(self):
                        mo_occ[i]=2.0
                    if remove_HOMO(self):
                        mo_occ[i-1]=0.0
                else:
                    if add_LUMO(self):
                        mo_occ[0][i]=1.0
                    if remove_HOMO(self):
                        mo_occ[0][i-1]=0.0
                break
        return mo_occ

    def occ_orbs(self):
        output=[]
        for mo_occ_arr, mo_coeff_arr in zip(self.mo_occ, self.mo_coeff):
            cur_occ_orbs=[]
            for basis_func in mo_coeff_arr:
                cur_line=[]
                for occ, orb_coeff in zip(mo_occ_arr, basis_func):
                    if occ > neglect_orb_occ:
                        cur_line.append(orb_coeff)
                cur_occ_orbs.append(cur_line)
            output.append(np.array(cur_occ_orbs))
        return output
    def HOMO_en(self):
        # HOMO energy.
        return self.mo_energy[0][self.LUMO_orb_id()-1]
    def LUMO_en(self):
        # LUMO energy.
        return self.mo_energy[0][self.LUMO_orb_id()]
    def HOMO_LUMO_gap(self):
        return self.LUMO_en()-self.HOMO_en()
    def LUMO_orb_id(self):
        # index of the LUMO orbital.
        for orb_id, occ in enumerate(self.mo_occ[0]):
            if occ < neglect_orb_occ:
                return orb_id
    def adjust_spin_mat_dimen(self, matrices):
        if is_restricted[self.calc_type]:
            return jnp.array([matrices])
        else:
            return jnp.array(matrices)

def remove_HOMO(oml_comp):
    used_orb_type=oml_comp.used_orb_type
    return ((used_orb_type=="IBO_HOMO_removed") or (used_orb_type=="IBO_first_excitation"))

def add_LUMO(oml_comp):
    used_orb_type=oml_comp.used_orb_type
    return ((used_orb_type=="IBO_LUMO_added") or (used_orb_type=="IBO_first_excitation"))

def generate_ovlp_mat(pyscf_mol):
    mf=scf.RHF(pyscf_mol)
    return jnp.array(mf.get_ovlp())

def reconstruct_effective_Hamiltonian(mo_coeff, mo_energy):
    from jax.ops import index_update
    n=len(mo_coeff)
    diag_mat=np.zeros((n, n))
    for i, el in enumerate(mo_energy):
        diag_mat[i,i]=el
    inv_mat=jnp.linalg.inv(mo_coeff)
    diag_mat=jnp.array(diag_mat)
    return jnp.matmul(inv_mat.T, jnp.matmul(diag_mat, inv_mat))

def orb_occ_prop_coeff(comp):
    if (is_restricted[comp.calc_type]):
        return 2.0
    else:
        return 1.0


class OML_pyscf_calc_params:
#   Parameters of how Fock orbitals and IBOs are calculated.
#   For now just includes IBOs.
    def __init__(self, ibo_max_iter=500, ibo_grad_tol=1.0E-9):
        self.ibo_max_iter=ibo_max_iter
        self.ibo_grad_tol=ibo_grad_tol


class OML_Slater_pair:
    def __init__(self, xyz = None, mats_savefile = None, first_calc_type="HF", second_calc_type="HF", basis="sto-3g", second_charge=0,
        second_orb_type="standard_IBO", optimize_geometry=False, use_Huckel=False):
        comp1=OML_compound(xyz = xyz, mats_savefile = mats_savefile, calc_type=first_calc_type,
                        basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)
        comp2=OML_compound(xyz = xyz, mats_savefile = mats_savefile, calc_type=second_calc_type,
                        basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry, charge=second_charge, used_orb_type=second_orb_type)
        self.comps=[comp1, comp2]
    def run_calcs(self, pyscf_calc_params=None):
        self.comps[0].run_calcs(pyscf_calc_params=pyscf_calc_params)
        self.comps[1].run_calcs(pyscf_calc_params=pyscf_calc_params)
    def generate_orb_reps(self, rep_params):
        self.comps[0].generate_orb_reps(rep_params)
        self.comps[1].generate_orb_reps(rep_params)
