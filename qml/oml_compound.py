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
from os.path import isfile

neglect_orb_occ=0.1

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
    def __init__(self, xyz = None, mats_savefile = None, calc_type="HF", basis="min_bas", used_orb_type="standard_IBO"):
        super().__init__(xyz=xyz)

        self.calc_type=calc_type
        self.mats_savefile=mats_savefile
        self.pyscf_chkfile=None
        self.basis=basis
        self.used_orb_type=used_orb_type
        if self.mats_savefile is None:
            self.mats_created=False
        else:
            if not self.mats_savefile.endswith(".npz"):
                savefile_prename=self.mats_savefile+"."+self.calc_type+"."+self.basis
                self.pyscf_chkfile=savefile_prename+".chkfile"
                self.mats_savefile=savefile_prename+"."+self.used_orb_type+".npz"
            self.mats_created=isfile(self.mats_savefile)
        if self.mats_created:
            # Import ab initio results from the savefile.
            precalc_mats=np.load(self.mats_savefile, allow_pickle=True)
            self.mo_coeff=precalc_mats["mo_coeff"]
            self.mo_occ=precalc_mats["mo_occ"]
            self.mo_energy=precalc_mats["mo_energy"]
            self.aos=precalc_mats["aos"]
            self.atom_ao_ranges=precalc_mats["atom_ao_ranges"]


            self.j_mat=precalc_mats["j_mat"]
            self.k_mat=precalc_mats["k_mat"]
            self.fock_mat=precalc_mats["fock_mat"]
            self.ovlp_mat=precalc_mats["ovlp_mat"]
            self.iao_mat=precalc_mats["iao_mat"]
            self.ibo_mat=precalc_mats["ibo_mat"]
        else:
            self.mo_coeff=None
            self.mo_occ=None
            self.mo_energy=None
            self.aos=None
            self.atom_ao_ranges=None

            self.j_mat=None
            self.k_mat=None
            self.fock_mat=None
            self.ovlp_mat=None
            self.iao_mat=None
            self.ibo_mat=None
        self.orb_reps=[]
    def run_calcs(self, pyscf_calc_params=None):
        """ Runs the ab initio calculations if they are necessary.

            mf_chkfile          - name of pySCF checkfile.
            mf_import_chkfile   - if True the pySCF checkfile will be used to restart pySCF calculation.
            pyscf_calc_params   - object of OML_pyscf_calc_params class containing parameters of the pySCF calculation. (To be made more useful.)
        """
        if not self.mats_created:
            # Run the pySCF calculations.
            from pyscf import scf, lo
            from qml.oml_representations import generate_ao_arr, generate_atom_ao_ranges
            pyscf_mol=self.generate_pyscf_mol()
            if self.calc_type=="HF":
                mf=scf.RHF(pyscf_mol)
            else:
                mf=scf.UHF(pyscf_mol)
            mf.chkfile=self.pyscf_chkfile
            if isfile(self.pyscf_chkfile):
                mf.init_guess='chkfile'
            mf.run()
            # TO-DO think of a nice way to include UHF here.
            ### Special operations.
            if self.used_orb_type=="IBO_HOMO_removed":
                for i, orb_occ in enumerate(mf.mo_occ):
                    if orb_occ < neglect_orb_occ:
                        mf.mo_occ[i-1]=0.0
                        break
            if self.used_orb_type=="IBO_LUMO_added":
                for i, orb_occ in enumerate(mf.mo_occ):
                    if orb_occ < neglect_orb_occ:
                        mf.mo_occ[i]=2.0
                        break
            ###
            self.mo_coeff=jnp.array(mf.mo_coeff)
            self.mo_occ=jnp.array(mf.mo_occ)
            self.mo_energy=jnp.array(mf.mo_energy)
            self.aos=generate_ao_arr(pyscf_mol)
            self.atom_ao_ranges=generate_atom_ao_ranges(pyscf_mol)

            occ_orb_array=self.occ_orbs()
            self.j_mat=jnp.array(mf.get_j())
            self.k_mat=jnp.array(mf.get_k())
            self.fock_mat=jnp.array(mf.get_fock())
            self.ovlp_mat=jnp.array(mf.get_ovlp())
            self.iao_mat=jnp.array(lo.iao.iao(pyscf_mol, occ_orb_array))
            #TO-DO create ibo_mats - one for each spin occupation array. In case of RHF make list of single array.
            if pyscf_calc_params is None:
                self.ibo_mat=lo.ibo.ibo(pyscf_mol, occ_orb_array)
            else:
                self.ibo_mat=jnp.array(lo.ibo.ibo(pyscf_mol, occ_orb_array, max_iter=pyscf_calc_params.ibo_max_iter,grad_tol=pyscf_calc_params.ibo_grad_tol))
            self.mats_created=True
            if self.mats_savefile is not None:
                # Save ab initio results.
                print("printing mats_savefile")
                #TO-DO Once jax.numpy implements savez_compressed put it here instead.
                jnp.savez(self.mats_savefile, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, mo_energy=self.mo_energy,
                                    aos=self.aos, j_mat=self.j_mat, k_mat=self.k_mat, fock_mat=self.fock_mat,
                                    ovlp_mat=self.ovlp_mat, iao_mat=self.iao_mat, ibo_mat=self.ibo_mat, atom_ao_ranges=self.atom_ao_ranges)
    def generate_orb_reps(self, rep_params):
        """ Generates orbital representation.

            rep_params  - object of oml_representations.OML_rep_params class containing parameters of the orbital representation.
        """
        if not self.mats_created:
            self.run_calcs()
        #   Generate the array of orbital representations.
        if rep_params.fock_based_coup_mat:
            if rep_params.characteristic_energy is None:
                characteristic_energy=self.HOMO_LUMO_gap()
            else:
                characteristic_energy=rep_params.characteristic_energy
            coupling_matrices=gen_fock_based_coup_mats(rep_params, self.mo_coeff, self.mo_energy, characteristic_energy)
            coupling_matrices=(*coupling_matrices, self.fock_mat)
        else:
            coupling_matrices=(self.fock_mat, self.j_mat, self.k_mat)
        if rep_params.ibo_spectral_representation: # I would be very funny if this representation proves to be useful.
            self.orb_reps=generate_ibo_spectral_rep_array(self.ibo_mat, rep_params, self.orb_overlap, self.mo_coeff, self.mo_occ, self.mo_energy, self.HOMO_en())
        else:
            # TO-DO via for-loop add orb_reps for ibos from both ibo_mats (or single IBO_mat)
            self.orb_reps=generate_ibo_rep_array(self.ibo_mat, rep_params, self.aos, self.atom_ao_ranges, self.ovlp_mat, *coupling_matrices)
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
        from pyscf import gto
        mol=gto.Mole()
        # atom_coords should be in Angstrom.
        mol.atom=[ [atom_type, atom_coords] for atom_type, atom_coords in zip(self.atomtypes, self.coordinates)]
        mol.build()
        return mol
    def occ_orbs(self):
        # Array of occupied orbitals.
        output=[]
        for basis_func in self.mo_coeff:
            cur_line=[]
            for occ, orb_coeff in zip(self.mo_occ, basis_func):
                if occ > neglect_orb_occ:
                    cur_line.append(orb_coeff)
            output.append(cur_line)
        return np.array(output)
    def HOMO_en(self):
        # HOMO energy.
        return self.mo_energy[self.LUMO_orb_id()-1]
    def LUMO_en(self):
        # LUMO energy.
        return self.mo_energy[self.LUMO_orb_id()]
    def HOMO_LUMO_gap(self):
        return self.LUMO_en()-self.HOMO_en()
    def LUMO_orb_id(self):
        # index of the LUMO orbital.
        for orb_id, occ in enumerate(self.mo_occ):
            if occ < neglect_orb_occ:
                return orb_id
                
                
                
class OML_pyscf_calc_params:
#   Parameters of how Fock orbitals and IBOs are calculated.
#   For now just includes IBOs.
    def __init__(self, ibo_max_iter=500, ibo_grad_tol=1.0E-9):
        self.ibo_max_iter=ibo_max_iter
        self.ibo_grad_tol=ibo_grad_tol


class OML_Slater_pair:
    def __init__(self, xyz = None, mats_savefile = None, calc_type="IBO_HF_min_bas", second_orb_type="IBO_HOMO_removed"):
        comp1=OML_compound(xyz = xyz, mats_savefile = mats_savefile, calc_type=calc_type)
        comp2=OML_compound(xyz = xyz, mats_savefile = mats_savefile, calc_type=calc_type, used_orb_type=second_orb_type)
        self.comps=[comp1, comp2]
    def run_calcs(self, pyscf_calc_params):
        self.comps[0].run_calcs(pyscf_calc_params=pyscf_calc_params)
        self.comps[1].run_calcs(pyscf_calc_params=pyscf_calc_params)
    def generate_orb_reps(self, rep_params):
        self.comps[0].generate_orb_reps(rep_params)
        self.comps[1].generate_orb_reps(rep_params)
