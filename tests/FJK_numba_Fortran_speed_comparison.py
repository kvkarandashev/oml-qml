
import numpy as np
import qml, random, sys

# Auxiliary functions normally imported from a test module.

### A family of classes corresponding to different representations.
class representation:
    def check_param_validity(self, compound_list_in):
        pass
    def compound_list(self, xyz_list):
        return [self.xyz2compound(xyz=f) for f in xyz_list]
    def init_compound_list(self, comp_list=None, xyz_list=None, parallel=False, disable_openmp=True):
        if xyz_list is not None:
            comp_list=self.compound_list(xyz_list)
        if parallel:
            return embarassingly_parallel(self.initialized_compound, comp_list, disable_openmp=disable_openmp)
        else:
            return [self.initialized_compound(compound=comp) for comp in comp_list]
    def xyz2compound(self, xyz=None):
        return qml.Compound(xyz=xyz)
    def check_compound_defined(self, compound=None, xyz=None):
        if compound==None:
            return self.xyz2compound(xyz=xyz)
        else:
            return compound
    def adjust_representation(self, compound_array):
        return compound_array


class OML_representation(representation):
    def __init__(self,  use_Huckel=False, optimize_geometry=False, calc_type="HF",
                    basis="sto-3g", software="pySCF", pyscf_calc_params=None, use_pyscf_localization=True,
                    localization_procedure="IBO", **rep_params_kwargs):
        self.rep_params=qml.oml_representations.OML_rep_params(**rep_params_kwargs)
        self.OML_compound_kwargs={"use_Huckel" : use_Huckel, "optimize_geometry" : optimize_geometry, "calc_type" : calc_type,
                            "software" : software, "pyscf_calc_params" : pyscf_calc_params, "use_pyscf_localization" : use_pyscf_localization,
                            "basis" : basis, "localization_procedure" : localization_procedure}
    def xyz2compound(self, xyz=None):
        return qml.oml_compound.OML_compound(xyz = xyz, mats_savefile = xyz, **self.OML_compound_kwargs)
    def compound_list(self, xyz_list):
        return qml.OML_compound_list_from_xyzs(xyz_list, **self.OML_compound_kwargs)
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_orb_reps(self.rep_params)
        return comp
    def init_compound_list(self, comp_list=None, xyz_list=None, disable_openmp=True):
        if comp_list is not None:
            new_list=comp_list
        else:
            new_list=self.compound_list(xyz_list)
        new_list.generate_orb_reps(self.rep_params, disable_openmp=disable_openmp)
        return new_list
    def __str__(self):
        return "OML_rep,"+str(self.rep_params)
        
        
class OML_Slater_pair_rep(OML_representation):
    def __init__(self, second_charge=0, second_orb_type="standard_IBO", second_calc_type="HF", second_spin=None, **OML_representation_kwargs):
        super().__init__(**OML_representation_kwargs)
        self.second_calc_kwargs={"second_calc_type" : second_calc_type, "second_orb_type" : second_orb_type, "second_charge" : second_charge, "second_spin" : second_spin}
    def xyz2compound(self, xyz=None):
        return qml.oml_compound.OML_Slater_pair(xyz=xyz, **self.second_calc_kwargs, **self.OML_compound_kwargs)
    def compound_list(self, xyz_list):
        return qml.OML_Slater_pair_list_from_xyzs(xyz_list, **self.second_calc_kwargs, **self.OML_compound_kwargs)
    def __str__(self):
        return "OML_Slater_pair,"+self.second_calc_kwargs["second_orb_type"]+","+str(self.rep_params)


def dirs_xyz_list(QM9_dir):
    import glob
    output=glob.glob(QM9_dir+"/*.xyz")
    output.sort()
    return output

seed=1
num_test_mols_1=50
num_test_mols_2=40

test_xyz_dir="./qm7"
all_xyzs=dirs_xyz_list(test_xyz_dir)
tested_xyzs_1=random.Random(seed).sample(all_xyzs, num_test_mols_1)

tested_xyzs_2=random.Random(seed+1).sample(all_xyzs, num_test_mols_2)

my_representation=OML_Slater_pair_rep(max_angular_momentum=1, use_Fortran=True, ibo_atom_rho_comp=0.95,
                                        second_calc_type="UHF", second_orb_type="IBO_HOMO_removed")
oml_compounds_1=my_representation.init_compound_list(xyz_list=tested_xyzs_1, disable_openmp=True)
oml_compounds_2=my_representation.init_compound_list(xyz_list=tested_xyzs_2, disable_openmp=False)

oml_samp_orbs=qml.oml_kernels.random_ibo_sample(oml_compounds_1, pair_reps=True)

width_params=qml.oml_kernels.oml_ensemble_widths_estimate(oml_samp_orbs)

sigmas=np.array([0.5, *width_params])

kernels_wders_sym={}
kernels_wders_asym={}

from datetime import datetime as dt

for use_Fortran in [True, False]:
    print("Fortran used:", use_Fortran)
    start_sym=dt.now()
    kernels_wders_sym[use_Fortran]=qml.oml_kernels.gauss_sep_IBO_sym_kernel(oml_compounds_1, sigmas, with_ders=True, global_Gauss=True, use_Fortran=use_Fortran)
    print("sym kernel calculation", dt.now()-start_sym)
    start_asym=dt.now()
    kernels_wders_asym[use_Fortran]=qml.oml_kernels.gauss_sep_IBO_kernel(oml_compounds_1, oml_compounds_2, sigmas, with_ders=True,
                                                        global_Gauss=True, use_Fortran=use_Fortran)
    print("asym kernel calculation", dt.now()-start_asym)
