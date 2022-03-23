from learning_curve_building import dirs_xyz_list, logfile, OML_Slater_pair_rep
import numpy as np
import qml, random, sys

seed=5
num_test_mols_1=50
num_test_mols_2=40

logfile_name=sys.argv[1]

test_xyz_dir="./qm7"
all_xyzs=dirs_xyz_list(test_xyz_dir)
tested_xyzs_1=random.Random(seed).sample(all_xyzs, num_test_mols_1)

tested_xyzs_2=random.Random(seed+1).sample(all_xyzs, num_test_mols_2)

logfile=logfile(logfile_name)

my_representation=OML_Slater_pair_rep(max_angular_momentum=1, use_Fortran=True, ibo_atom_rho_comp=0.95, second_orb_type="IBO_HOMO_removed",
                                            software="xTB", localization_procedure="Boys", ofd_coup_mats=True, ofd_extra_inversions=True)


oml_compounds_1=my_representation.init_compound_list(xyz_list=tested_xyzs_1, disable_openmp=True)
oml_compounds_2=my_representation.init_compound_list(xyz_list=tested_xyzs_2, disable_openmp=False)

oml_samp_orbs=qml.oml_kernels.random_ibo_sample(oml_compounds_1, pair_reps=True)

width_params=qml.oml_kernels.oml_ensemble_widths_estimate(oml_samp_orbs)

logfile.write("Width params")
logfile.write(width_params)

sigmas=np.array([1.0, *width_params])

logfile.write("xyz list 1")
logfile.write(tested_xyzs_1)

logfile.write("xyz list 2")
logfile.write(tested_xyzs_2)

logfile.write("kernel_11")
kernel_wders=qml.oml_kernels.gauss_sep_IBO_sym_kernel(oml_compounds_1, sigmas, with_ders=True)
logfile.randomized_export_3D_arr(kernel_wders, seed+2)

logfile.write("kernel_12")
kernel_wders=qml.oml_kernels.gauss_sep_IBO_kernel(oml_compounds_1, oml_compounds_2, sigmas, with_ders=True)
logfile.randomized_export_3D_arr(kernel_wders, seed+2)
logfile.close()
