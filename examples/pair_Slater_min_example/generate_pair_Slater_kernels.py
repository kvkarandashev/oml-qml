# Minimal example that calculates model MAEs for LUMO and HOMO energies, and the gap.
# Note that it requires having a directory with QM9 xyz files.


# For brevity import some functions from python_script_modules/learning_curve_building
import os, random, qml, glob
import numpy as np
from qml import OML_Slater_pair_list_from_xyzs
from qml.oml_representations import OML_rep_params
from qml.oml_kernels import GMO_kernel_params, random_ibo_sample, oml_ensemble_widths_estimate, GMO_sep_IBO_sym_kernel, GMO_sep_IBO_kernel
from qml.utils import dump2pkl

def_float_format='{:.8E}'

quant_names=['HOMO eigenvalue', 'LUMO eigenvalue', 'HOMO-LUMO gap']
ibo_types=["IBO_HOMO_removed", "IBO_LUMO_added", "IBO_first_excitation"]

opt_sigma_rescalings=[0.5, 0.25, 1.0]
opt_final_sigmas=[8.0, 2.0, 1.0]

seed=1

# Replace with path to QM9 directory
QM9_dir=os.environ["DATA"]+"/QM9_formatted"

train_num=500
check_num=100

xyz_list=glob.glob(QM9_dir+"/*.xyz")
xyz_list.sort()
random.seed(seed)
random.shuffle(xyz_list)

os.environ["OML_NUM_PROCS"]=os.environ["OMP_NUM_THREADS"] # OML_NUM_PROCS says how many processes to use during joblib-parallelized parts; by default most of the latter disable OpenMP parallelization.

oml_representation_parameters=OML_rep_params(ibo_atom_rho_comp=0.95, max_angular_momentum=1, use_Fortran=True)


def get_comps(xyz_list, oml_representation_parameters, ibo_type):
    comps=OML_Slater_pair_list_from_xyzs(xyz_list, calc_type="HF", second_calc_type="UHF",
                                second_orb_type=ibo_type, basis='sto-3g')
    comps.generate_orb_reps(oml_representation_parameters)
    return comps

for quant_name, ibo_type, opt_sigma_rescaling, opt_final_sigma in zip(quant_names, ibo_types, opt_sigma_rescalings, opt_final_sigmas):

    training_comps=get_comps(xyz_list[:train_num], oml_representation_parameters, ibo_type)
    # Find standard deviations of representation vectors.
    orb_sample=random_ibo_sample(training_comps, pair_reps=True)
    stddevs=oml_ensemble_widths_estimate(orb_sample)
    # (Rescale those deviations to get estimates of optimal hyperparameters.)
    width_params=stddevs/opt_sigma_rescaling
    gmo_kernel_parameters=GMO_kernel_params(final_sigma=opt_final_sigma, use_Fortran=True,
                        normalize_lb_kernel=True, use_Gaussian_kernel=True, pair_reps=True, width_params=width_params)
    # Generate the training and check kernels.
    K_train=GMO_sep_IBO_sym_kernel(training_comps, gmo_kernel_parameters)

    check_comps=get_comps(xyz_list[-check_num:], oml_representation_parameters, ibo_type)
    K_check=GMO_sep_IBO_kernel(check_comps, training_comps, gmo_kernel_parameters)
    print("Dumping for:", ibo_type)
    dump2pkl(K_train, "training_kernel_"+ibo_type+".pkl")
    dump2pkl(K_check, "check_kernel_"+ibo_type+".pkl")
