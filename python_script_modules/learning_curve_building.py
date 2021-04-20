
import qml
import jax.numpy as jnp
import math
import numpy as np
from qml.math import cho_solve as fcho_solve
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve as scipy_cho_solve

byprod_result_ending=".brf"

intermediate_res_ending=".dat"

def_float_format='{:.8E}'

### Auxiliary functions to use in the representation class.
# Appears, for example, in CM representation.
def find_max_size(compound_list):
    output=0
    for mol in compound_list:
        output=max(output, len(mol.atomtypes))
    return output

### END

## Some useful np-based routines. Analogous to some from qml.math,
### introduced for cases when parallelization of qml.math does not work
### properly.
def np_svd_solve(a,b,rcond=0.0):
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
#    w = np.linalg.solve(np.diag(s),c[:len(s)])
    w=np.zeros(len(s))
    for i, (cur_c, cur_s) in enumerate(zip(c[:len(s)], s)):
        if (abs(cur_s>rcond)):
            w[i]=cur_c/cur_s
    x = np.dot(v.T,w)
    return x

def np_eigh_posd_solve(mat, vec, rcond=1e-9):
    eigenvals, eigenvecs=np.linalg.eigh(mat)
    vec_transformed=np.dot(vec, eigenvecs)
    for eigenval_id, eigenval in enumerate(eigenvals):
        if eigenval > rcond:
            vec_transformed[eigenval_id]/=eigenval
        else:
            vec_transformed[eigenval_id]=0
    return np.dot(eigenvecs, vec_transformed)


def np_cho_solve(mat, vec):
    c, low=cho_factor(mat)
    return scipy_cho_solve((c, low), vec)

def np_cho_solve_wcheck(mat, vec, eigh_rcond=1e-9):
    try:
        return np_cho_solve(mat, vec)
    except np.linalg.LinAlgError:
        print("WARNING: Cholesky failed.")
        return np_eigh_posd_solve(mat, vec, rcond=eigh_rcond)

cho_solve_implementations={'qml' : fcho_solve, 'scipy' : np_cho_solve}

def cho_solve(mat, vec, cho_impl='qml'):
    return cho_solve_implementations[cho_impl](mat, vec)

# Model class.
class KR_model:
    def __init__(self, kernel_function=None, representation=None):
        self.kernel_function=kernel_function
        self.representation=representation
    def __str__(self):
        return str(self.kernel_function)+";"+str(self.representation)
    def adjust_hyperparameters(self, xyz_list=None, init_compound_list=None):
        if init_compound_list is None:
            init_compound_list=self.representation.init_compound_list(xyz_list=xyz_list)
        self.kernel_function.adjust_hyperparameters(init_compound_list)
    def adjust_representation(self, compound_array):
        return self.representation.adjust_representation(compound_array)

### A family of classes corresponding to different representations.
class representation:
    def check_param_validity(self, compound_list_in):
        pass
    def compound_list(self, xyz_list):
        return [self.xyz2compound(xyz=f) for f in xyz_list]
    def init_compound_list(self, comp_list=None, xyz_list=None):
        if xyz_list is not None:
            comp_list=self.compound_list(xyz_list)
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

class CM_representation(representation):
    def __init__(self, max_size=0, sorting="row-norm"):
        self.max_size=max_size
        self.sorting=sorting
    def check_param_validity(self, compound_list_in):
        self.max_size=max(self.max_size, find_max_size(compound_list_in))
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_coulomb_matrix(size=self.max_size, sorting=self.sorting)
        return comp
    def __str__(self):
        return "CM_rep,sorting:"+self.sorting
        

class FCHL_representation(representation):
    def __init__(self, max_size=0, neighbors=0, cut_distance=5.0):
        self.max_size=max_size
        self.neighbors=neighbors
        self.cut_distance=cut_distance
    def check_param_validity(self, compound_list_in):
        self.max_size=max(self.max_size, find_max_size(compound_array_in))
        self.neighbors=self.max_size
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_fchl_representation(self, max_size = self.max_size, neighbors=self.neighbors,
                            cut_distance=self.cut_distance)
        return comp
    def __str__(self):
        return "FCHL"

class SLATM_representation(representation):
    def __init__(self, local=False, sigmas=[0.05,0.05], dgrids=[0.03,0.03], rcut=4.8, pbc='000',
        alchemy=False, rpower=6):
        self.local=local
        self.sigmas=sigmas
        self.dgrids=dgrids
        self.rcut=rcut
        self.pbc=pbc
        self.alchemy=alchemy
        self.rpower=rpower
        self.mbtypes=None
    def check_param_validity(self, compound_list_in):
        from qml.representations import get_slatm_mbtypes
        nuclear_charge_list=[]
        for comp in compound_list_in:
            nuclear_charge_list.append(comp.nuclear_charges)
        self.mbtypes=get_slatm_mbtypes(nuclear_charge_list)
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_slatm(self.mbtypes, local=self.local, sigmas=self.sigmas, dgrids=self.dgrids, rcut=self.rcut, pbc=self.pbc,
        alchemy=self.alchemy, rpower=self.rpower)
        return comp
    def __str__(self):
        return "SLATM"



class OML_representation(representation):
    def __init__(self,  use_Huckel=False, optimize_geometry=False, calc_type="HF",
                    basis="sto-3g", software="pySCF", pyscf_calc_params=None, use_pyscf_localization=True,
                    **rep_params_kwargs):
        self.rep_params=qml.oml_representations.OML_rep_params(**rep_params_kwargs)
        self.OML_compound_kwargs={"use_Huckel" : use_Huckel, "optimize_geometry" : optimize_geometry, "calc_type" : calc_type,
                            "software" : software, "pyscf_calc_params" : pyscf_calc_params, "use_pyscf_localization" : use_pyscf_localization,
                            "basis" : basis}
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
    def __init__(self, second_charge=0, second_orb_type="standard_IBO", second_calc_type="HF", **OML_representation_kwargs):
        super().__init__(**OML_representation_kwargs)
        self.second_calc_kwargs={"second_calc_type" : second_calc_type, "second_orb_type" : second_orb_type, "second_charge" : second_charge}
    def xyz2compound(self, xyz=None):
        return qml.oml_compound.OML_Slater_pair(xyz=xyz, **self.second_calc_kwargs, **self.OML_compound_kwargs)
    def compound_list(self, xyz_list):
        return qml.OML_Slater_pair_list_from_xyzs(xyz_list, **self.second_calc_kwargs, **self.OML_compound_kwargs)
    def __str__(self):
        return "OML_Slater_pair,"+self.second_calc_kwargs["second_orb_type"]+","+str(self.rep_params)

### END

### Auxiliary functions to use in kernel_function class.
# Combine all representation arrays into one array.
def combined_representation_arrays(compound_list):
    return jnp.array([mol.representation for mol in compound_list])
### END

### A family of classes corresponding to different kernel functions.

class kernel_function:
    def __init__(self, lambda_val=0.0, diag_el_unity=False):
        self.lambda_val=lambda_val
        self.diag_el_unity=diag_el_unity
    def km_added_lambda(self, array):
        K=self.sym_kernel_matrix(array)
        self.print_diag_deviation(K)
        # TO-DO: Check whether this is necessary:
        K_size=len(array)
        for i1 in range(K_size):
            for i2 in range(i1):
                true_K=(K[i1,i2]+K[i2,i1])/2
                K[i1,i2]=true_K
                K[i2,i1]=true_K
        if self.diag_el_unity:
            K[jnp.diag_indices_from(K)] = 1.0
        K[jnp.diag_indices_from(K)]+=self.lambda_val
        return K
    def adjust_hyperparameters(self, compound_array):
        pass
    def print_diag_deviation(self, matrix):
        print("#TEST Total deviation of diagonal elements from 1: ", sum(abs(el-1.0) for el in matrix[jnp.diag_indices_from(matrix)]))
    def sym_kernel_matrix(self, array):
        return self.kernel_matrix(array, array)

class standard_geometric_kernel_function(kernel_function):
    def __init__(self, lambda_val=0.0, diag_el_unity=False):
        super().__init__(lambda_val=lambda_val, diag_el_unity=diag_el_unity)
    def make_kernel_input_arrs(self, arr1, arr2):
        inp_arr1=combined_representation_arrays(arr1)
        inp_arr2=combined_representation_arrays(arr2)
        return inp_arr1, inp_arr2

class Gaussian_kernel_function(standard_geometric_kernel_function):
    def __init__(self, sigma, lambda_val=0.0, diag_el_unity=False):
        super().__init__(lambda_val=lambda_val, diag_el_unity=diag_el_unity)
        self.sigma=sigma
    def kernel_matrix(self, arr1, arr2):
        inp_arr1, inp_arr2=self.make_kernel_input_arrs(arr1, arr2)
        from qml.kernels import gaussian_kernel
        return gaussian_kernel(inp_arr1, inp_arr2, self.sigma)
    def __str__(self):
        return "Gaussian_KF,sigma:"+str(self.sigma)

class Laplacian_kernel_function(standard_geometric_kernel_function):
    def __init__(self, sigma, lambda_val=0.0, diag_el_unity=False):
        super().__init__(lambda_val=lambda_val, diag_el_unity=diag_el_unity)
        self.sigma=sigma
    def kernel_matrix(self, arr1, arr2):
        inp_arr1, inp_arr2=self.make_kernel_input_arrs(arr1, arr2)
        from qml.kernels import laplacian_kernel
        return laplacian_kernel(inp_arr1, inp_arr2, self.sigma)
    def __str__(self):
        return "Laplacian_KF,sigma:"+str(self.sigma)

class OML_GMO_kernel_function(kernel_function):
    def __init__(self, lambda_val=1e-9, final_sigma=1.0, sigma_rescale=1.0, use_Fortran=True, pair_reps=True,
                    normalize_lb_kernel=False, use_Gaussian_kernel=False, diag_el_unity=False, width_params=None,
                    density_neglect=1e-9):
        super().__init__(lambda_val=lambda_val, diag_el_unity=diag_el_unity)
        self.kernel_params=qml.oml_kernels.GMO_kernel_params(final_sigma=final_sigma, use_Fortran=use_Fortran,
                        normalize_lb_kernel=normalize_lb_kernel, use_Gaussian_kernel=use_Gaussian_kernel, pair_reps=pair_reps,
                        density_neglect=density_neglect)
        self.sigma_rescale=sigma_rescale
        if width_params is not None:
            self.kernel_params.update_width(jnp.array(width_params)/self.sigma_rescale)
    def adjust_hyperparameters(self, compound_array, var_cutoff_val=0.0):
        orb_sample=qml.oml_kernels.random_ibo_sample(compound_array, pair_reps=self.kernel_params.pair_reps)
        self.kernel_params.update_width(qml.oml_kernels.oml_ensemble_widths_estimate(orb_sample, var_cutoff_val=var_cutoff_val)/self.sigma_rescale)
    def kernel_matrix(self, arr1, arr2):
        return qml.oml_kernels.generate_GMO_kernel(arr1, arr2, self.kernel_params)
    def sym_kernel_matrix(self, array):
        return qml.oml_kernels.generate_GMO_kernel(array, array, self.kernel_params, sym_kernel_mat=True)
    def __str__(self):
        output="GMO,sigma_rescale:"+str(self.sigma_rescale)
        if self.kernel_params.use_Gaussian_kernel:
            output+=",fin_sigma:"+str(self.kernel_params.final_sigma)
        return output

class OML_GMO_sep_IBO_kernel_function(OML_GMO_kernel_function):
    def kernel_matrix(self, arr1, arr2):
        return qml.oml_kernels.GMO_sep_IBO_kernel(arr1, arr2, self.kernel_params)
    def sym_kernel_matrix(self, array):
        return qml.oml_kernels.GMO_sep_IBO_sym_kernel(array, self.kernel_params)
    def __str__(self):
        output="GMO_sep_IBO,sigma_rescale:"+str(self.sigma_rescale)
        if self.kernel_params.use_Gaussian_kernel:
            output+=",fin_sigma:"+str(self.kernel_params.final_sigma)
        return output

class IBOFR_kernel_function(kernel_function):
    def __init__(self, lambda_val=1e-9, gen_mult=1.0, use_Fortran=True, pair_reps=False, density_neglect=1e-9, rep_params=None):
        super().__init__(lambda_val=lambda_val, diag_el_unity=False)
        self.gen_mult=gen_mult
        vec_rep_mult=qml.oml_kernels.ibofr_smoothed_mult(gen_mult, rep_params)
        self.kernel_params=qml.oml_kernels.IBOFR_kernel_params(pair_reps=pair_reps,
                        density_neglect=density_neglect, vec_rep_mult=vec_rep_mult)
    def adjust_hyperparameters(self, compound_array, var_cutoff_val=0.0):
        pass
    def kernel_matrix(self, arr1, arr2):
        return qml.oml_kernels.gen_ibofr_kernel(arr1, arr2, self.kernel_params)
    def sym_kernel_matrix(self, array):
        return qml.oml_kernels.gen_ibofr_kernel(array, array, self.kernel_params, sym_kernel_mat=True)
    def __str__(self):
        output="IBOFR_kernel,gen_mult:"+str(self.gen_mult)
        return output


### END


### Printing and importing learning curve data.
def print_means_errs_to_log_file(x_args, means_errs, filename):
    if filename != None:
        output=open(filename+intermediate_res_ending, "w")
        for i in range(len(x_args)):
            output.write(('{} '+def_float_format+' '+def_float_format+'\n').format(x_args[i], means_errs[0][i], means_errs[1][i]))
        output.close()

def print_full_learning_curve_data(x_args, y_data, filename):
    if filename != None:
        output=open(filename+intermediate_res_ending, "w")
        for curve_number, curve_ys in enumerate(y_data):
            output.write('curve number {}\n'.format(curve_number))
            for i in range(len(x_args)):
                output.write('{} {}\n'.format(x_args[i], curve_ys[i]))
        output.close()

def import_x_args_means_errs(filename):
    file=open(filename, 'r')
    lines=file.readlines()
    x_args=[]
    means=[]
    errs=[]
    for l in lines:
        lsplit=l.split()
        x_args.append(float(lsplit[0]))
        means.append(float(lsplit[1]))
        errs.append(float(lsplit[2]))
    file.close()
    return [x_args, [means, errs]]
### END

### Used to estimate slope/intercept of the learning curves.
def ln_rescale(array_in):
    return jnp.array([math.log(val) for val in array_in])

def ln_lr_model(x_vals, y_vals):
    from sklearn.linear_model import LinearRegression
    x = ln_rescale(x_vals).reshape((-1, 1))
    y = ln_rescale(y_vals)
    return LinearRegression().fit(x, y)

def ln_lr_slope(x_vals, y_vals):
    model=ln_lr_model(x_vals, y_vals)
    return model.coef_[0]
### END

### Routines for handling training array data.
def import_quantity_array(xyz_list, quantity, delta_learning_params=None, disable_openmp=True):
    output=qml.python_parallelization.embarassingly_parallel(import_quantity_val, xyz_list, (quantity, delta_learning_params), disable_openmp=disable_openmp)
    return jnp.array(output)

def import_quantity_val(xyz_file, quantity, delta_learning_params=None):
    added_val=quantity.extract_xyz(xyz_file)
    if delta_learning_params!=None:
        if delta_learning_params.use_delta_learning:
            added_val-=quantity.OML_calc_quant(xyz_file, **delta_learning_params.other_kwargs)
    return added_val
### END

### An auxiliary class for storing parameters for how delta_learning is used.
class Delta_learning_parameters:
    def __init__(self, use_delta_learning=False, **kwargs):
        self.use_delta_learning=use_delta_learning
        self.other_kwargs=kwargs

### Auxiliary procedures for running and analyzing the calculations.
#   Create and shuffle list of xyz files in the directory.
def create_shuffled_xyz_list(QM9_dir, seed=None):
    import random
    list_of_xyzs=dirs_xyz_list(QM9_dir)
    random.Random(seed).shuffle(list_of_xyzs)
    return list_of_xyzs

def dirs_xyz_list(QM9_dir):
    import glob
    output=glob.glob(QM9_dir+"/*.xyz")
    output.sort()
    return output

def another_seed(seed_init, seed_add):
    if seed_init is None:
        return None
    else:
        return seed_init+seed_add
def file_add_int_param(filename, seed_add):
    if filename is None:
        return None
    else:
        return filename+"_"+str(seed_add)
#   Transform several learning curves into arrays of corresponding averages and statistical errors.
def regroup_curves_into_means_and_errors(data_in):
    all_data=jnp.array(data_in).T
    means=[]
    errs=[]
    for diff_seed_data in all_data:
        means.append(jnp.mean(diff_seed_data))
        errs.append(jnp.std(diff_seed_data)/math.sqrt(len(diff_seed_data)))
    return [jnp.array(means), jnp.array(errs)]
### END


def cutout_train_arr(arr_in, training_size):
    return arr_in[:training_size]

def cutout_check_arr(arr_in, check_size, hyperparameter_opt_set=False):
    if hyperparameter_opt_set:
        return arr_in[-2*check_size:-check_size]
    else:
        return arr_in[-check_size:]
### 

def create_training_check_compounds(xyz_list, training_size, check_size, model, hyperparameter_opt_set=False):
    training_xyzs=cutout_train_arr(xyz_list, training_size)
    check_xyzs=cutout_check_arr(xyz_list, check_size, hyperparameter_opt_set=hyperparameter_opt_set)
    training_comp_uninit=model.representation.compound_list(training_xyzs)
    check_comp_uninit=model.representation.compound_list(check_xyzs)
    model.representation.check_param_validity(training_comp_uninit+check_comp_uninit)
    training_compounds=model.representation.init_compound_list(training_comp_uninit)
    check_compounds=model.representation.init_compound_list(check_comp_uninit)
    return training_compounds, check_compounds

def import_train_check_arr(xyz_list, training_size, check_size, quantity, delta_learning_params=None, hyperparameter_opt_set=False):
    training_quants = import_quantity_array(cutout_train_arr(xyz_list, training_size), quantity, delta_learning_params=delta_learning_params)
    check_quants=import_quantity_array(cutout_check_arr(xyz_list, check_size, hyperparameter_opt_set=hyperparameter_opt_set), quantity, delta_learning_params=delta_learning_params)
    return training_quants, check_quants

#   For sanity check of generated kernel matrices.
def check_els_leq_unity(matrix, skip_diag=False):
    for id1, row in enumerate(matrix):
        for id2, el in enumerate(row):
            if (not skip_diag) or (id1 != id2):
                if el > 1.0:
                    print("#TEST An element of the kernel matrix exceeds unity: ", el)
                    return
    print("#TEST All kernel matrix elements leq 1.")


class logfile:
    def __init__(self, base_name=None, ending=None):
        self.not_empty=(base_name is not None)
        if self.not_empty:
            full_filename=base_name
            if ending is not None:
                full_filename+=ending
            self.output=open(full_filename, 'w')
    def write(self, *args):
        if self.not_empty:
            print(*args, file=self.output)
    def close(self):
        if self.not_empty:
            self.output.close()
    def export_quantity_array(self, quantity_array):
        if self.not_empty:
            for quant_val in quantity_array:
                self.output.write((def_float_format+'\n').format(quant_val))
    def export_matrix(self, matrix_in):
        if self.not_empty:
            for id1, row in enumerate(matrix_in):
                for id2, val in enumerate(row):
                    self.output.write(('{} {} '+def_float_format+'\n').format(id1, id2, val))


#   Calculate MAE using first training_size entries of xyz_list and last check_size entries of xyz_list.
def calculate_MAE(xyz_list, training_size, check_size, quantity, model, delta_learning_params=None, calc_logfile=logfile(None, None), quant_logfile=logfile(None, None),
                    quantity_train_array=None, quantity_check_array=None, compound_train_array=None, compound_check_array=None,
                    hyperparameter_opt_set=False, eigh_rcond=1e-9):
    from qml.math import cho_solve

    if (compound_train_array is None) or (compound_check_array is None):
        training_compounds, check_compounds=create_training_check_compounds(xyz_list, training_size, check_size, model, hyperparameter_opt_set=hyperparameter_opt_set)
    else:
        training_compounds=cutout_train_arr(compound_train_array, training_size)
        check_compounds=compound_check_array #cutout_check_arr(compound_check_array, check_size, hyperparameter_opt_set=hyperparameter_opt_set)

    if (quantity_train_array is None) or (quantity_check_array is None):
        training_quants, check_quants=import_train_check_arr(xyz_list, training_size, check_size, quantity,
                                        delta_learning_params=delta_learning_params, hyperparameter_opt_set=hyperparameter_opt_set)
    else:
        training_quants=cutout_train_arr(quantity_train_array, training_size)
        check_quants=quantity_check_array #cutout_check_arr(quantity_check_array, check_size)

    for log in [calc_logfile, quant_logfile]:
        log.write("Training size: ", training_size, "Model: ", model)
    K=model.kernel_function.km_added_lambda(training_compounds)
    calc_logfile.write("Determinant of K: ", jnp.linalg.det(K))
    calc_logfile.write("Asymmetry measure of K: ", asymmetry_measure(K))
    alpha=np_cho_solve_wcheck(K, training_quants, eigh_rcond=eigh_rcond)
#    alpha = cho_solve(K, training_quants)
#    alpha = jnp.linalg.solve(K, training_quants)
    quant_logfile.write("Training K:")
    quant_logfile.export_matrix(K)
    del(K)
    Ks = model.kernel_function.kernel_matrix(check_compounds, training_compounds)
    #check_els_leq_unity(Ks)
    check_predictions=jnp.dot(Ks, alpha)



    quant_logfile.write("Check kernel:")
    quant_logfile.export_matrix(Ks)
    quant_logfile.write("Training quantities:")
    quant_logfile.export_quantity_array(training_quants)
    quant_logfile.write("Check quantities:")
    quant_logfile.export_quantity_array(training_quants)

    MAE=jnp.mean(jnp.abs(check_predictions - check_quants))
    calc_logfile.write("Model:", model, ", training size: ", training_size, "MAE: ", MAE)
    return MAE

def asymmetry_measure(K):
    return ((K-K.T)**2).sum()

#   TO-DO Cycle over model or training sizes depending on which one is a list?
def make_learning_curve_data(quantity, training_sizes, check_size, model, QM9_dir, seed=0, delta_learning_params=None, calc_log=None, quant_log=None):

    print("Making learning curve, seed: ", seed)
    list_of_xyzs=create_shuffled_xyz_list(QM9_dir, seed=seed)
    compound_train_array, compound_check_array=create_training_check_compounds(list_of_xyzs, max(training_sizes), check_size, model)

    calc_logfile=logfile(calc_log, '.log')
    quant_logfile=logfile(quant_log, '.log')
    output_array=[ calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params,
            calc_logfile=calc_logfile, quant_logfile=quant_logfile, compound_train_array=compound_train_array, compound_check_array=compound_check_array)
            for training_size in training_sizes ]

    return output_array

def make_model_param_scan_curve(quantity, training_size, check_size, scanned_models, QM9_dir, seed=None, delta_learning_params=None, calc_log=None, quant_log=None):
    hyperparameter_opt_set=True
    print("Making model parameter scan, seed: ", seed)
    list_of_xyzs=create_shuffled_xyz_list(QM9_dir, seed=seed)
    compound_train_array, compound_check_array=create_training_check_compounds(list_of_xyzs, training_size, check_size, scanned_models[0], hyperparameter_opt_set=hyperparameter_opt_set)
    output_array=[]
    calc_logfile=logfile(calc_log, '.log')
    quant_logfile=logfile(quant_log, '.log')
    for model in scanned_models:
#        cur_train_array=model.adjust_representation(compound_train_array)
#        cur_check_array=model.adjust_representation(compound_check_array)
#        output_array.append(calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params, calc_logfile=calc_logfile,
#                            quant_logfile=quant_logfile, compound_train_array=cur_train_array, compound_check_array=cur_check_array, hyperparameter_opt_set=hyperparameter_opt_set))
#        output_array.append(calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params, calc_logfile=calc_logfile,
#                            quant_logfile=quant_logfile, hyperparameter_opt_set=hyperparameter_opt_set))
        output_array.append(calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params, calc_logfile=calc_logfile,
                            quant_logfile=quant_logfile, compound_train_array=compound_train_array, compound_check_array=compound_check_array, hyperparameter_opt_set=hyperparameter_opt_set))
    return output_array

def make_learning_curves_with_stdev(quantity, training_sizes, check_size,  model, QM9_dir, num_iters=1, seed=None, output_file=None, delta_learning_params=None, calc_logs=None, quant_logs=None):
    all_data=[ make_learning_curve_data(quantity, training_sizes, check_size, model, QM9_dir, seed=another_seed(seed, seed_add),
               delta_learning_params=delta_learning_params, calc_log=file_add_int_param(calc_logs, seed_add), quant_log=file_add_int_param(quant_logs, seed_add)) for seed_add in range(num_iters) ]
    final_data=regroup_curves_into_means_and_errors(all_data)
    if output_file != None:
        print_means_errs_to_log_file(training_sizes, final_data, output_file+"_means_errs")
        print_full_learning_curve_data(training_sizes, all_data, output_file+"_full_data")
    return final_data

def best_model_params(quantity, training_size, check_size, models, QM9_dir, num_iters=1, seed=None, output_file=None, delta_learning_params=None, calc_logs=None, quant_logs=None):
    all_data=[ make_model_param_scan_curve(quantity, training_size, check_size, models, QM9_dir, seed=another_seed(seed, seed_add),
            delta_learning_params=delta_learning_params, quant_log=file_add_int_param(quant_logs, seed_add), calc_log=file_add_int_param(calc_logs, seed_add)) for seed_add in range(num_iters)]
    final_data=regroup_curves_into_means_and_errors(all_data)
    if output_file != None:
        print_means_errs_to_log_file(models, final_data, output_file+"_means_errs")
        print_full_learning_curve_data(models, all_data, output_file+"_full_data")
    return models[jnp.argmin(final_data[0])]
    
### These functions appear when generating model parameters to be scanned (for example with the best_model_params procedure).
def geom_progression(start_val, multiplier, num_vals):
    output=[]
    for val_id in range(num_vals):
        output.append(start_val*multiplier**val_id)
    return output

def linear_interpolation_points(start_val, end_val, num_vals):
    output=[]
    for val_id in range(num_vals):
        output.append(start_val+val_id*(end_val-start_val)/(num_vals-1))
    return output

### END

