
import qml
import jax.numpy as jnp
import math
from qm9_format_specs import Quantity

byprod_result_ending=".brf"

intermediate_res_ending=".dat"

### Auxiliary functions to use in the representation class.
# Appears, for example, in CM representation.
def find_max_size(compound_list):
    output=0
    for mol in compound_list:
        output=max(output, len(mol.atomtypes))
    return output

### END

class KR_model:
    def __init__(self, kernel_function=None, representation=None):
        self.kernel_function=kernel_function
        self.representation=representation
    def __str__(self):
        return str(self.kernel_function)+";"+str(self.representation)
    def adjust_hyperparameters(self, xyz_array):
        init_compound_array=self.representation.init_compound_list(xyz_list=xyz_array)
        self.kernel_function.adjust_hyperparameters(init_compound_array)
    def adjust_representation(self, compound_array):
        return self.representation.adjust_representation(compound_array)

### A family of classes corresponding to different representations.
class representation:
    def check_param_validity(self, compound_array_in):
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
    def check_param_validity(self, compound_array_in):
        self.max_size=max(self.max_size, find_max_size(compound_array_in))
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_coulomb_matrix(size=self.max_size, sorting=self.sorting)
        return comp
    def __str__(self):
        return "CM_rep,sorting:"+self.sorting
        

class OML_representation(representation):
    def __init__(self, ibo_atom_rho_comp=None, max_angular_momentum=3, use_Fortran=True,
                    fock_based_coup_mat=False, num_fbcm_omegas=2):
        import qml.oml_representations
        self.rep_params=qml.oml_representations.OML_rep_params(ibo_atom_rho_comp=ibo_atom_rho_comp, max_angular_momentum=max_angular_momentum,
                                                                        use_Fortran=use_Fortran, fock_based_coup_mat=fock_based_coup_mat,
                                                                        num_fbcm_omegas=num_fbcm_omegas)
    def xyz2compound(self, xyz=None):
        import qml.oml_compound
        return qml.oml_compound.OML_compound(xyz = xyz, mats_savefile = xyz)
    def compound_list(self, xyz_list):
        return qml.OML_compound_list_from_xyzs(xyz_list)
    def initialized_compound(self, compound=None, xyz = None):
        comp=self.check_compound_defined(compound, xyz)
        comp.generate_orb_reps(self.rep_params)
        return comp
    def init_compound_list(self, comp_list=None, xyz_list=None, disable_openmp=True):
        import pickle, subprocess
        if comp_list is not None:
            new_list=comp_list
        else:
            new_list=self.compound_list(xyz_list)
        if disable_openmp:
            temp_dat_name="tmp_arr.tmp"
            temp_rep_name="tmp_rep.tmp"
            plf=open(temp_dat_name, 'wb')
            pickle.dump(new_list, plf)
            plf.close()

            plf=open(temp_rep_name, 'wb')
            pickle.dump(self.rep_params, plf)
            plf.close()
            subprocess.run(["oml_gen_orb_reps.sh", temp_dat_name, temp_rep_name])
            plf=open(temp_dat_name, 'rb')
            new_list=pickle.load(plf)
            plf.close()
        else:
            new_list.generate_orb_reps(self.rep_params)
        return new_list
    def __str__(self):
        return "OML_rep,"+str(self.rep_params)
### END

### Auxiliary functions to use in kernel_function class.
# Combine all representation arrays into one array.
def combined_representation_arrays(compound_list):
    return jnp.array([mol.representation for mol in compound_list])
### END

### A family of classes corresponding to different kernel functions.

class kernel_function:
    def __init__(self, lambda_val):
        self.lambda_val=lambda_val
    def km_added_lambda(self, arr1, arr2):
        K=self.kernel_matrix(arr1,arr2)
        self.print_diag_deviation(K)
        # TO-DO: Check whether this is necessary:
        K_size=len(arr1)
        for i1 in range(K_size):
            for i2 in range(i1):
                true_K=(K[i1,i2]+K[i2,i1])/2
                K[i1,i2]=true_K
                K[i2,i1]=true_K
        K[jnp.diag_indices_from(K)] = 1.0+self.lambda_val
        return K
    def adjust_hyperparameters(self, compound_array):
        pass
    def print_diag_deviation(self, matrix):
        print("#TEST Total deviation of diagonal elements from 1: ", sum(abs(el-1.0) for el in matrix[jnp.diag_indices_from(matrix)]))

class Gaussian_kernel_function(kernel_function):
    def __init__(self, sigma, lambda_val=1e-9):
        self.sigma=sigma
        self.lambda_val=lambda_val
    def kernel_matrix(self, arr1, arr2):
        from qml.kernels import gaussian_kernel
        rep_arr1=combined_representation_arrays(arr1)
        rep_arr2=combined_representation_arrays(arr2)
        return gaussian_kernel(rep_arr1, rep_arr2, self.sigma)
    def __str__(self):
        return "sigma:"+str(self.sigma)

class OML_GMO_kernel_function(kernel_function):
    def __init__(self, lambda_val=1e-9, final_sigma=1.0, sigma_rescale=1.0, use_Fortran=True):
        from qml.oml_kernels import GMO_kernel_params
        self.kernel_params=GMO_kernel_params(final_sigma=final_sigma, use_Fortran=use_Fortran)
        self.lambda_val=lambda_val
        self.sigma_rescale=sigma_rescale
    def adjust_hyperparameters(self, compound_array):
        from qml.oml_kernels import oml_ensemble_widths_estimate
        orb_sample=qml.oml_kernels.random_ibo_sample(compound_array)
        self.kernel_params.update_width(oml_ensemble_widths_estimate(orb_sample)/self.sigma_rescale)
    def kernel_matrix(self, arr1, arr2):
        from qml.oml_kernels import generate_GMO_kernel
        return generate_GMO_kernel(arr1, arr2, self.kernel_params)
    def __str__(self):
        return "GMO,sigma_rescale:"+str(self.sigma_rescale)+",fin_sigma:"+str(self.kernel_params.final_sigma)


### END


### Printing and importing learning curve data.
def print_means_errs_to_log_file(x_args, means_errs, filename):
    if filename != None:
        output=open(filename+intermediate_res_ending, "w")
        for i in range(len(x_args)):
            output.write('{} {} {}\n'.format(x_args[i], means_errs[0][i], means_errs[1][i]))
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
def import_quantity_array(xyz_list, quantity, delta_learning_params=None):
    output=[]
    for f in xyz_list:
        added_val=quantity.extract_xyz(f)
        if delta_learning_params!=None:
            if delta_learning_params.use_delta_learning:
                comp=qml.oml_compound.OML_compound(xyz = f, mats_savefile = f, calc_type=delta_learning_params.calc_type)
                comp.run_calcs()
                added_val-=quantity.OML_comp_extract_byprod(comp)
        output.append(added_val)
    return jnp.array(output)

def import_byprod_results(xyz_list, quantity, calc_type="IBO_HF_min_bas"):
    return jnp.array([quantity.extract_byprod_result(xyz_name2byprod_filename(f, calc_type)) for f in xyz_list])


### END

### An auxiliary class for storing parameters for how delta_learning is used.
class Delta_learning_parameters:
    def __init__(self, use_delta_learning=False, calc_type="IBO_HF_min_bas"):
        self.use_delta_learning=use_delta_learning
        self.calc_type=calc_type

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

### For writing byproduct results of ab initio calculations.

def xyz_name2byprod_filename(filename, calc_type):
    if filename.endswith(".npz"):
        return filename[:-4]+byprod_result_ending
    else:
        return filename+'.'+calc_type+byprod_result_ending

def write_byprod_line(output_file, val, quant_name):
    Quantity(quant_name).write_byprod_result(val, output_file)

def create_byproduct_file(oml_compound_in, filename):
    output=open(filename, "w")
    write_byprod_line(output, oml_compound_in.HOMO_en(), 'HOMO eigenvalue')
    write_byprod_line(output, oml_compound_in.LUMO_en(), 'LUMO eigenvalue')
    output.close()

def create_byproduct_files(oml_compound_list):
    for comp in oml_compound_list:
        filename=comp.mats_savefile[:-4]+byprod_result_ending # remove "npz" suffix and replace with byproduct file suffix.
        comp.run_calcs()
        create_byproduct_file(comp, filename)
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
    model.representation.check_param_validity(training_comp_uninit)
    model.representation.check_param_validity(check_comp_uninit)
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
                self.output.write('{}\n'.format(quant_val))
    def export_matrix(self, matrix_in):
        if self.not_empty:
            for id1, row in enumerate(matrix_in):
                for id2, val in enumerate(row):
                    self.output.write('{} {} {}\n'.format(id1, id2, val))


#   Calculate MAE using first training_size entries of xyz_list and last check_size entries of xyz_list.
def calculate_MAE(xyz_list, training_size, check_size, quantity, model, delta_learning_params=None, calc_logfile=logfile(None, None), quant_logfile=logfile(None, None),
                    quantity_train_array=None, quantity_check_array=None, compound_train_array=None, compound_check_array=None,
                    hyperparameter_opt_set=False):
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
        training_quants = cutout_train_arr(quantity_train_array, training_size)
        check_quants =quantity_check_array #cutout_check_arr(quantity_check_array, check_size)

    for log in [calc_logfile, quant_logfile]:
        log.write("Training size: ", training_size, "Model: ", model)
    K=model.kernel_function.km_added_lambda(training_compounds, training_compounds)
    calc_logfile.write("Determinant of K: ", jnp.linalg.det(K))
    calc_logfile.write("Asymmetry measure of K: ", asymmetry_measure(K))
    alpha = cho_solve(K, training_quants)
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
    output_array=[ calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params, calc_logfile=calc_logfile, quant_logfile=quant_logfile, compound_train_array=compound_train_array, compound_check_array=compound_check_array)
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
        output_array.append(calculate_MAE(list_of_xyzs, training_size, check_size, quantity, model, delta_learning_params=delta_learning_params, calc_logfile=calc_logfile,
                            quant_logfile=quant_logfile, hyperparameter_opt_set=hyperparameter_opt_set))
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

