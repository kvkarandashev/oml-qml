from numba import njit, prange
import numpy as np
import itertools

def iterated_orb_reps(oml_comp, pair_reps=False, single_ibo_list=False):
    if pair_reps:
        return itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps)
    else:
        if single_ibo_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps


def orb_rep_rho_list(oml_comp, pair_reps=False):
    output=[]
    for ibo in iterated_orb_reps(oml_comp, pair_reps=pair_reps):
        output.append([ibo.rho, ibo])
    if pair_reps:
        for i in range(len(oml_comp.comps[0].orb_reps)):
            output[i][0]*=-1
    return output


class GMO_sep_IBO_kern_input:
    def __init__(self, oml_compound_array=None, pair_reps=None, single_virtual_ibo=False):
        if pair_reps is None:
            pair_reps=is_pair_reps(oml_compound_array)
        if pair_reps:
            self.max_num_scalar_reps=len(oml_compound_array[0].comps[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)
        else:
            self.max_num_scalar_reps=len(oml_compound_array[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)

        self.num_mols=len(oml_compound_array)
        self.ibo_nums=np.zeros((self.num_mols,), dtype=int)
        self.max_num_ibos=0
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            self.ibo_nums[comp_id]=len(rho_orb_list)
        self.max_num_ibos=max(self.ibo_nums)
        self.ibo_atom_nums=np.zeros((self.num_mols, self.max_num_ibos), dtype=int)
        self.ibo_rhos=np.zeros((self.num_mols, self.max_num_ibos))
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            for orb_id, [orb_rho, orb_rep] in enumerate(rho_orb_list):
                self.ibo_atom_nums[comp_id, orb_id]=len(orb_rep.ibo_atom_reps)
                self.ibo_rhos[comp_id, orb_id]=orb_rho

        self.max_num_ibo_atom_reps=np.amax(self.ibo_atom_nums)

        self.ibo_arep_rhos=np.zeros((self.num_mols, self.max_num_ibos, self.max_num_ibo_atom_reps))
        self.ibo_atom_sreps=np.zeros((self.num_mols, self.max_num_ibos, self.max_num_ibo_atom_reps, self.max_num_scalar_reps))
        for ind_comp, oml_comp in enumerate(oml_compound_array):
            for ind_ibo, [ibo_rho, ibo_rep] in enumerate(orb_rep_rho_list(oml_comp, pair_reps=pair_reps)):
                for ind_ibo_arep, ibo_arep in enumerate(ibo_rep.ibo_atom_reps):
                    self.ibo_arep_rhos[ind_comp, ind_ibo, ind_ibo_arep]=ibo_arep.rho
                    self.ibo_atom_sreps[ind_comp, ind_ibo, ind_ibo_arep, :]=ibo_arep.scalar_reps[:]
        self.ibo_atom_scaled_sreps=None

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def width_rescaling(ibo_atom_sreps, ibo_nums, ibo_atom_nums, sigmas):
        ibo_atom_scaled_sreps=np.copy(ibo_atom_sreps)
        num_mols=ibo_atom_scaled_sreps.shape[0]
        for mol_id in prange(num_mols):
            for ibo_id in range(ibo_nums[mol_id]):
                for arep_id in range(ibo_atom_nums[mol_id, ibo_id]):
                    ibo_atom_scaled_sreps[mol_id, ibo_id, arep_id, :]/=2*sigmas
        return ibo_atom_scaled_sreps

    # For renormalizing orbital arep rho coefficients.
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def numba_lin_sep_kern_renormalized_arep_rhos(ibo_atom_scaled_sreps, ibo_arep_rhos, ibo_nums, ibo_atom_nums):
        num_mols=ibo_atom_scaled_sreps.shape[0]

        output=np.copy(ibo_arep_rhos)
        for mol_id in prange(num_mols):
            for ibo_id in range(ibo_nums[mol_id]):
                output[mol_id, ibo_id, :]/=orb_sqrt_self_cov(ibo_atom_scaled_sreps[mol_id, ibo_id, :, :], ibo_arep_rhos[mol_id, ibo_id, :], ibo_atom_nums[mol_id, ibo_id])
        return output

    def rescale_reps(self, sigmas):
        self.ibo_atom_scaled_sreps=self.width_rescaling(self.ibo_atom_sreps, self.ibo_nums, self.ibo_atom_nums, sigmas)


    def lin_sep_kern_renormalize_arep_rhos(self, sigmas):

        self.rescale_reps(sigmas)
        self.ibo_arep_rhos=self.numba_lin_sep_kern_renormalized_arep_rhos(self.ibo_atom_scaled_sreps, self.ibo_arep_rhos, self.ibo_nums, self.ibo_atom_nums)

### For linear kernel with separable IBOs.

# Orbital-orbital covariance.
@njit(fastmath=True)
def orb_areps_sqdiffs(orb_areps_A, arep_A_id, orb_areps_B, arep_B_id):
    return (orb_areps_A[arep_A_id, :]-orb_areps_B[arep_B_id, :])**2

@njit(fastmath=True)
def orb_orb_cov(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B):
    output=0.0
    for arep_A_id in range(num_ibo_areps_A):
        rho_A=arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_ibo_areps_B):
            rho_B=arep_rhos_B[arep_B_id]
            sqdiffs=orb_areps_sqdiffs(orb_areps_A, arep_A_id, orb_areps_B, arep_B_id)
            orb_orb_cov_comp=rho_A*rho_B*np.exp(-np.sum(sqdiffs))
            output+=orb_orb_cov_comp
    return output

@njit(fastmath=True)
def orb_orb_cov_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, orb_comp_dim):
    output=np.zeros((orb_comp_dim,))
    for arep_A_id in range(num_ibo_areps_A):
        rho_A=arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_ibo_areps_B):
            rho_B=arep_rhos_B[arep_B_id]
            sqdiffs=orb_areps_sqdiffs(orb_areps_A, arep_A_id, orb_areps_B, arep_B_id)
            orb_orb_cov_comp=rho_A*rho_B*np.exp(-np.sum(sqdiffs))
            output[0]+=orb_orb_cov_comp
            output[1:]+=orb_orb_cov_comp*sqdiffs
    return output

@njit(fastmath=True)
def orb_orb_cov_wders_log_incl(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, orb_comp_dim, A_sp_log_ders, B_sp_log_ders):
    output=orb_orb_cov_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, orb_comp_dim)
    output[1:]-=output[0]*(A_sp_log_ders+B_sp_log_ders)/2
    return output


# Self-covariance (used for normalizing the orbitals).
@njit(fastmath=True)
def orb_sqrt_self_cov(orb_areps, arep_rhos, num_ibo_areps):
    return np.sqrt(orb_orb_cov(orb_areps, orb_areps, arep_rhos, arep_rhos, num_ibo_areps, num_ibo_areps))


@njit(fastmath=True)
def lin_sep_IBO_kernel_row(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_num, A_ibo_atom_nums, upper_B_mol_id, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums,
                                B_ibo_atom_nums):


    kernel_row=np.zeros((B_num_mols,))

    for B_mol_id in range(upper_B_mol_id):
        for A_ibo_id in range(A_ibo_num):
            rho_A=A_orb_rhos[A_ibo_id]
            for B_ibo_id in range(B_ibo_nums[B_mol_id]):
                rho_B=B_orb_rhos[B_mol_id, B_ibo_id]
                cur_ibo_contribution=orb_orb_cov(A_orb_areps[A_ibo_id, :, :], B_orb_areps[B_mol_id, B_ibo_id, :, :], A_arep_rhos[A_ibo_id, :],
                                           B_arep_rhos[B_mol_id, B_ibo_id, :], A_ibo_atom_nums[A_ibo_id], B_ibo_atom_nums[B_mol_id, B_ibo_id])
                kernel_row[B_mol_id]+=rho_A*rho_B*cur_ibo_contribution

    return kernel_row



@njit(fastmath=True)
def numba_lin_sep_IBO_kernel_row_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_num, A_ibo_atom_nums, upper_B_mol_id, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, kern_comp_dim,
                                A_sp_log_ders, B_sp_log_ders):

    cur_ibo_contribution=np.zeros((kern_comp_dim,))

    kernel_row=np.zeros((B_num_mols,kern_comp_dim))

    for B_mol_id in range(upper_B_mol_id):
        for A_ibo_id in range(A_ibo_num):
            rho_A=A_orb_rhos[A_ibo_id]
            for B_ibo_id in range(B_ibo_nums[B_mol_id]):
                rho_B=B_orb_rhos[B_mol_id, B_ibo_id]
                cur_ibo_contribution=orb_orb_cov_wders_log_incl(A_orb_areps[A_ibo_id, :, :], B_orb_areps[B_mol_id, B_ibo_id, :, :], A_arep_rhos[A_ibo_id, :], B_arep_rhos[B_mol_id, B_ibo_id, :],
                                                A_ibo_atom_nums[A_ibo_id], B_ibo_atom_nums[B_mol_id, B_ibo_id], kern_comp_dim,
                                                A_sp_log_ders[A_ibo_id, :], B_sp_log_ders[B_mol_id, B_ibo_id, :])
                kernel_row[B_mol_id,:]+=rho_A*rho_B*cur_ibo_contribution

    return kernel_row


@njit(fastmath=True, parallel=True)
def numba_lin_sep_IBO_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    Kernel=np.zeros((A_num_mols, B_num_mols))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :]=numba_lin_sep_IBO_kernel_row(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id,:, :],
                                A_orb_rhos[A_mol_id, :], A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], B_num_mols, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums, B_ibo_atom_nums)
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_lin_sep_IBO_sym_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums):

    A_num_mols=A_orb_areps.shape[0]
    Kernel=np.zeros((A_num_mols, A_num_mols))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :]=numba_lin_sep_IBO_kernel_row(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id,:, :],
                                A_orb_rhos[A_mol_id, :], A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], A_mol_id+1, A_num_mols,
                                A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums)

    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id]=Kernel[A_mol_id, A_mol_id2]
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_lin_sep_IBO_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    A_sp_log_ders=self_product_log_ders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums)
    B_sp_log_ders=self_product_log_ders(B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums, B_ibo_atom_nums)

    kern_comp_dim=A_orb_areps.shape[-1]+1
    Kernel=np.zeros((A_num_mols, B_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :, :]=numba_lin_sep_IBO_kernel_row_wders(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id, :, :], A_orb_rhos[A_mol_id, :],
                                A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], B_num_mols, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, kern_comp_dim,
                                A_sp_log_ders[A_mol_id, :, :], B_sp_log_ders)
        Kernel[A_mol_id, :, :]*=kern_der_resc
    return Kernel


@njit(fastmath=True, parallel=True)
def numba_lin_sep_IBO_sym_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    A_sp_log_ders=self_product_log_ders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums)

    kern_comp_dim=A_orb_areps.shape[-1]+1
    Kernel=np.zeros((A_num_mols, A_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :, :]=numba_lin_sep_IBO_kernel_row_wders(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id, :, :], A_orb_rhos[A_mol_id, :],
                                A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], A_mol_id+1, A_num_mols,
                                A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums, kern_comp_dim,
                                A_sp_log_ders[A_mol_id, :, :], A_sp_log_ders)
        Kernel[A_mol_id, :A_mol_id+1, :]*=kern_der_resc
    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id, :]=Kernel[A_mol_id, A_mol_id2, :]

    return Kernel



def lin_sep_IBO_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=True, with_ders=False):
    if preserve_converted_arrays:
        Ac_renormed=copy.deepcopy(Ac)
        Bc_renormed=copy.deepcopy(Bc)
    else:
        Ac_renormed=Ac
        Bc_renormed=Bc
    Ac_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas)
    sym_kernel=(Bc is None)
    if not sym_kernel:
        Bc_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas)
    if with_ders:
        kern_der_resc=1.0/sigmas**3 # TO-DO test properly
        if sym_kernel:
            output=numba_lin_sep_IBO_sym_kernel_wders(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums, kern_der_resc)
        else:
            output=numba_lin_sep_IBO_kernel_wders(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums,
                                Bc_renormed.ibo_atom_scaled_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos,
                                Bc_renormed.ibo_nums, Bc_renormed.ibo_atom_nums, kern_der_resc)
    else:
        if sym_kernel:
            return numba_lin_sep_IBO_sym_kernel(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums)
        else:
            return numba_lin_sep_IBO_kernel(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums,
                                Bc_renormed.ibo_atom_scaled_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos,
                                Bc_renormed.ibo_nums, Bc_renormed.ibo_atom_nums)




def is_pair_reps(comp_arr):
    return (hasattr(comp_arr[0], 'comps'))

def lin_sep_IBO_kernel(A, B, sigmas, with_ders=False):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A)
    Bc=GMO_sep_IBO_kern_input(oml_compound_array=B)
    return lin_sep_IBO_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

def lin_sep_IBO_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=True, with_ders=False):
    return lin_sep_IBO_kernel_conv(Ac, None, sigmas, preserve_converted_arrays=preserve_converted_arrays, with_ders=with_ders)

def lin_sep_IBO_sym_kernel(A, sigmas, with_ders=False):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A)
    return lin_sep_IBO_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

### Generate log derivatives.
@njit(fastmath=True)
def orb_self_cov_wders(orb_areps, arep_rhos, ibo_atom_nums, orb_comp_dim):
    return orb_orb_cov_wders(orb_areps, orb_areps, arep_rhos, arep_rhos, ibo_atom_nums, ibo_atom_nums, orb_comp_dim)

@njit(fastmath=True, parallel=True)
def self_product_log_ders(orb_areps, arep_rhos, ibo_rhos, ibo_nums, ibo_atom_nums):
    num_mols=orb_areps.shape[0]
    max_num_ibos=orb_areps.shape[1]

    num_ders=orb_areps.shape[-1]
    orb_comp_dim=num_ders+1

    log_ders=np.zeros((num_mols, max_num_ibos, num_ders))

    cur_prod_self_der=np.zeros((orb_comp_dim,))
    for mol_id in prange(num_mols):
        for ibo_id in range(ibo_nums[mol_id]):
            cur_prod_self_der=orb_self_cov_wders(orb_areps[mol_id, ibo_id, :, :], arep_rhos[mol_id, ibo_id, :], ibo_atom_nums[mol_id, ibo_id], orb_comp_dim)
            log_ders[mol_id, ibo_id, :]=cur_prod_self_der[1:]/cur_prod_self_der[0]
    return log_ders


@njit(fastmath=True)
def numba_find_ibo_vec_rep_moments(ibo_atom_sreps, ibo_arep_rhos, ibo_rhos, ibo_nums, ibo_atom_nums, moment):
    num_mols=ibo_arep_rhos.shape[0]

    output=np.zeros(ibo_atom_sreps.shape[3])
    norm_const=0.0
    for mol_id in range(num_mols):
        for ibo_id in range(ibo_nums[mol_id]):
            rho_ibo=ibo_rhos[mol_id, ibo_id]
            for arep_id in range(ibo_atom_nums[mol_id, ibo_id]):
                rho_arep=ibo_arep_rhos[mol_id, ibo_id, arep_id]
                cur_rho=np.abs(rho_arep*rho_ibo)
                norm_const+=cur_rho
                output+=cur_rho*ibo_atom_sreps[mol_id, ibo_id, arep_id, :]**moment
    return output/norm_const

def find_ibo_vec_rep_moments(compound_list_converted, moment):
    return numba_find_ibo_vec_rep_moments(compound_list_converted.ibo_atom_sreps, compound_list_converted.ibo_arep_rhos, compound_list_converted.ibo_rhos,
                        compound_list_converted.ibo_nums, compound_list_converted.ibo_atom_nums, moment)

def oml_ensemble_avs_stddevs(compound_list):
    compound_list_converted=GMO_sep_IBO_kern_input(compound_list)

    avs=find_ibo_vec_rep_moments(compound_list_converted, 1)
    avs2=find_ibo_vec_rep_moments(compound_list_converted, 2)
    stddevs=np.sqrt(avs2-avs**2)
    return avs, stddevs


############
# For the Gaussian "sep IBO" kernel.

@njit(fastmath=True)
def lin2gauss_kern_el(lin_cov, inv_sq_global_sigma):
    return np.exp(-(1.0-lin_cov)*inv_sq_global_sigma)

@njit(fastmath=True)
def orb_orb_cov_gauss(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, inv_sq_global_sigma):
    lin_cov=orb_orb_cov(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B)
    return lin2gauss_kern_el(lin_cov, inv_sq_global_sigma)

@njit(fastmath=True)
def orb_orb_cov_gauss_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, inv_sq_global_sigma, orb_comp_dim, A_log_ders, B_log_ders):
    lin_cov_wders=orb_orb_cov_wders_log_incl(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_ibo_areps_A, num_ibo_areps_B, orb_comp_dim-1, A_log_ders, B_log_ders)
    
    output=np.zeros((orb_comp_dim,))
    output[0]=lin2gauss_kern_el(lin_cov_wders[0], inv_sq_global_sigma)
    output[1]=output[0]*(1-lin_cov_wders[0])
    output[2:]=output[0]*lin_cov_wders[1:]
    return output

@njit(fastmath=True)
def gauss_sep_IBO_kernel_row(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_num, A_ibo_atom_nums, upper_B_mol_id, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums, B_ibo_atom_nums,
                                inv_sq_global_sigma):


    kernel_row=np.zeros((B_num_mols,))

    for B_mol_id in range(upper_B_mol_id):
        for A_ibo_id in range(A_ibo_num):
            rho_A=A_orb_rhos[A_ibo_id]
            for B_ibo_id in range(B_ibo_nums[B_mol_id]):
                rho_B=B_orb_rhos[B_mol_id, B_ibo_id]
                cur_ibo_contribution=orb_orb_cov_gauss(A_orb_areps[A_ibo_id, :, :], B_orb_areps[B_mol_id, B_ibo_id, :, :], A_arep_rhos[A_ibo_id, :], B_arep_rhos[B_mol_id, B_ibo_id, :],
                                                A_ibo_atom_nums[A_ibo_id], B_ibo_atom_nums[B_mol_id, B_ibo_id], inv_sq_global_sigma)
                kernel_row[B_mol_id]+=rho_A*rho_B*cur_ibo_contribution
    return kernel_row

@njit(fastmath=True)
def gauss_sep_IBO_kernel_row_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_num, A_ibo_atom_nums, upper_B_mol_id, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, inv_sq_global_sigma,
                                kern_comp_dim, A_sp_log_ders, B_sp_log_ders):

    cur_ibo_contribution=np.zeros((kern_comp_dim,))

    kernel_row=np.zeros((B_num_mols,kern_comp_dim))

    for B_mol_id in range(upper_B_mol_id):
        for A_ibo_id in range(A_ibo_num):
            rho_A=A_orb_rhos[A_ibo_id]
            for B_ibo_id in range(B_ibo_nums[B_mol_id]):
                rho_B=B_orb_rhos[B_mol_id, B_ibo_id]
                cur_ibo_contribution=orb_orb_cov_gauss_wders(A_orb_areps[A_ibo_id, :, :], B_orb_areps[B_mol_id, B_ibo_id, :, :], A_arep_rhos[A_ibo_id, :], B_arep_rhos[B_mol_id, B_ibo_id, :],
                                                A_ibo_atom_nums[A_ibo_id], B_ibo_atom_nums[B_mol_id, B_ibo_id], inv_sq_global_sigma, kern_comp_dim,
                                                A_sp_log_ders[A_ibo_id, :], B_sp_log_ders[B_mol_id, B_ibo_id, :])
                kernel_row[B_mol_id,:]+=rho_A*rho_B*cur_ibo_contribution

    return kernel_row




@njit(fastmath=True, parallel=True)
def numba_gauss_sep_IBO_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, inv_sq_global_sigma):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    Kernel=np.zeros((A_num_mols, B_num_mols))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :]=gauss_sep_IBO_kernel_row(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id,:, :],
                                A_orb_rhos[A_mol_id, :], A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], B_num_mols, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums, B_ibo_atom_nums, inv_sq_global_sigma)
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_gauss_sep_IBO_sym_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums, inv_sq_global_sigma):

    A_num_mols=A_orb_areps.shape[0]
    Kernel=np.zeros((A_num_mols, A_num_mols))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :]=gauss_sep_IBO_kernel_row(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id,:, :],
                                A_orb_rhos[A_mol_id, :], A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], A_mol_id+1, A_num_mols,
                                A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums, inv_sq_global_sigma)

    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id]=Kernel[A_mol_id, A_mol_id2]
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_gauss_sep_IBO_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, inv_sq_global_sigma,
                                kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    A_sp_log_ders=self_product_log_ders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums)
    B_sp_log_ders=self_product_log_ders(B_orb_areps, B_arep_rhos, B_orb_rhos, B_ibo_nums, B_ibo_atom_nums)

    kern_comp_dim=A_orb_areps.shape[-1]+2
    Kernel=np.zeros((A_num_mols, B_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :, :]=gauss_sep_IBO_kernel_row_wders(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id, :, :], A_orb_rhos[A_mol_id, :],
                                A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], B_num_mols, B_num_mols,
                                B_orb_areps, B_arep_rhos, B_orb_rhos,
                                B_ibo_nums, B_ibo_atom_nums, inv_sq_global_sigma,
                                kern_comp_dim, A_sp_log_ders[A_mol_id, :, :], B_sp_log_ders)
        for j in range(B_num_mols):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
    return Kernel


@njit(fastmath=True, parallel=True)
def numba_gauss_sep_IBO_sym_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums, inv_sq_global_sigma,
                                kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    A_sp_log_ders=self_product_log_ders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_ibo_nums, A_ibo_atom_nums)

    kern_comp_dim=A_orb_areps.shape[-1]+2
    Kernel=np.zeros((A_num_mols, A_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        Kernel[A_mol_id, :, :]=gauss_sep_IBO_kernel_row_wders(A_orb_areps[A_mol_id, :, :, :], A_arep_rhos[A_mol_id, :, :], A_orb_rhos[A_mol_id, :],
                                A_ibo_nums[A_mol_id], A_ibo_atom_nums[A_mol_id, :], A_mol_id+1, A_num_mols,
                                A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_ibo_nums, A_ibo_atom_nums, inv_sq_global_sigma,
                                kern_comp_dim, A_sp_log_ders[A_mol_id, :, :], A_sp_log_ders)
        for j in range(A_mol_id+1):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
        
    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id, :]=Kernel[A_mol_id, A_mol_id2, :]

    return Kernel

def gauss_sep_IBO_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=True, with_ders=False, use_Fortran=True):
    if preserve_converted_arrays:
        Ac_renormed=copy.deepcopy(Ac)
        Bc_renormed=copy.deepcopy(Bc)
    else:
        Ac_renormed=Ac
        Bc_renormed=Bc
    Ac_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas[1:])
    sym_kernel=(Bc is None)
    if not sym_kernel:
        Bc_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas[1:])
    inv_sq_global_sigma=1.0/sigmas[0]**2
    if with_ders:
        kern_der_resc=np.array([2.0/sigmas[0]**3, *2.0/sigmas[1:]])
        if sym_kernel:
            return numba_gauss_sep_IBO_sym_kernel_wders(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums, inv_sq_global_sigma, kern_der_resc)
        else:
            return numba_gauss_sep_IBO_kernel_wders(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums,
                                Bc_renormed.ibo_atom_scaled_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos,
                                Bc_renormed.ibo_nums, Bc_renormed.ibo_atom_nums, inv_sq_global_sigma, kern_der_resc)
    else:
        if sym_kernel:
            return numba_gauss_sep_IBO_sym_kernel(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums, inv_sq_global_sigma)
        else:
            return numba_gauss_sep_IBO_kernel(Ac_renormed.ibo_atom_scaled_sreps, Ac_renormed.ibo_arep_rhos, Ac_renormed.ibo_rhos,
                                Ac_renormed.ibo_nums, Ac_renormed.ibo_atom_nums,
                                Bc_renormed.ibo_atom_scaled_sreps, Bc_renormed.ibo_arep_rhos, Bc_renormed.ibo_rhos,
                                Bc_renormed.ibo_nums, Bc_renormed.ibo_atom_nums, inv_sq_global_sigma)


def gauss_sep_IBO_kernel(A, B, sigmas, with_ders=False, global_Gauss=False):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, single_virtual_ibo=global_Gauss)
    Bc=GMO_sep_IBO_kern_input(oml_compound_array=B, single_virtual_ibo=global_Gauss)
    return gauss_sep_IBO_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

def gauss_sep_IBO_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=True, with_ders=False, global_Gauss=False):
    return gauss_sep_IBO_kernel_conv(Ac, None, sigmas, preserve_converted_arrays=preserve_converted_arrays, with_ders=with_ders)

def gauss_sep_IBO_sym_kernel(A, sigmas, with_ders=False, global_Gauss=False):
    Ac=GMO_sep_IBO_kern_input(oml_compound_array=A, single_virtual_ibo=global_Gauss)
    return gauss_sep_IBO_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=False, with_ders=with_ders)



