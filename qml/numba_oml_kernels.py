from numba import njit, prange
import numpy as np
import itertools, copy

def iterated_orb_reps(oml_comp, pair_reps=False, single_orb_list=False):
    if pair_reps:
        return itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps)
    else:
        if single_orb_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps


def orb_rep_rho_list(oml_comp, pair_reps=False):
    output=[]
    for orb in iterated_orb_reps(oml_comp, pair_reps=pair_reps):
        output.append([orb.rho, orb])
    if pair_reps:
        for i in range(len(oml_comp.comps[0].orb_reps)):
            output[i][0]*=-1
    return output


class GMO_sep_orb_kern_input:
    def __init__(self, oml_compound_array=None, pair_reps=None):
        if pair_reps is None:
            pair_reps=is_pair_reps(oml_compound_array)
        if pair_reps:
            self.max_num_scalar_reps=len(oml_compound_array[0].comps[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)
        else:
            self.max_num_scalar_reps=len(oml_compound_array[0].orb_reps[0].ibo_atom_reps[0].scalar_reps)

        self.num_mols=len(oml_compound_array)
        self.orb_nums=np.zeros((self.num_mols,), dtype=int)
        self.max_num_orbs=0

        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            self.orb_nums[comp_id]=len(rho_orb_list)

        self.max_num_orbs=max(self.orb_nums)
        self.orb_atom_nums=np.zeros((self.num_mols, self.max_num_orbs), dtype=int)
        self.orb_rhos=np.zeros((self.num_mols, self.max_num_orbs))
        for comp_id, oml_comp in enumerate(oml_compound_array):
            rho_orb_list=orb_rep_rho_list(oml_comp, pair_reps=pair_reps)
            for orb_id, [orb_rho, orb_rep] in enumerate(rho_orb_list):
                self.orb_atom_nums[comp_id, orb_id]=len(orb_rep.ibo_atom_reps)
                self.orb_rhos[comp_id, orb_id]=orb_rho

        self.max_num_orb_atom_reps=np.amax(self.orb_atom_nums)


        self.orb_arep_rhos=np.zeros((self.num_mols, self.max_num_orbs, self.max_num_orb_atom_reps))
        self.orb_atom_sreps=np.zeros((self.num_mols, self.max_num_orbs, self.max_num_orb_atom_reps, self.max_num_scalar_reps))
        for ind_comp, oml_comp in enumerate(oml_compound_array):
            for ind_orb, [orb_rho, orb_rep] in enumerate(orb_rep_rho_list(oml_comp, pair_reps=pair_reps)):
                for ind_orb_arep, orb_arep in enumerate(orb_rep.ibo_atom_reps):
                    self.orb_arep_rhos[ind_comp, ind_orb, ind_orb_arep]=orb_arep.rho
                    self.orb_atom_sreps[ind_comp, ind_orb, ind_orb_arep, :]=orb_arep.scalar_reps[:]
        self.orb_atom_scaled_sreps=None

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def width_rescaling(orb_atom_sreps, orb_atom_nums, orb_nums, sigmas):
        orb_atom_scaled_sreps=np.copy(orb_atom_sreps)
        num_mols=orb_atom_scaled_sreps.shape[0]
        for mol_id in prange(num_mols):
            for orb_id in range(orb_nums[mol_id]):
                for arep_id in range(orb_atom_nums[mol_id, orb_id]):
                    orb_atom_scaled_sreps[mol_id, orb_id, arep_id, :]/=2*sigmas
        return orb_atom_scaled_sreps

    # For renormalizing orbital arep rho coefficients.
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def numba_lin_sep_kern_renormalize_arep_rhos(orb_atom_scaled_sreps, orb_nums, orb_atom_nums, orb_arep_rhos):
        num_mols=orb_atom_scaled_sreps.shape[0]

        for mol_id in prange(num_mols):
            sqdiffs_arr=np.zeros((orb_atom_scaled_sreps.shape[-1], ))
            temp_arr=np.zeros((1, ))
            for orb_id in range(orb_nums[mol_id]):
                orb_self_cov(orb_atom_scaled_sreps[mol_id, orb_id, :, :], orb_arep_rhos[mol_id, orb_id, :],
                                                    orb_atom_nums[mol_id, orb_id], sqdiffs_arr, temp_arr)
                orb_arep_rhos[mol_id, orb_id, :]/=np.sqrt(temp_arr[0])
        return orb_arep_rhos

    def rescale_reps(self, sigmas):
        self.orb_atom_scaled_sreps=self.width_rescaling(self.orb_atom_sreps, self.orb_atom_nums, self.orb_nums, sigmas)

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def orb_self_product_log_ders(orb_areps, arep_rhos, orb_atom_nums, orb_nums):
        num_mols=orb_areps.shape[0]
        max_num_orbs=orb_areps.shape[1]

        num_ders=orb_areps.shape[-1]
        orb_comp_dim=num_ders+1

        log_ders=np.zeros((num_mols, max_num_orbs, num_ders))

        for mol_id in prange(num_mols):
            # TO-DO is there a Fortran-like way to define sqdiffs_arr and orb_temp_arr outside the loop and make them private??
            sqdiffs_arr=np.zeros((num_ders,))
            orb_temp_arr=np.zeros((orb_comp_dim,))
            for orb_id in range(orb_nums[mol_id]):
                orb_self_cov_wders(orb_areps[mol_id, orb_id, :, :], arep_rhos[mol_id, orb_id, :], orb_atom_nums[mol_id, orb_id], sqdiffs_arr, orb_temp_arr)
                log_ders[mol_id, orb_id, :]=orb_temp_arr[1:]/orb_temp_arr[0]
    
        return log_ders

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def calc_mol_norm_constants(orb_areps, arep_rhos, orb_rhos, orb_atom_nums, orb_nums):
        num_mols=orb_areps.shape[0]
        output=np.zeros((num_mols,))

        for mol_id in prange(num_mols):
            sqdiffs_arr=np.zeros((orb_areps.shape[-1],))
            orb_temp_arr=np.zeros((1,))
            mol_self_cov(orb_areps[mol_id, :, :, :], arep_rhos[mol_id, :, :], orb_rhos[mol_id, :], orb_atom_nums[mol_id, :],
                                    orb_nums[mol_id], sqdiffs_arr, orb_temp_arr, output[mol_id:mol_id+1])
            output[mol_id]=np.sqrt(output[mol_id])**(-1)
        return output

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def calc_mol_norm_constants_wders(orb_areps, arep_rhos, orb_rhos, orb_sp_log_ders, orb_atom_nums, orb_nums, der_resc):
        num_mols=orb_areps.shape[0]
        kern_comp_dim=orb_areps.shape[-1]+1
        output=np.zeros((num_mols,kern_comp_dim))

        for mol_id in prange(num_mols):
            sqdiffs_arr=np.zeros((orb_areps.shape[-1],))
            orb_temp_arr=np.zeros((kern_comp_dim,))
            mol_self_cov_wders(orb_areps[mol_id], arep_rhos[mol_id], orb_rhos[mol_id], orb_sp_log_ders[mol_id], 
                                    orb_atom_nums[mol_id], orb_nums[mol_id], sqdiffs_arr, orb_temp_arr, output[mol_id])
            output[mol_id, 1:]/=2*output[mol_id, 0]
            output[mol_id, 1:]*=der_resc
            output[mol_id, 0]=np.sqrt(output[mol_id, 0])**(-1)
        return output

    def lin_sep_kern_renormalize_arep_rhos(self, sigmas, with_ders=False, mol_lin_norm=False):

        self.rescale_reps(sigmas)
        self.numba_lin_sep_kern_renormalize_arep_rhos(self.orb_atom_scaled_sreps, self.orb_nums, self.orb_atom_nums, self.orb_arep_rhos)
        if with_ders:
            self.orb_sp_log_ders=self.orb_self_product_log_ders(self.orb_atom_scaled_sreps, self.orb_arep_rhos, self.orb_atom_nums, self.orb_nums)
        if mol_lin_norm:
            if with_ders:
                self.mol_norm_constants=self.calc_mol_norm_constants_wders(self.orb_atom_scaled_sreps, self.orb_arep_rhos, self.orb_rhos, self.orb_sp_log_ders, self.orb_atom_nums, self.orb_nums, 2.0/sigmas)
            else:
                self.mol_norm_constants=self.calc_mol_norm_constants(self.orb_atom_scaled_sreps, self.orb_arep_rhos, self.orb_rhos, self.orb_atom_nums, self.orb_nums)
                

### For linear kernel with separable orbs.

# Orbital-orbital covariance.
@njit(fastmath=True)
def make_sqdiffs_arr(vec1, vec2, sqdiffs_arr):
    sqdiffs_arr[:]=vec1[:]
    sqdiffs_arr-=vec2
    sqdiffs_arr**=2

@njit(fastmath=True)
def orb_orb_cov_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, inout_arr):
    inout_arr[:]=0.0
    for arep_A_id in range(num_orb_areps_A):
        rho_A=arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_orb_areps_B):
            rho_B=arep_rhos_B[arep_B_id]
            make_sqdiffs_arr(orb_areps_A[arep_A_id, :], orb_areps_B[arep_B_id, :], sqdiffs_arr)
            orb_comp=rho_A*rho_B*np.exp(-np.sum(sqdiffs_arr))
            inout_arr[0]+=orb_comp
            inout_arr[1:]+=orb_comp*sqdiffs_arr

@njit(fastmath=True)
def orb_orb_cov(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, inout_arr):
    inout_arr[:]=0.0
    for arep_A_id in range(num_orb_areps_A):
        rho_A=arep_rhos_A[arep_A_id]
        for arep_B_id in range(num_orb_areps_B):
            rho_B=arep_rhos_B[arep_B_id]
            make_sqdiffs_arr(orb_areps_A[arep_A_id, :], orb_areps_B[arep_B_id, :], sqdiffs_arr)
            inout_arr[0]+=rho_A*rho_B*np.exp(-np.sum(sqdiffs_arr))


@njit(fastmath=True)
def orb_orb_cov_wders_log_incl(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, A_sp_log_ders, B_sp_log_ders, num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, inout_arr):
    orb_orb_cov_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, inout_arr)
    inout_arr[1:]-=inout_arr[0]*(A_sp_log_ders+B_sp_log_ders)/2


# Self-covariance (used for normalizing the orbitals).
@njit(fastmath=True)
def orb_self_cov(orb_areps, arep_rhos, num_orb_areps, sqdiffs_arr, inout_arr):
    orb_orb_cov(orb_areps, orb_areps, arep_rhos, arep_rhos, num_orb_areps, num_orb_areps, sqdiffs_arr, inout_arr)

@njit(fastmath=True)
def mol_self_cov(orb_areps, arep_rhos, orb_rhos, orb_atom_nums, orb_num, sqdiffs_arr, orb_temp_arr, inout_arr):
    lin_sep_orb_kernel_el(orb_areps, orb_areps, arep_rhos, arep_rhos, orb_rhos, orb_rhos, orb_atom_nums, orb_atom_nums,
                            orb_num, orb_num, sqdiffs_arr, orb_temp_arr, inout_arr)

@njit(fastmath=True)
def mol_self_cov_wders(orb_areps, arep_rhos, orb_rhos, sp_log_ders, orb_atom_nums, orb_num, sqdiffs_arr, orb_temp_arr, inout_arr):
    lin_sep_orb_kernel_el_wders(orb_areps, orb_areps, arep_rhos, arep_rhos, orb_rhos, orb_rhos, sp_log_ders, sp_log_ders,
                            orb_atom_nums, orb_atom_nums, orb_num, orb_num, sqdiffs_arr, orb_temp_arr, inout_arr)


@njit(fastmath=True)
def lin_sep_orb_kernel_el(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                     A_orb_rhos, B_orb_rhos, A_orb_atom_nums, B_orb_atom_nums,
                     A_orb_num, B_orb_num, sqdiffs_arr, orb_temp_arr, inout_arr):

    inout_arr[:]=0.0
    for A_orb_id in range(A_orb_num):
        rho_A=A_orb_rhos[A_orb_id]
        for B_orb_id in range(B_orb_num):
            orb_orb_cov(A_orb_areps[A_orb_id, :, :], B_orb_areps[B_orb_id, :, :], A_arep_rhos[A_orb_id, :], B_arep_rhos[B_orb_id, :],
                        A_orb_atom_nums[A_orb_id], B_orb_atom_nums[B_orb_id], sqdiffs_arr, orb_temp_arr)
            inout_arr+=rho_A*B_orb_rhos[B_orb_id]*orb_temp_arr

@njit(fastmath=True)
def lin_sep_orb_kernel_row(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                            A_orb_rhos, B_orb_rhos, A_orb_atom_nums, B_orb_atom_nums,
                            A_orb_num, B_orb_nums, upper_B_mol_id, inout_arr):

    inout_arr[:]=0.0

    sqdiffs_arr=np.zeros((A_orb_areps.shape[-1],))
    orb_temp_arr=np.zeros((1,))

    for B_mol_id in range(upper_B_mol_id):
        lin_sep_orb_kernel_el(A_orb_areps, B_orb_areps[B_mol_id], A_arep_rhos, B_arep_rhos[B_mol_id],
             A_orb_rhos, B_orb_rhos[B_mol_id], A_orb_atom_nums, B_orb_atom_nums[B_mol_id], A_orb_num, B_orb_nums[B_mol_id],
             sqdiffs_arr, orb_temp_arr, inout_arr[B_mol_id:B_mol_id+1])

@njit(fastmath=True)
def lin_sep_orb_kernel_el_wders(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_sp_log_ders, B_sp_log_ders, 
                                A_orb_atom_nums, B_orb_atom_nums, A_orb_num, B_orb_num,
                                sqdiffs_arr, orb_temp_arr, inout_arr):

    inout_arr[:]=0.0
    for A_orb_id in range(A_orb_num):
        rho_A=A_orb_rhos[A_orb_id]
        for B_orb_id in range(B_orb_num):
            orb_orb_cov_wders_log_incl(A_orb_areps[A_orb_id], B_orb_areps[B_orb_id], A_arep_rhos[A_orb_id], B_arep_rhos[B_orb_id],
                                                A_sp_log_ders[A_orb_id], B_sp_log_ders[B_orb_id], A_orb_atom_nums[A_orb_id], B_orb_atom_nums[B_orb_id],
                                                sqdiffs_arr, orb_temp_arr)
            inout_arr+=rho_A*B_orb_rhos[B_orb_id]*orb_temp_arr

@njit(fastmath=True)
def lin_sep_orb_kernel_row_wders(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_sp_log_ders, B_sp_log_ders,
                                A_orb_atom_nums, B_orb_atom_nums, A_orb_num, B_orb_nums,
                                upper_B_mol_id, kern_comp_dim, inout_arr):

    inout_arr[:, :]=0.0

    sqdiffs_arr=np.zeros((A_orb_areps.shape[-1],))
    orb_temp_arr=np.zeros((kern_comp_dim,))

    for B_mol_id in range(upper_B_mol_id):
        lin_sep_orb_kernel_el_wders(A_orb_areps, B_orb_areps[B_mol_id], A_arep_rhos, B_arep_rhos[B_mol_id],
                                A_orb_rhos, B_orb_rhos[B_mol_id],  A_sp_log_ders, B_sp_log_ders[B_mol_id],
                                A_orb_atom_nums, B_orb_atom_nums[B_mol_id], A_orb_num, B_orb_nums[B_mol_id],
                                sqdiffs_arr, orb_temp_arr, inout_arr[B_mol_id])


@njit(fastmath=True, parallel=True)
def numba_lin_sep_orb_kernel(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_orb_atom_nums, B_orb_atom_nums,
                                A_orb_nums, B_orb_nums):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    Kernel=np.zeros((A_num_mols, B_num_mols))

    for A_mol_id in prange(A_num_mols):
        lin_sep_orb_kernel_row(A_orb_areps[A_mol_id], B_orb_areps, A_arep_rhos[A_mol_id], B_arep_rhos,
                                A_orb_rhos[A_mol_id], B_orb_rhos, A_orb_atom_nums[A_mol_id], B_orb_atom_nums,
                                A_orb_nums[A_mol_id], B_orb_nums, B_num_mols, Kernel[A_mol_id])
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_lin_sep_orb_sym_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_orb_atom_nums, A_orb_nums):

    A_num_mols=A_orb_areps.shape[0]
    Kernel=np.zeros((A_num_mols, A_num_mols))

    for A_mol_id in prange(A_num_mols):
        lin_sep_orb_kernel_row(A_orb_areps[A_mol_id], A_orb_areps, A_arep_rhos[A_mol_id], A_arep_rhos,
                            A_orb_rhos[A_mol_id], A_orb_rhos, A_orb_atom_nums[A_mol_id], A_orb_atom_nums,
                            A_orb_nums[A_mol_id], A_orb_nums, A_mol_id+1, Kernel[A_mol_id, :A_mol_id+1])

    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id]=Kernel[A_mol_id, A_mol_id2]
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_lin_sep_orb_kernel_wders(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                            A_orb_rhos, B_orb_rhos, A_sp_log_ders, B_sp_log_ders, A_orb_atom_nums, B_orb_atom_nums,
                            A_orb_nums, B_orb_nums, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    kern_comp_dim=A_orb_areps.shape[-1]+1
    Kernel=np.zeros((A_num_mols, B_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        lin_sep_orb_kernel_row_wders(A_orb_areps[A_mol_id], B_orb_areps, A_arep_rhos[A_mol_id], B_arep_rhos,
                                    A_orb_rhos[A_mol_id], B_orb_rhos, A_sp_log_ders[A_mol_id], B_sp_log_ders,
                                    A_orb_atom_nums[A_mol_id], B_orb_atom_nums, A_orb_nums[A_mol_id], B_orb_nums,
                                    B_num_mols, kern_comp_dim, Kernel[A_mol_id])
        for j in range(B_num_mols):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
    return Kernel


@njit(fastmath=True, parallel=True)
def numba_lin_sep_orb_sym_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_sp_log_ders,
                                A_orb_atom_nums, A_orb_nums, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    kern_comp_dim=A_orb_areps.shape[-1]+1
    Kernel=np.zeros((A_num_mols, A_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        lin_sep_orb_kernel_row_wders(A_orb_areps[A_mol_id], A_orb_areps, A_arep_rhos[A_mol_id], A_arep_rhos,
                                A_orb_rhos[A_mol_id], A_orb_rhos, A_sp_log_ders[A_mol_id], A_sp_log_ders,
                                A_orb_atom_nums[A_mol_id], A_orb_atom_nums, A_orb_nums[A_mol_id], A_orb_nums,
                                A_mol_id+1, kern_comp_dim, Kernel[A_mol_id, :A_mol_id+1])
        for j in range(A_mol_id+1):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id, :]=Kernel[A_mol_id, A_mol_id2, :]

    return Kernel



def lin_sep_orb_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=True, with_ders=False, uninit_input=True):
    if preserve_converted_arrays:
        Ac_renormed=copy.deepcopy(Ac)
        Bc_renormed=copy.deepcopy(Bc)
    else:
        Ac_renormed=Ac
        Bc_renormed=Bc
    sym_kernel=(Bc is None)
    if uninit_input:
        Ac_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas, with_ders=with_ders)
        if not sym_kernel:
            Bc_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas, with_ders=with_ders)
    if with_ders:
        kern_der_resc=2.0/sigmas
        if sym_kernel:
            return numba_lin_sep_orb_sym_kernel_wders(Ac_renormed.orb_atom_scaled_sreps, Ac_renormed.orb_arep_rhos, Ac_renormed.orb_rhos,
                                Ac_renormed.orb_sp_log_ders, Ac_renormed.orb_atom_nums, Ac_renormed.orb_nums, kern_der_resc)
        else:
            return numba_lin_sep_orb_kernel_wders(Ac_renormed.orb_atom_scaled_sreps, Bc_renormed.orb_atom_scaled_sreps, 
                                Ac_renormed.orb_arep_rhos, Bc_renormed.orb_arep_rhos, Ac_renormed.orb_rhos, Bc_renormed.orb_rhos,
                                Ac_renormed.orb_sp_log_ders, Bc_renormed.orb_sp_log_ders, Ac_renormed.orb_atom_nums, Bc_renormed.orb_atom_nums,
                                Ac_renormed.orb_nums, Bc_renormed.orb_nums, kern_der_resc)
    else:
        if sym_kernel:
            return numba_lin_sep_orb_sym_kernel(Ac_renormed.orb_atom_scaled_sreps, Ac_renormed.orb_arep_rhos, Ac_renormed.orb_rhos,
                                Ac_renormed.orb_atom_nums, Ac_renormed.orb_nums)
        else:
            return numba_lin_sep_orb_kernel(Ac_renormed.orb_atom_scaled_sreps, Bc_renormed.orb_atom_scaled_sreps, 
                                Ac_renormed.orb_arep_rhos, Bc_renormed.orb_arep_rhos, Ac_renormed.orb_rhos, Bc_renormed.orb_rhos,
                                Ac_renormed.orb_atom_nums, Bc_renormed.orb_atom_nums, Ac_renormed.orb_nums, Bc_renormed.orb_nums)




def is_pair_reps(comp_arr):
    return (hasattr(comp_arr[0], 'comps'))

def lin_sep_orb_kernel(A, B, sigmas, with_ders=False):
    Ac=GMO_sep_orb_kern_input(oml_compound_array=A)
    Bc=GMO_sep_orb_kern_input(oml_compound_array=B)
    return lin_sep_orb_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

def lin_sep_orb_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=True, with_ders=False, uninit_input=True):
    return lin_sep_orb_kernel_conv(Ac, None, sigmas, preserve_converted_arrays=preserve_converted_arrays,
                                            with_ders=with_ders, uninit_input=uninit_input)

def lin_sep_orb_sym_kernel(A, sigmas, with_ders=False):
    Ac=GMO_sep_orb_kern_input(oml_compound_array=A)
    return lin_sep_orb_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

### Generate log derivatives.
@njit(fastmath=True)
def orb_self_cov_wders(orb_areps, arep_rhos, orb_atom_nums, sqdiffs_arr, inout_arr):
    orb_orb_cov_wders(orb_areps, orb_areps, arep_rhos, arep_rhos, orb_atom_nums, orb_atom_nums, sqdiffs_arr, inout_arr)


@njit(fastmath=True)
def numba_find_orb_vec_rep_moments(orb_atom_sreps, orb_arep_rhos, orb_rhos, orb_atom_nums, orb_nums, moment_list):
    num_mols=orb_arep_rhos.shape[0]
    num_moments=moment_list.shape[0]
    output=np.zeros((num_moments, orb_atom_sreps.shape[3]))
    norm_const=0.0
    for mol_id in prange(num_mols):
        for orb_id in range(orb_nums[mol_id]):
            rho_orb=orb_rhos[mol_id, orb_id]
            for arep_id in range(orb_atom_nums[mol_id, orb_id]):
                rho_arep=orb_arep_rhos[mol_id, orb_id, arep_id]
                cur_rho=np.abs(rho_arep*rho_orb)
                norm_const+=cur_rho
                for moment_id in range(num_moments):
                    output[moment_id,:]+=cur_rho*orb_atom_sreps[mol_id, orb_id, arep_id, :]**moment_list[moment_id]
    return output/norm_const

# Auxiliary for hyperparameter optimization.

def find_orb_vec_rep_moments(compound_list_converted, moment):
    return numba_find_orb_vec_rep_moments(compound_list_converted.orb_atom_sreps, compound_list_converted.orb_arep_rhos, compound_list_converted.orb_rhos,
                        compound_list_converted.orb_atom_nums, compound_list_converted.orb_nums, moment)

def oml_ensemble_avs_stddevs(compound_list):
    if isinstance(compound_list, GMO_sep_orb_kern_input):
        compound_list_converted=compound_list
    else:
        compound_list_converted=GMO_sep_orb_kern_input(compound_list)

    moment_vals=find_orb_vec_rep_moments(compound_list_converted, np.array([1, 2]))
    avs=moment_vals[0]
    avs2=moment_vals[1]
    stddevs=np.sqrt(avs2-avs**2)
    return avs, stddevs


############
# For the Gaussian "sep orb" kernel.

@njit(fastmath=True)
def lin2gauss_kern_el(lin_cov, inv_sq_global_sigma):
    return np.exp(-(1.0-lin_cov)*inv_sq_global_sigma)

@njit(fastmath=True)
def lin2gauss_kern_el_wders(orb_temp_arr, inv_sq_global_sigma):
    orb_temp_arr[0]=lin2gauss_kern_el(orb_temp_arr[1], inv_sq_global_sigma)
    orb_temp_arr[1]=orb_temp_arr[0]*(1-orb_temp_arr[1])
    orb_temp_arr[2:]*=orb_temp_arr[0]*inv_sq_global_sigma
    

@njit(fastmath=True)
def orb_orb_cov_gauss(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_orb_areps_A, num_orb_areps_B, inv_sq_global_sigma, sqdiffs_arr, orb_temp_arr):
    orb_orb_cov(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, orb_temp_arr)
    orb_temp_arr[0]=lin2gauss_kern_el(orb_temp_arr[0], inv_sq_global_sigma)

@njit(fastmath=True)
def orb_orb_cov_gauss_wders(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, A_log_ders, B_log_ders, 
                            num_orb_areps_A, num_orb_areps_B, inv_sq_global_sigma, sqdiffs_arr, orb_temp_arr):
    orb_orb_cov_wders_log_incl(orb_areps_A, orb_areps_B, arep_rhos_A, arep_rhos_B, A_log_ders, B_log_ders, 
                                    num_orb_areps_A, num_orb_areps_B, sqdiffs_arr, orb_temp_arr[1:])
    lin2gauss_kern_el_wders(orb_temp_arr, inv_sq_global_sigma)


@njit(fastmath=True)
def gauss_sep_orb_kernel_row(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_orb_atom_nums, B_orb_atom_nums,
                                A_orb_num, B_orb_nums, upper_B_mol_id, inv_sq_global_sigma, inout_arr):


    inout_arr[:]=0.0

    sqdiffs_arr=np.zeros((A_orb_areps.shape[-1],))
    orb_temp_arr=np.zeros((1,))

    for B_mol_id in range(upper_B_mol_id):
        for A_orb_id in range(A_orb_num):
            rho_A=A_orb_rhos[A_orb_id]
            for B_orb_id in range(B_orb_nums[B_mol_id]):
                orb_orb_cov_gauss(A_orb_areps[A_orb_id], B_orb_areps[B_mol_id, B_orb_id], A_arep_rhos[A_orb_id], B_arep_rhos[B_mol_id, B_orb_id],
                                                A_orb_atom_nums[A_orb_id], B_orb_atom_nums[B_mol_id, B_orb_id], inv_sq_global_sigma, sqdiffs_arr, orb_temp_arr)
                inout_arr[B_mol_id]+=rho_A*B_orb_rhos[B_mol_id, B_orb_id]*orb_temp_arr[0]

@njit(fastmath=True)
def gauss_sep_orb_kernel_row_wders(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_sp_log_ders, B_sp_log_ders,
                                A_orb_atom_nums, B_orb_atom_nums, A_orb_num, B_orb_nums,
                                upper_B_mol_id, kern_comp_dim, inv_sq_global_sigma, inout_arr):

    inout_arr[:,:]=0.0

    sqdiffs_arr=np.zeros((A_orb_areps.shape[-1],))
    orb_temp_arr=np.zeros((kern_comp_dim,))

    for B_mol_id in range(upper_B_mol_id):
        for A_orb_id in range(A_orb_num):
            rho_A=A_orb_rhos[A_orb_id]
            for B_orb_id in range(B_orb_nums[B_mol_id]):
                rho_B=B_orb_rhos[B_mol_id, B_orb_id]
                orb_orb_cov_gauss_wders(A_orb_areps[A_orb_id], B_orb_areps[B_mol_id, B_orb_id], A_arep_rhos[A_orb_id], B_arep_rhos[B_mol_id, B_orb_id],
                                                A_sp_log_ders[A_orb_id], B_sp_log_ders[B_mol_id, B_orb_id],
                                                A_orb_atom_nums[A_orb_id], B_orb_atom_nums[B_mol_id, B_orb_id], inv_sq_global_sigma,
                                                sqdiffs_arr, orb_temp_arr)
                inout_arr[B_mol_id,:]+=rho_A*rho_B*orb_temp_arr


@njit(fastmath=True, parallel=True)
def numba_gauss_sep_orb_kernel(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                            A_orb_rhos, B_orb_rhos, A_orb_atom_nums, B_orb_atom_nums,
                            A_orb_nums, B_orb_nums, inv_sq_global_sigma):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    Kernel=np.zeros((A_num_mols, B_num_mols))
    
    for A_mol_id in prange(A_num_mols):
        gauss_sep_orb_kernel_row(A_orb_areps[A_mol_id], B_orb_areps, A_arep_rhos[A_mol_id], B_arep_rhos,
                                A_orb_rhos[A_mol_id], B_orb_rhos, A_orb_atom_nums[A_mol_id], B_orb_atom_nums,
                                A_orb_nums[A_mol_id], B_orb_nums, B_num_mols, inv_sq_global_sigma, Kernel[A_mol_id])
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_gauss_sep_orb_sym_kernel(A_orb_areps, A_arep_rhos, A_orb_rhos,
                                A_orb_atom_nums, A_orb_nums, inv_sq_global_sigma):

    A_num_mols=A_orb_areps.shape[0]
    Kernel=np.zeros((A_num_mols, A_num_mols))

    for A_mol_id in prange(A_num_mols):
        gauss_sep_orb_kernel_row(A_orb_areps[A_mol_id], A_orb_areps, A_arep_rhos[A_mol_id], A_arep_rhos,
                                A_orb_rhos[A_mol_id], A_orb_rhos, A_orb_atom_nums[A_mol_id], A_orb_atom_nums,
                                A_orb_nums[A_mol_id], A_orb_nums, A_mol_id+1, inv_sq_global_sigma, Kernel[A_mol_id])

    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id]=Kernel[A_mol_id, A_mol_id2]
    return Kernel

@njit(fastmath=True, parallel=True)
def numba_gauss_sep_orb_kernel_wders(A_orb_areps, B_orb_areps, A_arep_rhos, B_arep_rhos,
                                A_orb_rhos, B_orb_rhos, A_sp_log_ders, B_sp_log_ders,
                                A_orb_atom_nums, B_orb_atom_nums, A_orb_nums, B_orb_nums,
                                inv_sq_global_sigma, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    B_num_mols=B_orb_areps.shape[0]

    kern_comp_dim=A_orb_areps.shape[-1]+2
    Kernel=np.zeros((A_num_mols, B_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        gauss_sep_orb_kernel_row_wders(A_orb_areps[A_mol_id], B_orb_areps, A_arep_rhos[A_mol_id], B_arep_rhos,
                                A_orb_rhos[A_mol_id], B_orb_rhos, A_sp_log_ders[A_mol_id], B_sp_log_ders,
                                A_orb_atom_nums[A_mol_id], B_orb_atom_nums, A_orb_nums[A_mol_id], B_orb_nums,
                                B_num_mols, kern_comp_dim, inv_sq_global_sigma, Kernel[A_mol_id])
        for j in range(B_num_mols):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
    return Kernel


@njit(fastmath=True, parallel=True)
def numba_gauss_sep_orb_sym_kernel_wders(A_orb_areps, A_arep_rhos, A_orb_rhos, A_sp_log_ders,
                                A_orb_atom_nums, A_orb_nums, inv_sq_global_sigma, kern_der_resc):

    A_num_mols=A_orb_areps.shape[0]

    kern_comp_dim=A_orb_areps.shape[-1]+2
    Kernel=np.zeros((A_num_mols, A_num_mols, kern_comp_dim))

    for A_mol_id in prange(A_num_mols):
        gauss_sep_orb_kernel_row_wders(A_orb_areps[A_mol_id], A_orb_areps, A_arep_rhos[A_mol_id], A_arep_rhos,
                                    A_orb_rhos[A_mol_id], A_orb_rhos, A_sp_log_ders[A_mol_id], A_sp_log_ders,
                                    A_orb_atom_nums[A_mol_id], A_orb_atom_nums, A_orb_nums[A_mol_id], A_orb_nums,
                                    A_mol_id+1, kern_comp_dim, inv_sq_global_sigma, Kernel[A_mol_id])
        for j in range(A_mol_id+1):
            Kernel[A_mol_id, j, 1:]*=kern_der_resc
        
    for A_mol_id in range(A_num_mols):
        for A_mol_id2 in range(A_mol_id):
            Kernel[A_mol_id2, A_mol_id, :]=Kernel[A_mol_id, A_mol_id2, :]

    return Kernel

def gauss_sep_orb_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=True, with_ders=False, use_Fortran=True):
    if preserve_converted_arrays:
        Ac_renormed=copy.deepcopy(Ac)
        Bc_renormed=copy.deepcopy(Bc)
    else:
        Ac_renormed=Ac
        Bc_renormed=Bc
    Ac_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=with_ders)
    sym_kernel=(Bc is None)
    if not sym_kernel:
        Bc_renormed.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=with_ders)
    inv_sq_global_sigma=1.0/sigmas[0]**2
    if with_ders:
        kern_der_resc=np.array([2.0/sigmas[0]**3, *2.0/sigmas[1:]])
        if sym_kernel:
            return numba_gauss_sep_orb_sym_kernel_wders(Ac_renormed.orb_atom_scaled_sreps, Ac_renormed.orb_arep_rhos, Ac_renormed.orb_rhos,
                            Ac_renormed.orb_sp_log_ders, Ac_renormed.orb_atom_nums, Ac_renormed.orb_nums, inv_sq_global_sigma, kern_der_resc)
        else:
            return numba_gauss_sep_orb_kernel_wders(Ac_renormed.orb_atom_scaled_sreps, Bc_renormed.orb_atom_scaled_sreps,
                            Ac_renormed.orb_arep_rhos, Bc_renormed.orb_arep_rhos, Ac_renormed.orb_rhos, Bc_renormed.orb_rhos,
                            Ac_renormed.orb_sp_log_ders, Bc_renormed.orb_sp_log_ders, Ac_renormed.orb_atom_nums, Bc_renormed.orb_atom_nums,
                            Ac_renormed.orb_nums, Bc_renormed.orb_nums, inv_sq_global_sigma, kern_der_resc)
    else:
        if sym_kernel:
            return numba_gauss_sep_orb_sym_kernel(Ac_renormed.orb_atom_scaled_sreps, Ac_renormed.orb_arep_rhos, Ac_renormed.orb_rhos,
                                Ac_renormed.orb_atom_nums, Ac_renormed.orb_nums, inv_sq_global_sigma)
        else:
            return numba_gauss_sep_orb_kernel(Ac_renormed.orb_atom_scaled_sreps, Bc_renormed.orb_atom_scaled_sreps,
                                    Ac_renormed.orb_arep_rhos, Bc_renormed.orb_arep_rhos, Ac_renormed.orb_rhos, Bc_renormed.orb_rhos,
                                    Ac_renormed.orb_atom_nums, Bc_renormed.orb_atom_nums, Ac_renormed.orb_nums, Bc_renormed.orb_nums,
                                    inv_sq_global_sigma)

@njit(fastmath=True, parallel=True)
def numba_kernel_convert_linear_to_Gauss_wders(kernel_in, global_sigma):
    inv_sq_global_sigma=global_sigma**(-2)
    num_A_mols, num_B_mols=kernel_in.shape[:2]
    for i_A in prange(num_A_mols):
        for i_B in range(num_B_mols):
            lin2gauss_kern_el_wders(kernel_in[i_A, i_B, :], inv_sq_global_sigma)
    kernel_in[:, :, 1]*=2.0/global_sigma**3

@njit(fastmath=True, parallel=True)
def numba_kernel_convert_linear_to_Gauss(kernel_in, global_sigma):
    inv_sq_global_sigma=global_sigma**(-2)
    num_A_mols, num_B_mols=kernel_in.shape
    for i_A in prange(num_A_mols):
        for i_B in range(num_B_mols):
            kernel_in[i_A, i_B]=lin2gauss_kern_el(kernel_in[i_A, i_B], inv_sq_global_sigma)


def kernel_convert_linear_to_Gauss(kernel_in, global_sigma, with_ders=False):
    if with_ders:
        kernel_in=np.concatenate((np.zeros((*kernel_in.shape[:2], 1), dtype=int), kernel_in), axis=2)
        numba_kernel_convert_linear_to_Gauss_wders(kernel_in, global_sigma)
    else:
        numba_kernel_convert_linear_to_Gauss(kernel_in, global_sigma)
    return kernel_in

@njit(fastmath=True)
def numba_lin_kernel_norm_readjust(kernel_in, Ac_norms, Bc_norms):
    for i in prange(kernel_in.shape[0]):
        kernel_in[i, :]*=Ac_norms[i]*Bc_norms

@njit(fastmath=True)
def numba_lin_kernel_norm_readjust_wders(kernel_in, Ac_norms, Bc_norms):
    for i in prange(kernel_in.shape[0]):
        for j in range(kernel_in.shape[1]):
            kernel_in[i, j, 1:]/=kernel_in[i, j, 0]
            kernel_in[i, j, 1:]-=Ac_norms[i, 1:]+Bc_norms[j, 1:]
        kernel_in[i, :, 0]*=Ac_norms[i, 0]*Bc_norms[:, 0]
        for j in range(kernel_in.shape[1]):
            kernel_in[i, j, 1:]*=kernel_in[i, j, 0]

def lin_kernel_norm_readjust(kernel_in, Ac, Bc, with_ders=False):
    if with_ders:
        readjust_func=numba_lin_kernel_norm_readjust_wders
    else:
        readjust_func=numba_lin_kernel_norm_readjust
    readjust_func(kernel_in, Ac.mol_norm_constants, Bc.mol_norm_constants)

def gauss_sep_orb_kernel(A, B, sigmas, with_ders=False, global_Gauss=False):
    Ac=GMO_sep_orb_kern_input(oml_compound_array=A)
    Bc=GMO_sep_orb_kern_input(oml_compound_array=B)
    if global_Gauss:
        Ac.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=with_ders, mol_lin_norm=True)
        Bc.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=with_ders, mol_lin_norm=True)
        output=lin_sep_orb_kernel_conv(Ac, Bc, sigmas[1:], with_ders=with_ders, preserve_converted_arrays=False, uninit_input=False)
        lin_kernel_norm_readjust(output, Ac, Bc, with_ders=with_ders)
        return kernel_convert_linear_to_Gauss(output, sigmas[0], with_ders=with_ders)
    else:
        return gauss_sep_orb_kernel_conv(Ac, Bc, sigmas, preserve_converted_arrays=False, with_ders=with_ders)

def gauss_sep_orb_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=True, with_ders=False, global_Gauss=False):
    return gauss_sep_orb_kernel_conv(Ac, None, sigmas, preserve_converted_arrays=preserve_converted_arrays, with_ders=with_ders)

def gauss_sep_orb_sym_kernel(A, sigmas, with_ders=False, global_Gauss=False):
    Ac=GMO_sep_orb_kern_input(oml_compound_array=A)
    if global_Gauss:
        Ac.lin_sep_kern_renormalize_arep_rhos(sigmas[1:], with_ders=with_ders, mol_lin_norm=True)
        output=lin_sep_orb_sym_kernel_conv(Ac, sigmas[1:], preserve_converted_arrays=False, with_ders=with_ders, uninit_input=False)
        lin_kernel_norm_readjust(output, Ac, Ac, with_ders=with_ders)
        return kernel_convert_linear_to_Gauss(output, sigmas[0], with_ders=with_ders)
    else:
        return gauss_sep_orb_sym_kernel_conv(Ac, sigmas, preserve_converted_arrays=False, with_ders=with_ders)



