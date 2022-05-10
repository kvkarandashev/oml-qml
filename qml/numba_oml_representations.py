from numba import njit, prange
import numpy as np


@njit(fastmath=True)
def gen_orb_atom_scalar_rep(coup_mats, coeffs, angular_momenta, atom_ao_range, max_ang_mom, scalar_reps):

    cur_array_position=0
    num_coup_mats=coup_mats.shape[0]
    scalar_reps[:]=0.0

    for mat_counter in range(num_coup_mats):
        for same_atom_check in range(2):
            for ang_mom1 in range(max_ang_mom+1):
                for ang_mom2 in range(max_ang_mom+1):
                    if not ((same_atom_check==0) and (ang_mom1 > ang_mom2)):
                        add_orb_atom_coupling(coup_mats[mat_counter], coeffs, angular_momenta, atom_ao_range,
                                same_atom_check, ang_mom1, ang_mom2, scalar_reps[cur_array_position:cur_array_position+1])
                        cur_array_position=cur_array_position+1

@njit(fastmath=True)
def add_orb_atom_coupling(mat, coeffs, angular_momenta, atom_ao_range, same_atom, ang_mom1, ang_mom2, cur_coupling_val):

    for ao1 in range(atom_ao_range[0], atom_ao_range[1]):
        if angular_momenta[ao1]==ang_mom1:
            if (same_atom==0):
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, atom_ao_range[0], atom_ao_range[1], ao1, cur_coupling_val)
            else:
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, 0, atom_ao_range[0], ao1, cur_coupling_val)
                add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, atom_ao_range[1], coeffs.shape[0], ao1, cur_coupling_val)

@njit(fastmath=True)
def add_aos_coupling(mat, coeffs, angular_momenta, ang_mom2, lower_index, upper_index, ao1, cur_coupling_val):

    for ao2 in range(lower_index, upper_index):
        if angular_momenta[ao2] == ang_mom2:
            cur_coupling_val[:]+=mat[ao2, ao1]*coeffs[ao1]*coeffs[ao2]



@njit(fastmath=True)
def ang_mom_descr(ovlp_mat, coeffs, angular_momenta, atom_ao_range, max_ang_mom, scalar_reps, rho_val):
    scalar_reps[:]=0.0
    for ang_mom in range(1, max_ang_mom+1):
        add_orb_atom_coupling(ovlp_mat, coeffs, angular_momenta, atom_ao_range, 0, ang_mom, ang_mom, scalar_reps[ang_mom-1:ang_mom])
    rho_val[0]=np.sum(scalar_reps)
    add_orb_atom_coupling(ovlp_mat, coeffs, angular_momenta, atom_ao_range, 0, 0, 0, rho_val)

