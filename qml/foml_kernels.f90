! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen, Lars A. Bratholm, Felix A. Faber
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.


SUBROUTINE fgmo_sep_ibo_kernel(num_scalar_reps,&
                    A_ibo_atom_reps, A_ibo_arep_rhos, A_ibo_rhos,&
                    A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols,&
                    B_ibo_atom_reps, B_ibo_arep_rhos, B_ibo_rhos,&
                    B_max_num_ibo_atom_reps, B_max_num_ibos, B_num_mols,&
                    width_params, sigma, density_neglect,&
                    normalize_lb_kernel, sym_kernel_mat, kernel_mat)
use foml_module, only : scalar_rep_resc_ibo_sep, flin_ibo_self_products,&
            fgmo_sep_ibo_kernel_element, symmetrize_matrix
implicit none
integer, intent(in):: num_scalar_reps
integer, intent(in):: A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols
integer, intent(in):: B_max_num_ibo_atom_reps, B_max_num_ibos, B_num_mols
double precision, dimension(:,:,:,:), intent(in):: A_ibo_atom_reps, B_ibo_atom_reps
double precision, dimension(:,:,:), intent(in):: A_ibo_arep_rhos, B_ibo_arep_rhos
double precision, dimension(:, :), intent(in):: A_ibo_rhos, B_ibo_rhos
double precision, dimension(:), intent(in):: width_params
double precision, intent(in):: sigma
double precision, intent(in):: density_neglect
logical, intent(in):: normalize_lb_kernel
logical, intent(in):: sym_kernel_mat
double precision, dimension(:, :), intent(inout):: kernel_mat
double precision, dimension(:, :, :, :), allocatable:: A_ibo_atom_sreps, B_ibo_atom_sreps
double precision, dimension(:, :), allocatable:: A_ibo_self_products, B_ibo_self_products
integer:: B_mol_counter, upper_A_mol_counter, A_mol_counter

allocate(A_ibo_atom_sreps(num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols),&
    B_ibo_atom_sreps(num_scalar_reps, B_max_num_ibo_atom_reps, B_max_num_ibos, B_num_mols))
call scalar_rep_resc_ibo_sep(A_ibo_atom_reps, width_params, num_scalar_reps, A_max_num_ibo_atom_reps,&
        A_max_num_ibos, A_num_mols, A_ibo_atom_sreps)
call scalar_rep_resc_ibo_sep(B_ibo_atom_reps, width_params, num_scalar_reps, B_max_num_ibo_atom_reps,&
        B_max_num_ibos, B_num_mols, B_ibo_atom_sreps)

allocate(A_ibo_self_products(A_max_num_ibos, A_num_mols),&
        B_ibo_self_products(B_max_num_ibos, B_num_mols))
call flin_ibo_self_products(num_scalar_reps, A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_rhos,&
        A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols, density_neglect, A_ibo_self_products)
call flin_ibo_self_products(num_scalar_reps, B_ibo_atom_sreps, B_ibo_arep_rhos, B_ibo_rhos,&
        B_max_num_ibo_atom_reps, B_max_num_ibos, B_num_mols, density_neglect, B_ibo_self_products)

!$OMP PARALLEL DO PRIVATE(upper_A_mol_counter) SCHEDULE(DYNAMIC)
do B_mol_counter=1, B_num_mols
    if (sym_kernel_mat) then
        upper_A_mol_counter=B_mol_counter
    else
        upper_A_mol_counter=A_num_mols
    endif
    do A_mol_counter=1, upper_A_mol_counter
        call fgmo_sep_ibo_kernel_element(num_scalar_reps, A_ibo_atom_sreps(:,:,:,A_mol_counter),&
            A_ibo_arep_rhos(:,:,A_mol_counter), A_ibo_rhos(:,A_mol_counter),&
            A_ibo_self_products(:,A_mol_counter), A_max_num_ibo_atom_reps, A_max_num_ibos,&
            B_ibo_atom_sreps(:,:,:,B_mol_counter), B_ibo_arep_rhos(:,:,B_mol_counter),&
            B_ibo_rhos(:,B_mol_counter), B_ibo_self_products(:,B_mol_counter),&
            B_max_num_ibo_atom_reps, B_max_num_ibos, sigma, density_neglect,&
            normalize_lb_kernel, kernel_mat(A_mol_counter, B_mol_counter))
    enddo
enddo
!$OMP END PARALLEL DO

if (sym_kernel_mat) call symmetrize_matrix(kernel_mat, A_num_mols)

END SUBROUTINE fgmo_sep_ibo_kernel

SUBROUTINE fgmo_kernel(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, sigma, density_neglect, normalize_lb_kernel, sym_kernel_mat, kernel_mat)
use foml_module, only : fgmo_sq_dist_halfmat, symmetrize_matrix
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(:,:,:), intent(in):: A_ibo_atom_sreps
double precision, dimension(:,:,:), intent(in):: B_ibo_atom_sreps
double precision, dimension(:, :), intent(in):: A_rhos, B_rhos
double precision, dimension(:), intent(in):: width_params
double precision, intent(in):: sigma
double precision, intent(in):: density_neglect
logical, intent(in):: normalize_lb_kernel
logical, intent(in):: sym_kernel_mat
double precision, dimension(:, :), intent(inout):: kernel_mat ! (A_num_mols, B_num_mols)
integer:: A_mol_counter, B_mol_counter, upper_A_mol_counter

call fgmo_sq_dist_halfmat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, normalize_lb_kernel, sym_kernel_mat, kernel_mat)

!$OMP PARALLEL DO PRIVATE(upper_A_mol_counter) SCHEDULE(DYNAMIC)
do B_mol_counter = 1, B_num_mols
    if (sym_kernel_mat) then
        upper_A_mol_counter=B_mol_counter
    else
        upper_A_mol_counter=A_num_mols
    endif
    do A_mol_counter=1, upper_A_mol_counter
        kernel_mat(A_mol_counter, B_mol_counter)=exp(-kernel_mat(A_mol_counter, B_mol_counter)/2/sigma**2)
    enddo
enddo
!$OMP END PARALLEL DO

if (sym_kernel_mat) call symmetrize_matrix(kernel_mat, A_num_mols)

END SUBROUTINE fgmo_kernel

SUBROUTINE flinear_base_kernel_mat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, sym_kernel_mat, kernel_mat)
use foml_module, only : flinear_base_kernel_mat_with_opt
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(:, :, :), intent(in):: A_ibo_atom_sreps
double precision, dimension(:, :, :), intent(in):: B_ibo_atom_sreps
double precision, dimension(:, :), intent(in):: A_rhos
double precision, dimension(:, :), intent(in):: B_rhos
double precision, dimension(:), intent(in):: width_params
double precision, intent(in):: density_neglect
logical, intent(in):: sym_kernel_mat
double precision, dimension(:, :), intent(inout):: kernel_mat


call flinear_base_kernel_mat_with_opt(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, sym_kernel_mat, kernel_mat)

END SUBROUTINE flinear_base_kernel_mat


SUBROUTINE fgmo_sq_dist(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, normalize_lb_kernel, sym_kernel_mat, sq_dist_mat)
use foml_module, only : symmetrize_matrix, fgmo_sq_dist_halfmat
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(:, :, :), intent(in):: A_ibo_atom_sreps
double precision, dimension(:, :, :), intent(in):: B_ibo_atom_sreps
double precision, dimension(:, :), intent(in):: A_rhos
double precision, dimension(:, :), intent(in):: B_rhos
double precision, dimension(:), intent(in):: width_params
double precision, intent(in):: density_neglect
logical, intent(in):: normalize_lb_kernel
logical, intent(in):: sym_kernel_mat
double precision, dimension(:, :), intent(inout):: sq_dist_mat ! (A_num_mols, B_num_mols)

    call fgmo_sq_dist_halfmat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, normalize_lb_kernel, sym_kernel_mat, sq_dist_mat)
    if (sym_kernel_mat) call symmetrize_matrix(sq_dist_mat, A_num_mols)

END SUBROUTINE fgmo_sq_dist
