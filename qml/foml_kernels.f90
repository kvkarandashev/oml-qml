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



SUBROUTINE fgmo_kernel(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, sigma, normalize_lb_kernel, kernel_mat)
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(:,:,:), intent(in):: A_ibo_atom_sreps
double precision, dimension(:,:,:), intent(in):: B_ibo_atom_sreps
double precision, dimension(:, :), intent(in):: A_rhos, B_rhos
double precision, dimension(:), intent(in):: width_params
double precision, intent(in):: sigma
logical, intent(in):: normalize_lb_kernel
double precision, dimension(:, :), intent(inout):: kernel_mat ! (A_num_mols, B_num_mols)
double precision, dimension(A_num_mols, B_num_mols):: lb_kernel_mat
double precision, dimension(A_num_mols):: AA_products
double precision, dimension(B_num_mols):: BB_products
integer:: A_mol_counter, B_mol_counter

call flinear_base_kernel_mat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, lb_kernel_mat, AA_products, BB_products)

!$OMP PARALLEL DO
do B_mol_counter = 1, B_num_mols
    do A_mol_counter=1, A_num_mols
        if (normalize_lb_kernel) then
            kernel_mat(A_mol_counter, B_mol_counter)=&
                    exp(-(1.0-lb_kernel_mat(A_mol_counter, B_mol_counter)/&
                    sqrt(AA_products(A_mol_counter)*BB_products(B_mol_counter)))/sigma**2)
        else
            kernel_mat(A_mol_counter, B_mol_counter)=&
                    exp(-(AA_products(A_mol_counter)+BB_products(B_mol_counter)-&
                    2*lb_kernel_mat(A_mol_counter, B_mol_counter))/2/sigma**2)
        endif
    enddo
enddo
!$OMP END PARALLEL DO


END SUBROUTINE

SUBROUTINE flinear_base_kernel_mat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, kernel_mat, AA_products, BB_products)
use foml_module, only : flinear_base_kernel_row, flin_base_self_products
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(num_scal_reps,&
    A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_ibo_atom_sreps
double precision, dimension(num_scal_reps,&
    B_max_tot_num_ibo_atom_reps, B_num_mols), intent(in):: B_ibo_atom_sreps
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols),&
    intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps, B_num_mols),&
    intent(in):: B_rhos
double precision, dimension(num_scal_reps), intent(in):: width_params
double precision, dimension(A_num_mols, B_num_mols), intent(inout):: kernel_mat
double precision, dimension(A_num_mols), intent(inout):: AA_products
double precision, dimension(B_num_mols), intent(inout):: BB_products
double precision, dimension(num_scal_reps, A_max_tot_num_ibo_atom_reps,&
                                                A_num_mols):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps, B_max_tot_num_ibo_atom_reps,&
                                                B_num_mols):: B_ibo_atom_sreps_scaled
integer:: B_mol_counter

call scalar_rep_resc(A_ibo_atom_sreps_scaled, A_ibo_atom_sreps, width_params, num_scal_reps,&
                                        A_max_tot_num_ibo_atom_reps, A_num_mols)
call scalar_rep_resc(B_ibo_atom_sreps_scaled, B_ibo_atom_sreps, width_params, num_scal_reps,&
                                        B_max_tot_num_ibo_atom_reps, B_num_mols)

call flin_base_self_products(num_scal_reps, A_ibo_atom_sreps_scaled, A_rhos,&
                            A_max_tot_num_ibo_atom_reps, A_num_mols, AA_products)
call flin_base_self_products(num_scal_reps, B_ibo_atom_sreps_scaled, B_rhos,&
                            B_max_tot_num_ibo_atom_reps, B_num_mols, BB_products)

!$OMP PARALLEL DO
do B_mol_counter = 1, B_num_mols
    call flinear_base_kernel_row(num_scal_reps,&
            A_ibo_atom_sreps_scaled, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
            B_ibo_atom_sreps_scaled(:, :, B_mol_counter), B_rhos(:, B_mol_counter), B_max_tot_num_ibo_atom_reps,&
            kernel_mat(:, B_mol_counter))
enddo
!$OMP END PARALLEL DO

END SUBROUTINE flinear_base_kernel_mat




SUBROUTINE scalar_rep_resc(array_out, array_in, width_params, dim1, dim2, dim3)
implicit none
integer, intent(in):: dim1, dim2, dim3
double precision, dimension(dim1, dim2, dim3), intent(in):: array_in
double precision, dimension(dim1, dim2, dim3), intent(inout):: array_out
double precision, dimension(dim1), intent(in):: width_params
integer:: i2, i3

!$OMP PARALLEL DO
    do i3=1, dim3
        do i2=1, dim2
            array_out(:, i2, i3)=array_in(:, i2, i3)/width_params
        enddo
    enddo
!$OMP END PARALLEL DO


END SUBROUTINE


