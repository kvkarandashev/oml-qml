MODULE foml_module
implicit none
public

double precision, PARAMETER :: BoltzmannConstant=3.16681536D-6
double precision, parameter :: pi=3.14159265358979323846

contains

PURE SUBROUTINE flinear_base_kernel_row(num_scal_reps,&
                    A_ibo_atom_sreps_scaled, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps_scaled, B_rhos, B_max_tot_num_ibo_atom_reps,&
                    kernel_row)
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps
double precision, dimension(num_scal_reps,&
    A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps,&
    B_max_tot_num_ibo_atom_reps), intent(in):: B_ibo_atom_sreps_scaled
double precision, dimension(A_num_mols), intent(inout):: kernel_row
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps), intent(in):: B_rhos
integer:: A_mol_counter


do A_mol_counter=1, A_num_mols
    kernel_row(A_mol_counter)=flin_base_kernel_element(num_scal_reps, A_rhos(:, A_mol_counter),&
            A_ibo_atom_sreps_scaled(:, :, A_mol_counter), A_max_tot_num_ibo_atom_reps, B_rhos,&
            B_ibo_atom_sreps_scaled, B_max_tot_num_ibo_atom_reps)
enddo

END SUBROUTINE flinear_base_kernel_row

SUBROUTINE flin_base_self_products(num_scal_reps, A_ibo_atom_sreps_scaled, A_rhos,&
                                    A_max_tot_num_ibo_atom_reps, A_num_mols, AA_products)
integer, intent(in):: num_scal_reps, A_max_tot_num_ibo_atom_reps, A_num_mols
double precision, dimension(num_scal_reps,&
    A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_rhos
double precision, dimension(A_num_mols), intent(inout):: AA_products
integer:: A_mol_counter

!$OMP PARALLEL DO
do A_mol_counter=1, A_num_mols
    AA_products(A_mol_counter)=flin_base_kernel_element(num_scal_reps, A_rhos(:, A_mol_counter),&
        A_ibo_atom_sreps_scaled(:, :, A_mol_counter), A_max_tot_num_ibo_atom_reps,&
        A_rhos(:, A_mol_counter), A_ibo_atom_sreps_scaled(:, :, A_mol_counter),&
        A_max_tot_num_ibo_atom_reps)
enddo
!$OMP END PARALLEL DO


END SUBROUTINE




PURE FUNCTION flin_base_kernel_element(num_scal_reps,&
                                        A_rhos, A_ibo_atom_sreps_scaled, A_max_tot_num_ibo_atom_reps,&
                                        B_rhos, B_ibo_atom_sreps_scaled, B_max_tot_num_ibo_atom_reps)
implicit none
integer, intent(in):: num_scal_reps, A_max_tot_num_ibo_atom_reps, B_max_tot_num_ibo_atom_reps
double precision, dimension(num_scal_reps, A_max_tot_num_ibo_atom_reps), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps, B_max_tot_num_ibo_atom_reps), intent(in):: B_ibo_atom_sreps_scaled
double precision, dimension(A_max_tot_num_ibo_atom_reps), intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps), intent(in):: B_rhos
double precision:: flin_base_kernel_element
integer:: i_A, i_B

    flin_base_kernel_element=0.0
    do i_A=1, A_max_tot_num_ibo_atom_reps
        do i_B=1, B_max_tot_num_ibo_atom_reps
            flin_base_kernel_element=flin_base_kernel_element+&
                        exp(-sum((A_ibo_atom_sreps_scaled(:, i_A)-B_ibo_atom_sreps_scaled(:, i_B))**2)/4)&
                                *A_rhos(i_A)*B_rhos(i_B)
        enddo
    enddo

END FUNCTION

SUBROUTINE flinear_base_kernel_mat_with_opt(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, sym_kernel_mat, kernel_mat, AA_products, BB_products)
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
logical, intent(in):: sym_kernel_mat
double precision, dimension(A_num_mols, B_num_mols), intent(inout):: kernel_mat
double precision, dimension(A_num_mols), intent(inout), optional:: AA_products
double precision, dimension(B_num_mols), intent(inout), optional:: BB_products
double precision, dimension(num_scal_reps, A_max_tot_num_ibo_atom_reps,&
                                                A_num_mols):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps, B_max_tot_num_ibo_atom_reps,&
                                                B_num_mols):: B_ibo_atom_sreps_scaled
integer:: B_mol_counter
integer:: upper_A_mol_counter


call scalar_rep_resc(A_ibo_atom_sreps_scaled, A_ibo_atom_sreps, width_params, num_scal_reps,&
                                        A_max_tot_num_ibo_atom_reps, A_num_mols)
if (sym_kernel_mat) then
    B_ibo_atom_sreps_scaled=A_ibo_atom_sreps_scaled
else
    call scalar_rep_resc(B_ibo_atom_sreps_scaled, B_ibo_atom_sreps, width_params, num_scal_reps,&
                                        B_max_tot_num_ibo_atom_reps, B_num_mols)
endif
if (present(AA_products)) &
    call flin_base_self_products(num_scal_reps, A_ibo_atom_sreps_scaled, A_rhos,&
                            A_max_tot_num_ibo_atom_reps, A_num_mols, AA_products)
if (present(BB_products)) then
    if (sym_kernel_mat) then
        BB_products=AA_products
    else
        call flin_base_self_products(num_scal_reps, B_ibo_atom_sreps_scaled, B_rhos,&
                            B_max_tot_num_ibo_atom_reps, B_num_mols, BB_products)
    endif
endif
!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
do B_mol_counter = 1, B_num_mols
    if (sym_kernel_mat) then
        upper_A_mol_counter=B_num_mols
    else
        upper_A_mol_counter=A_num_mols
    endif
    call flinear_base_kernel_row(num_scal_reps,&
            A_ibo_atom_sreps_scaled, A_rhos, A_max_tot_num_ibo_atom_reps, upper_A_mol_counter,&
            B_ibo_atom_sreps_scaled(:, :, B_mol_counter), B_rhos(:, B_mol_counter), B_max_tot_num_ibo_atom_reps,&
            kernel_mat(:, B_mol_counter))
enddo
!$OMP END PARALLEL DO

if (sym_kernel_mat) call symmetrize_matrix(kernel_mat, A_num_mols)

END SUBROUTINE flinear_base_kernel_mat_with_opt

SUBROUTINE symmetrize_matrix(matrix, d)
implicit none
integer, intent(in):: d
double precision, dimension(d, d), intent(inout):: matrix
integer:: i1, i2

!$OMP PARALLEL DO
do i1=1, d
    do i2=1, i1
        matrix(i1, i2)=matrix(i2, i1)
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE


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



END MODULE
