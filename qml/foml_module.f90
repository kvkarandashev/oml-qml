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

END MODULE
