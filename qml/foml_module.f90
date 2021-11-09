MODULE foml_module
implicit none
public

contains

PURE SUBROUTINE fgmo_sep_ibo_kernel_element(num_scalar_reps, A_ibo_atom_sreps,&
            A_ibo_arep_rhos, A_ibo_rhos, A_ibo_self_products,&
            A_ibo_atom_nums, A_true_ibo_num,&
            A_max_num_ibo_atom_reps, A_max_num_ibos,&
            B_ibo_atom_sreps, B_ibo_arep_rhos, B_ibo_rhos, B_ibo_self_products,&
            B_ibo_atom_nums, B_true_ibo_num, &
            B_max_num_ibo_atom_reps, B_max_num_ibos, sigma,&
            kernel_element)
integer, intent(in):: num_scalar_reps
integer, intent(in):: A_max_num_ibo_atom_reps, A_max_num_ibos
integer, intent(in):: B_max_num_ibo_atom_reps, B_max_num_ibos
integer, intent(in):: A_true_ibo_num, B_true_ibo_num
double precision, dimension(num_scalar_reps,A_max_num_ibo_atom_reps,&
                        A_max_num_ibos), intent(in):: A_ibo_atom_sreps
double precision, dimension(num_scalar_reps,B_max_num_ibo_atom_reps,&
                        B_max_num_ibos), intent(in):: B_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps,A_max_num_ibos), intent(in):: A_ibo_arep_rhos
double precision, dimension(B_max_num_ibo_atom_reps,B_max_num_ibos), intent(in):: B_ibo_arep_rhos
double precision, dimension(A_max_num_ibos), intent(in):: A_ibo_rhos, A_ibo_self_products
double precision, dimension(B_max_num_ibos), intent(in):: B_ibo_rhos, B_ibo_self_products
integer, dimension(A_max_num_ibos), intent(in):: A_ibo_atom_nums
integer, dimension(B_max_num_ibos), intent(in):: B_ibo_atom_nums
double precision, intent(in):: sigma
double precision, intent(inout):: kernel_element
integer:: A_ibo_counter, B_ibo_counter
double precision:: klin, sqdist

kernel_element=0.0
do B_ibo_counter=1, B_true_ibo_num
    do A_ibo_counter=1, A_true_ibo_num
        klin=flin_ibo_product(num_scalar_reps,&
                A_ibo_arep_rhos(:, A_ibo_counter), A_ibo_atom_sreps(:, :, A_ibo_counter),&
                A_max_num_ibo_atom_reps, A_ibo_atom_nums(A_ibo_counter),&
                B_ibo_arep_rhos(:, B_ibo_counter),&
                B_ibo_atom_sreps(:, :, B_ibo_counter),&
                B_max_num_ibo_atom_reps, B_ibo_atom_nums(B_ibo_counter))
        sqdist=lineprod2sqdist(klin, A_ibo_self_products(A_ibo_counter), B_ibo_self_products(B_ibo_counter))
        kernel_element=kernel_element+exp(-sqdist/2/sigma**2)*A_ibo_rhos(A_ibo_counter)*B_ibo_rhos(B_ibo_counter)
    enddo
enddo

END SUBROUTINE

PURE SUBROUTINE fgmo_sep_ibo_sqdist_sum_num(num_scalar_reps, A1_ibo_atom_sreps,&
            A1_ibo_arep_rhos, A1_ibo_self_products,&
            A1_ibo_atom_nums, A1_true_ibo_num,&
            A2_ibo_atom_sreps, A2_ibo_arep_rhos, A2_ibo_self_products,&
            A2_ibo_atom_nums, A2_true_ibo_num,&
            A_max_num_ibo_atom_reps, A_max_num_ibos,&
            sqdist_sum)
integer, intent(in):: num_scalar_reps
integer, intent(in):: A_max_num_ibo_atom_reps, A_max_num_ibos
integer, intent(in):: A1_true_ibo_num, A2_true_ibo_num
double precision, dimension(num_scalar_reps,A_max_num_ibo_atom_reps,&
                        A_max_num_ibos), intent(in):: A1_ibo_atom_sreps
double precision, dimension(num_scalar_reps,A_max_num_ibo_atom_reps,&
                        A_max_num_ibos), intent(in):: A2_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps,A_max_num_ibos),&
                intent(in):: A1_ibo_arep_rhos, A2_ibo_arep_rhos
double precision, dimension(A_max_num_ibos), intent(in):: A1_ibo_self_products, &
                                                          A2_ibo_self_products
integer, dimension(A_max_num_ibos), intent(in):: A1_ibo_atom_nums, A2_ibo_atom_nums
double precision, intent(inout):: sqdist_sum
integer:: A1_ibo_counter, A2_ibo_counter
double precision:: klin

sqdist_sum=0.0
do A1_ibo_counter=1, A1_true_ibo_num
    do A2_ibo_counter=1, A2_true_ibo_num
        klin=flin_ibo_product(num_scalar_reps,&
                A1_ibo_arep_rhos(:, A1_ibo_counter), A1_ibo_atom_sreps(:, :, A1_ibo_counter),&
                A_max_num_ibo_atom_reps, A1_ibo_atom_nums(A1_ibo_counter),&
                A2_ibo_arep_rhos(:, A2_ibo_counter), A2_ibo_atom_sreps(:, :, A2_ibo_counter),&
                A_max_num_ibo_atom_reps, A2_ibo_atom_nums(A2_ibo_counter))
        sqdist_sum=sqdist_sum+lineprod2sqdist(klin, A1_ibo_self_products(A1_ibo_counter), A2_ibo_self_products(A2_ibo_counter))
    enddo
enddo

END SUBROUTINE

SUBROUTINE flin_ibo_prod_norms(num_scalar_reps, A_ibo_atom_sreps, A_ibo_arep_rhos,&
                A_ibo_atom_nums, A_ibo_nums,&
                A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols,&
                A_norms)
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols
double precision, dimension(num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols),&
                intent(in):: A_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols),&
                intent(in):: A_ibo_arep_rhos
integer, dimension(A_num_mols), intent(in):: A_ibo_nums
integer, dimension(A_max_num_ibos, A_num_mols), intent(in):: A_ibo_atom_nums
double precision, dimension(A_max_num_ibos, A_num_mols), intent(inout):: A_norms
integer:: A_mol_counter, A_ibo_counter

!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
do A_mol_counter=1, A_num_mols
    do A_ibo_counter=1, A_ibo_nums(A_mol_counter)
        A_norms(A_ibo_counter, A_mol_counter)=sqrt(flin_ibo_self_product(num_scalar_reps,&
                        A_ibo_arep_rhos(:, A_ibo_counter, A_mol_counter), A_ibo_atom_sreps(:,:,A_ibo_counter, A_mol_counter),&
                        A_max_num_ibo_atom_reps, A_ibo_atom_nums(A_ibo_counter, A_mol_counter)))
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE

PURE function flin_ibo_self_product(num_scalar_reps, A_rhos, A_ibo_atom_sreps,&
                        A_max_num_ibo_atom_reps, A_true_num_ibo_atom_reps)
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps, A_true_num_ibo_atom_reps
double precision, dimension(num_scalar_reps,A_max_num_ibo_atom_reps),&
        intent(in):: A_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps), intent(in):: A_rhos
double precision:: flin_ibo_self_product

flin_ibo_self_product=flin_ibo_product(num_scalar_reps, A_rhos,&
        A_ibo_atom_sreps, A_max_num_ibo_atom_reps, A_true_num_ibo_atom_reps,&
        A_rhos, A_ibo_atom_sreps, A_max_num_ibo_atom_reps, A_true_num_ibo_atom_reps)
                                        
END FUNCTION


SUBROUTINE scalar_rep_resc_ibo_sep(array_in, width_params, dim1, dim2, dim3, dim4, array_out)
implicit none
integer, intent(in):: dim1, dim2, dim3, dim4
double precision, dimension(dim1, dim2, dim3, dim4), intent(in):: array_in
double precision, dimension(dim1, dim2, dim3, dim4), intent(inout):: array_out
double precision, dimension(dim1), intent(in):: width_params
integer:: i2, i3, i4

!$OMP PARALLEL DO
do i4=1, dim4
    do i3=1, dim3
        do i2=1, dim2
            array_out(:, i2, i3, i4)=array_in(:, i2, i3, i4)/width_params
        enddo
    enddo
enddo
!$OMP END PARALLEL DO


END SUBROUTINE scalar_rep_resc_ibo_sep

SUBROUTINE symmetrize_matrix(matrix, d)
implicit none
integer, intent(in):: d
double precision, dimension(d, d), intent(inout):: matrix
integer:: i1, i2

!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
do i1=1, d
    do i2=1, i1
        matrix(i1, i2)=matrix(i2, i1)
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE

PURE FUNCTION flin_ibo_product(num_scal_reps,&
                                        A_rhos, A_ibo_atom_sreps_scaled,&
                                        A_max_num_ibo_atom_reps, A_true_num_ibo_atom_reps,&
                                        B_rhos, B_ibo_atom_sreps_scaled,&
                                        B_max_num_ibo_atom_reps, B_true_num_ibo_atom_reps)
implicit none
integer, intent(in):: num_scal_reps, A_max_num_ibo_atom_reps, B_max_num_ibo_atom_reps,&
                        A_true_num_ibo_atom_reps, B_true_num_ibo_atom_reps
double precision, dimension(num_scal_reps, A_max_num_ibo_atom_reps), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps, B_max_num_ibo_atom_reps), intent(in):: B_ibo_atom_sreps_scaled
double precision, dimension(A_max_num_ibo_atom_reps), intent(in):: A_rhos
double precision, dimension(B_max_num_ibo_atom_reps), intent(in):: B_rhos
double precision:: flin_ibo_product
integer:: i_A, i_B

    flin_ibo_product=0.0
    do i_A=1, A_true_num_ibo_atom_reps
        do i_B=1, B_true_num_ibo_atom_reps
            flin_ibo_product=flin_ibo_product+exp(-sum((A_ibo_atom_sreps_scaled(:, i_A)&
                            -B_ibo_atom_sreps_scaled(:, i_B))**2)/4)*A_rhos(i_A)*B_rhos(i_B)
        enddo
    enddo

END FUNCTION

PURE FUNCTION lineprod2sqdist(AB_prod, AA_prod, BB_prod)
double precision, intent(in):: AB_prod, AA_prod, BB_prod
double precision:: lineprod2sqdist

lineprod2sqdist=2*(1.0-AB_prod/AA_prod/BB_prod)

END FUNCTION


!!!! Part of earlier drafts of the code preserved for testing purposes. Should be deleted?

PURE SUBROUTINE flinear_base_kernel_row(num_scal_reps,&
                    A_ibo_atom_sreps_scaled, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps_scaled, B_rhos, B_max_tot_num_ibo_atom_reps,&
                    density_neglect, kernel_row)
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps
double precision, dimension(num_scal_reps,&
    A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps,&
    B_max_tot_num_ibo_atom_reps), intent(in):: B_ibo_atom_sreps_scaled
double precision, intent(in):: density_neglect
double precision, dimension(A_num_mols), intent(inout):: kernel_row
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps), intent(in):: B_rhos
integer:: A_mol_counter


do A_mol_counter=1, A_num_mols
    kernel_row(A_mol_counter)=flin_base_kernel_element(num_scal_reps, A_rhos(:, A_mol_counter),&
            A_ibo_atom_sreps_scaled(:, :, A_mol_counter), A_max_tot_num_ibo_atom_reps, B_rhos,&
            B_ibo_atom_sreps_scaled, B_max_tot_num_ibo_atom_reps, density_neglect)
enddo

END SUBROUTINE flinear_base_kernel_row

SUBROUTINE flin_base_self_products(num_scal_reps, A_ibo_atom_sreps_scaled, A_rhos,&
                                    A_max_tot_num_ibo_atom_reps, A_num_mols, density_neglect, AA_products)
integer, intent(in):: num_scal_reps, A_max_tot_num_ibo_atom_reps, A_num_mols
double precision, dimension(num_scal_reps,&
    A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_rhos
double precision, intent(in):: density_neglect
double precision, dimension(A_num_mols), intent(inout):: AA_products
integer:: A_mol_counter

!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
do A_mol_counter=1, A_num_mols
    AA_products(A_mol_counter)=flin_self_product(num_scal_reps,&
                        A_rhos(:, A_mol_counter), A_ibo_atom_sreps_scaled(:,:,A_mol_counter),&
                        A_max_tot_num_ibo_atom_reps, density_neglect)
enddo
!$OMP END PARALLEL DO


END SUBROUTINE

PURE function flin_self_product(num_scalar_reps, A_rhos, A_ibo_atom_sreps,&
                        A_max_num_ibo_atom_reps, density_neglect)
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps
double precision, dimension(num_scalar_reps,A_max_num_ibo_atom_reps),&
        intent(in):: A_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps), intent(in):: A_rhos
double precision, intent(in):: density_neglect
double precision:: flin_self_product

flin_self_product=flin_base_kernel_element(num_scalar_reps, A_rhos,&
        A_ibo_atom_sreps, A_max_num_ibo_atom_reps,&
        A_rhos, A_ibo_atom_sreps, A_max_num_ibo_atom_reps, density_neglect)
                                        
END FUNCTION


PURE FUNCTION flin_base_kernel_element(num_scal_reps,&
                                        A_rhos, A_ibo_atom_sreps_scaled, A_max_tot_num_ibo_atom_reps,&
                                        B_rhos, B_ibo_atom_sreps_scaled, B_max_tot_num_ibo_atom_reps,&
                                        density_neglect)
implicit none
integer, intent(in):: num_scal_reps, A_max_tot_num_ibo_atom_reps, B_max_tot_num_ibo_atom_reps
double precision, dimension(num_scal_reps, A_max_tot_num_ibo_atom_reps), intent(in):: A_ibo_atom_sreps_scaled
double precision, dimension(num_scal_reps, B_max_tot_num_ibo_atom_reps), intent(in):: B_ibo_atom_sreps_scaled
double precision, dimension(A_max_tot_num_ibo_atom_reps), intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps), intent(in):: B_rhos
double precision, intent(in):: density_neglect
double precision:: flin_base_kernel_element
integer:: i_A, i_B
double precision:: A_rho, B_rho

    flin_base_kernel_element=0.0
    do i_A=1, A_max_tot_num_ibo_atom_reps
        A_rho=A_rhos(i_A)
        if (abs(A_rho)<density_neglect) exit
        do i_B=1, B_max_tot_num_ibo_atom_reps
            B_rho=B_rhos(i_B)
            if (abs(B_rho)<density_neglect) exit
            flin_base_kernel_element=flin_base_kernel_element+&
                        exp(-sum((A_ibo_atom_sreps_scaled(:, i_A)-B_ibo_atom_sreps_scaled(:, i_B))**2)/4)&
                                *A_rho*B_rho
        enddo
    enddo

END FUNCTION

SUBROUTINE flinear_base_kernel_mat_with_opt(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, sym_kernel_mat, kernel_mat, AA_products, BB_products)
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
double precision, intent(in):: density_neglect
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
                            A_max_tot_num_ibo_atom_reps, A_num_mols, density_neglect, AA_products)
if (present(BB_products)) then
    if (sym_kernel_mat) then
        BB_products=AA_products
    else
        call flin_base_self_products(num_scal_reps, B_ibo_atom_sreps_scaled, B_rhos,&
                            B_max_tot_num_ibo_atom_reps, B_num_mols, density_neglect, BB_products)
    endif
endif
!$OMP PARALLEL DO PRIVATE(upper_A_mol_counter) SCHEDULE(DYNAMIC)
do B_mol_counter = 1, B_num_mols
    if (sym_kernel_mat) then
        upper_A_mol_counter=B_mol_counter
    else
        upper_A_mol_counter=A_num_mols
    endif
    call flinear_base_kernel_row(num_scal_reps,&
            A_ibo_atom_sreps_scaled, A_rhos, A_max_tot_num_ibo_atom_reps, upper_A_mol_counter,&
            B_ibo_atom_sreps_scaled(:, :, B_mol_counter), B_rhos(:, B_mol_counter), B_max_tot_num_ibo_atom_reps,&
            density_neglect, kernel_mat(:, B_mol_counter))
enddo
!$OMP END PARALLEL DO

if (sym_kernel_mat) call symmetrize_matrix(kernel_mat, A_num_mols)

END SUBROUTINE flinear_base_kernel_mat_with_opt


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

SUBROUTINE fgmo_sq_dist_halfmat(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, normalize_lb_kernel, sym_kernel_mat, sq_dist_mat)
implicit none
integer, intent(in):: num_scal_reps
integer, intent(in):: A_max_tot_num_ibo_atom_reps, A_num_mols
integer, intent(in):: B_max_tot_num_ibo_atom_reps, B_num_mols
double precision, dimension(num_scal_reps, A_max_tot_num_ibo_atom_reps, A_num_mols),&
                    intent(in):: A_ibo_atom_sreps
double precision, dimension(num_scal_reps, B_max_tot_num_ibo_atom_reps, B_num_mols),&
                    intent(in):: B_ibo_atom_sreps
double precision, dimension(A_max_tot_num_ibo_atom_reps, A_num_mols), intent(in):: A_rhos
double precision, dimension(B_max_tot_num_ibo_atom_reps, B_num_mols), intent(in):: B_rhos
double precision, dimension(num_scal_reps), intent(in):: width_params
double precision, intent(in):: density_neglect
logical, intent(in):: normalize_lb_kernel
logical, intent(in):: sym_kernel_mat
double precision, dimension(A_num_mols, B_num_mols), intent(inout):: sq_dist_mat ! (A_num_mols, B_num_mols)
double precision, dimension(A_num_mols):: AA_products
double precision, dimension(B_num_mols):: BB_products
integer:: A_mol_counter, B_mol_counter, upper_A_mol_counter


call flinear_base_kernel_mat_with_opt(num_scal_reps,&
                    A_ibo_atom_sreps, A_rhos, A_max_tot_num_ibo_atom_reps, A_num_mols,&
                    B_ibo_atom_sreps, B_rhos, B_max_tot_num_ibo_atom_reps, B_num_mols,&
                    width_params, density_neglect, sym_kernel_mat, sq_dist_mat, AA_products, BB_products)

!$OMP PARALLEL DO PRIVATE(upper_A_mol_counter) SCHEDULE(DYNAMIC)
do B_mol_counter = 1, B_num_mols
    if (sym_kernel_mat) then
        upper_A_mol_counter=B_mol_counter
    else
        upper_A_mol_counter=A_num_mols
    endif
    do A_mol_counter=1, upper_A_mol_counter
        sq_dist_mat(A_mol_counter, B_mol_counter)=linel2sqdist(sq_dist_mat(A_mol_counter, B_mol_counter),&
                AA_products(A_mol_counter), BB_products(B_mol_counter), normalize_lb_kernel)
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE

PURE FUNCTION linel2sqdist(AB_prod, AA_prod, BB_prod, normalize_lb_kernel)
double precision, intent(in):: AB_prod, AA_prod, BB_prod
logical, intent(in):: normalize_lb_kernel
double precision:: linel2sqdist

if (normalize_lb_kernel) then
    linel2sqdist=2*(1.0-AB_prod/sqrt(AA_prod*BB_prod))
else
    linel2sqdist=AA_prod+BB_prod-2*AB_prod
endif

END FUNCTION


!!!!!!!!!!!!!
!!! For GMO implementation with derivatives.
!!!!!!!!!!!!
PURE SUBROUTINE orb_orb_lin_cov_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_atom_num,&
            A_max_num_ibo_atom_reps,&
            B_ibo_atom_sreps, B_ibo_arep_rhos, B_ibo_atom_num,&
            B_max_num_ibo_atom_reps,&
            orb_cov_components, num_kern_comps)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps,&
                B_max_num_ibo_atom_reps, A_ibo_atom_num, B_ibo_atom_num,&
                num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_ibo_atom_reps),&
        intent(in):: A_ibo_atom_sreps
double precision, dimension(num_scalar_reps, B_max_num_ibo_atom_reps),&
        intent(in):: B_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps), intent(in):: A_ibo_arep_rhos
double precision, dimension(B_max_num_ibo_atom_reps), intent(in):: B_ibo_arep_rhos
double precision, dimension(num_kern_comps), intent(inout):: orb_cov_components
integer:: A_arep_id, B_arep_id
double precision:: exp_fac
double precision, dimension(num_scalar_reps):: sqdiff_vec
integer:: lin_kern_size

if (num_kern_comps==1) then
    lin_kern_size=1
else
    lin_kern_size=num_scalar_reps+1
endif


orb_cov_components=0.0
do A_arep_id=1, A_ibo_atom_num
    do B_arep_id=1, B_ibo_atom_num
        sqdiff_vec=(A_ibo_atom_sreps(:, A_arep_id)-B_ibo_atom_sreps(:, B_arep_id))**2
        exp_fac=exp(-sum(sqdiff_vec))*A_ibo_arep_rhos(A_arep_id)&
                            *B_ibo_arep_rhos(B_arep_id)
        orb_cov_components(1)=orb_cov_components(1)+exp_fac
        if (num_kern_comps/=1) then
            orb_cov_components(2:lin_kern_size)=orb_cov_components(2:lin_kern_size)&
                    -exp_fac*sqdiff_vec
        endif
    enddo
enddo

END SUBROUTINE

PURE SUBROUTINE el_norm_der_log(cov_components, A_self_covs, B_self_covs, num_kern_comps)
integer, intent(in):: num_kern_comps
double precision, dimension(num_kern_comps), intent(inout):: cov_components
double precision, dimension(num_kern_comps), intent(in):: A_self_covs, B_self_covs

    cov_components(:)=cov_components(:)/A_self_covs(1)/B_self_covs(1)
    if (num_kern_comps/=1) & 
        cov_components(2:num_kern_comps-1)=cov_components(2:num_kern_comps-1)&
            -(A_self_covs(2:num_kern_comps-1)+B_self_covs(2:num_kern_comps-1))/2.0*cov_components(1)

END SUBROUTINE


PURE SUBROUTINE orb_self_lin_cov_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_atom_num,&
            A_max_num_ibo_atom_reps, orb_cov_components, num_kern_comps)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps,&
                A_ibo_atom_num, num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_ibo_atom_reps),&
        intent(in):: A_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps), intent(in):: A_ibo_arep_rhos
double precision, dimension(num_kern_comps), intent(inout):: orb_cov_components

    call orb_orb_lin_cov_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_atom_num,&
            A_max_num_ibo_atom_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_atom_num,&
            A_max_num_ibo_atom_reps,&
            orb_cov_components, num_kern_comps)


END SUBROUTINE

PURE SUBROUTINE lin2gauss(converted_comps, inv_sq_sigma, num_kern_comps)
integer, intent(in):: num_kern_comps
double precision, intent(inout), dimension(num_kern_comps):: converted_comps
double precision, intent(in):: inv_sq_sigma

    if (num_kern_comps/=1) then
        converted_comps(3:num_kern_comps)=converted_comps(2:num_kern_comps-1)*inv_sq_sigma
        converted_comps(2)=converted_comps(1)-1.0
    endif
    converted_comps(1)=exp(-inv_sq_sigma*(1.0-converted_comps(1)))
    if (num_kern_comps/=1) converted_comps(2:num_kern_comps)=&
            converted_comps(2:num_kern_comps)*converted_comps(1)

END SUBROUTINE

PURE SUBROUTINE flmo_sep_ibo_kernel_element_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_rhos, A_ibo_atom_nums, A_ibo_num,&
            A_max_num_ibo_atom_reps, A_max_num_ibos,&
            B_ibo_atom_sreps, B_ibo_arep_rhos,&
            B_ibo_rhos, B_ibo_atom_nums, B_ibo_num,&
            B_max_num_ibo_atom_reps, B_max_num_ibos, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs, inv_sq_sigma)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos,&
                        num_kern_comps, B_max_num_ibo_atom_reps, B_max_num_ibos
integer, intent(in):: A_ibo_num, B_ibo_num
double precision, intent(in), dimension(num_scalar_reps, A_max_num_ibo_atom_reps,&
                A_max_num_ibos):: A_ibo_atom_sreps
double precision, intent(in), dimension(num_scalar_reps, B_max_num_ibo_atom_reps,&
                B_max_num_ibos):: B_ibo_atom_sreps
double precision, intent(in), dimension(A_max_num_ibo_atom_reps,&
            A_max_num_ibos):: A_ibo_arep_rhos
double precision, intent(in), dimension(B_max_num_ibo_atom_reps,&
            B_max_num_ibos):: B_ibo_arep_rhos
double precision, intent(in), dimension(A_max_num_ibos):: A_ibo_rhos
double precision, intent(in), dimension(B_max_num_ibos):: B_ibo_rhos
integer, dimension(A_max_num_ibos), intent(in):: A_ibo_atom_nums
integer, dimension(B_max_num_ibos), intent(in):: B_ibo_atom_nums
double precision, intent(in), optional:: inv_sq_sigma
logical, intent(in):: global_gauss
double precision, dimension(num_kern_comps), intent(inout):: kernel_elements
double precision, dimension(num_kern_comps, A_max_num_ibos), intent(in):: A_orb_self_covs
double precision, dimension(num_kern_comps, B_max_num_ibos), intent(in):: B_orb_self_covs
integer:: A_ibo_id, B_ibo_id
double precision, dimension(num_kern_comps):: orb_cov_components

kernel_elements=0.0

do A_ibo_id = 1, A_ibo_num
    do B_ibo_id = 1, B_ibo_num
        call orb_orb_lin_cov_wders(num_scalar_reps,&
            A_ibo_atom_sreps(:, :, A_ibo_id), A_ibo_arep_rhos(:, A_ibo_id),&
            A_ibo_atom_nums(A_ibo_id), A_max_num_ibo_atom_reps,&
            B_ibo_atom_sreps(:, :, B_ibo_id), B_ibo_arep_rhos(:, B_ibo_id),&
            B_ibo_atom_nums(B_ibo_id), B_max_num_ibo_atom_reps,&
            orb_cov_components, num_kern_comps)

        call el_norm_der_log(orb_cov_components, A_orb_self_covs(:, A_ibo_id),&
                                    B_orb_self_covs(:, B_ibo_id), num_kern_comps)

        if (.not.global_gauss) call lin2gauss(orb_cov_components, inv_sq_sigma, num_kern_comps)

        kernel_elements=kernel_elements+orb_cov_components*A_ibo_rhos(A_ibo_id)*B_ibo_rhos(B_ibo_id)
    enddo
enddo


END SUBROUTINE


PURE SUBROUTINE fgmo_sep_ibo_kernel_element_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_rhos, A_ibo_atom_nums, A_ibo_num,&
            A_max_num_ibo_atom_reps, A_max_num_ibos,&
            B_ibo_atom_sreps, B_ibo_arep_rhos,&
            B_ibo_rhos, B_ibo_atom_nums, B_ibo_num,&
            B_max_num_ibo_atom_reps, B_max_num_ibos, sigmas, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs,&
            A_self_covs, B_self_covs)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos,&
                        num_kern_comps, B_max_num_ibo_atom_reps, B_max_num_ibos
integer, intent(in):: A_ibo_num, B_ibo_num
double precision, intent(in), dimension(num_scalar_reps, A_max_num_ibo_atom_reps,&
                A_max_num_ibos):: A_ibo_atom_sreps
double precision, intent(in), dimension(num_scalar_reps, B_max_num_ibo_atom_reps,&
                B_max_num_ibos):: B_ibo_atom_sreps
double precision, intent(in), dimension(A_max_num_ibo_atom_reps,&
            A_max_num_ibos):: A_ibo_arep_rhos
double precision, intent(in), dimension(B_max_num_ibo_atom_reps,&
            B_max_num_ibos):: B_ibo_arep_rhos
double precision, intent(in), dimension(A_max_num_ibos):: A_ibo_rhos
double precision, intent(in), dimension(B_max_num_ibos):: B_ibo_rhos
integer, dimension(A_max_num_ibos), intent(in):: A_ibo_atom_nums
integer, dimension(B_max_num_ibos), intent(in):: B_ibo_atom_nums
double precision, dimension(num_scalar_reps+1), intent(in):: sigmas
logical, intent(in):: global_gauss
double precision, dimension(num_kern_comps), intent(inout):: kernel_elements
double precision, dimension(num_kern_comps, A_max_num_ibos), intent(in):: A_orb_self_covs
double precision, dimension(num_kern_comps, B_max_num_ibos), intent(in):: B_orb_self_covs
double precision, dimension(num_kern_comps), intent(in), optional:: A_self_covs, B_self_covs
double precision:: inv_sq_sigma

inv_sq_sigma=1.0/sigmas(1)**2

call flmo_sep_ibo_kernel_element_wders(num_scalar_reps,&
            A_ibo_atom_sreps, A_ibo_arep_rhos, A_ibo_rhos, A_ibo_atom_nums, A_ibo_num,&
            A_max_num_ibo_atom_reps, A_max_num_ibos,&
            B_ibo_atom_sreps, B_ibo_arep_rhos,&
            B_ibo_rhos, B_ibo_atom_nums, B_ibo_num,&
            B_max_num_ibo_atom_reps, B_max_num_ibos, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs,&
            inv_sq_sigma)

if (global_gauss) then
    call el_norm_der_log(kernel_elements, A_self_covs, B_self_covs, num_kern_comps)
    call lin2gauss(kernel_elements, inv_sq_sigma, num_kern_comps)
endif
if (num_kern_comps /= 1) then
    kernel_elements(2:num_kern_comps)=-kernel_elements(2:num_kern_comps)/sigmas*2
    kernel_elements(2)=kernel_elements(2)*inv_sq_sigma
endif


END SUBROUTINE fgmo_sep_ibo_kernel_element_wders


SUBROUTINE self_cov_prods(num_scalar_reps, A_ibo_atom_sreps,&
                    A_ibo_arep_rhos, A_ibo_atom_nums, A_ibo_nums,&
                    A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols,&
                    A_orb_self_covs, num_kern_comps, A_ibo_rhos, A_self_covs)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_ibo_atom_reps, A_max_num_ibos, A_num_mols, num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_ibo_atom_reps,&
                                A_max_num_ibos, A_num_mols), intent(in):: A_ibo_atom_sreps
double precision, dimension(A_max_num_ibo_atom_reps,&
                                A_max_num_ibos, A_num_mols), intent(in):: A_ibo_arep_rhos
integer, dimension(A_max_num_ibos, A_num_mols), intent(in):: A_ibo_atom_nums
integer, dimension(A_num_mols), intent(in):: A_ibo_nums
double precision, dimension(num_kern_comps, A_max_num_ibos, A_num_mols), intent(inout):: A_orb_self_covs
double precision, dimension(A_max_num_ibos, A_num_mols), intent(in), optional:: A_ibo_rhos
double precision, dimension(num_kern_comps, A_num_mols), intent(inout), optional:: A_self_covs
integer:: mol_id, ibo_id
integer:: der_upper_id
double precision:: dummy_inv_sq_sigma

    if (num_kern_comps/=1) der_upper_id=num_kern_comps-1

!$OMP PARALLEL DO PRIVATE(mol_id, ibo_id) SCHEDULE(DYNAMIC)
    do mol_id=1, A_num_mols
        do ibo_id=1, A_ibo_nums(mol_id)
            call orb_self_lin_cov_wders(num_scalar_reps,&
            A_ibo_atom_sreps(:, :, ibo_id, mol_id), A_ibo_arep_rhos(:, ibo_id, mol_id),&
            A_ibo_atom_nums(ibo_id, mol_id), A_max_num_ibo_atom_reps, A_orb_self_covs(:, ibo_id, mol_id), num_kern_comps)
            if (num_kern_comps/=1) &
            A_orb_self_covs(2:der_upper_id, ibo_id, mol_id)=A_orb_self_covs(2:der_upper_id, ibo_id, mol_id)&
                                /A_orb_self_covs(1, ibo_id, mol_id)
            A_orb_self_covs(1, ibo_id, mol_id)=sqrt(A_orb_self_covs(1, ibo_id, mol_id))
        enddo
        if (present(A_ibo_rhos)) then
            call flmo_sep_ibo_kernel_element_wders(num_scalar_reps,&
                A_ibo_atom_sreps(:, :, :, mol_id), A_ibo_arep_rhos(:, :, mol_id),&
                A_ibo_rhos(:, mol_id), A_ibo_atom_nums(:,mol_id), A_ibo_nums(mol_id),&
                A_max_num_ibo_atom_reps, A_max_num_ibos,&
                A_ibo_atom_sreps(:, :, :, mol_id), A_ibo_arep_rhos(:,:,mol_id),&
                A_ibo_rhos(:,mol_id), A_ibo_atom_nums(:,mol_id), A_ibo_nums(mol_id),&
                A_max_num_ibo_atom_reps, A_max_num_ibos, .TRUE.,&
                A_self_covs(:, mol_id), num_kern_comps,&
                A_orb_self_covs(:, :, mol_id), A_orb_self_covs(:, :, mol_id), dummy_inv_sq_sigma)
            if (num_kern_comps/=1) A_self_covs(2:der_upper_id, mol_id)=A_self_covs(2:der_upper_id, mol_id)/A_self_covs(1, mol_id)
            A_self_covs(1, mol_id)=sqrt(A_self_covs(1, mol_id))
        endif
    enddo
!$OMP END PARALLEL DO


END SUBROUTINE


END MODULE
