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
        sqdist=2.0*(1.0-klin/A_ibo_self_products(A_ibo_counter)/B_ibo_self_products(B_ibo_counter))
        kernel_element=kernel_element+exp(-sqdist/2/sigma**2)*A_ibo_rhos(A_ibo_counter)*B_ibo_rhos(B_ibo_counter)
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

END MODULE
