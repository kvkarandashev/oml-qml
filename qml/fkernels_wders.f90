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


! Gaussian kernel for representation vectors restricted by being normalized and 
! strictly positive.
SUBROUTINE fgaussian_pos_sum_restr_kernel(A, B, sigmas, nA, nB, dimf, kern_el_dim, kernel_wders)
implicit none
integer, intent(in):: nA, nB, dimf, kern_el_dim
double precision, intent(in), dimension(:):: sigmas
double precision, dimension(:, :), intent(in):: A, B
double precision, dimension(:, :, :), intent(inout):: kernel_wders
double precision, dimension(dimf, kern_el_dim, nA):: A_rescaled_wders
double precision, dimension(dimf, kern_el_dim, nB):: B_rescaled_wders
integer:: i_A, i_B
logical:: calc_ders

calc_ders=(kern_el_dim/=1)

call gen_renormalized_pos_sum_reps(A, sigmas, nA, dimf, kern_el_dim, calc_ders, A_rescaled_wders)
call gen_renormalized_pos_sum_reps(B, sigmas, nB, dimf, kern_el_dim, calc_ders, B_rescaled_wders)

!$OMP PARALLEL DO PRIVATE(i_A, i_B)
do i_A=1, nA
    do i_B=1, nB
        call gaussian_pos_sum_restr_kernel_element(A_rescaled_wders(:, :, i_A),&
                    B_rescaled_wders(:, :, i_B), dimf, kern_el_dim, calc_ders, kernel_wders(:, i_B, i_A))
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE

SUBROUTINE gen_renormalized_pos_sum_reps(A, sigmas, nA, dimf, kern_el_dim, calc_ders, A_rescaled_wders)
implicit none
integer, intent(in):: nA, dimf, kern_el_dim
double precision, dimension(dimf, nA), intent(in):: A
double precision, dimension(dimf), intent(in):: sigmas
logical, intent(in):: calc_ders
double precision, dimension(dimf, kern_el_dim, nA), intent(inout):: A_rescaled_wders
double precision:: sigma_sum, R_param, inv_sigma_sum
double precision:: vec_sqnorm, vec_norm
double precision, dimension(:), allocatable:: prefac_log_ders 
integer:: dim_id, i_A

if (calc_ders) allocate(prefac_log_ders(dimf))

sigma_sum=sum(sigmas)
R_param=sqrt(sigma_sum)
inv_sigma_sum=1.0/sigma_sum

!$OMP PARALLEL DO PRIVATE(i_A, vec_sqnorm, vec_norm, prefac_log_ders, dim_id)
do i_A=1, nA
    vec_sqnorm=sum(A(:, i_A)/sigmas)
    vec_norm=sqrt(vec_sqnorm)
    A_rescaled_wders(:, 1, i_A)=R_param/vec_norm*sqrt(A(:, i_A)/sigmas)
    if (calc_ders) then
        prefac_log_ders=inv_sigma_sum
        prefac_log_ders=(prefac_log_ders+A(:, i_A)/sigmas**2/vec_sqnorm)/2
        do dim_id=1, dimf
            A_rescaled_wders(:, dim_id+1, i_A)=prefac_log_ders
            A_rescaled_wders(dim_id, dim_id+1, i_A)=A_rescaled_wders(dim_id, dim_id+1, i_A)-0.5/sigmas(dim_id)
        enddo
    endif
enddo
!$OMP END PARALLEL DO

END SUBROUTINE


PURE SUBROUTINE gaussian_pos_sum_restr_kernel_element(A_resc, B_resc, dimf, kern_el_dim, calc_ders, el_wders)
implicit none
double precision, dimension(dimf, kern_el_dim), intent(in):: A_resc, B_resc
integer, intent(in):: dimf, kern_el_dim
logical, intent(in):: calc_ders
double precision, dimension(kern_el_dim), intent(inout):: el_wders
integer:: dim_id
double precision:: exp1, exp2, cosh_val, prod_comp

    el_wders(1)=1.0
    if (calc_ders) el_wders(2:kern_el_dim)=0.0
    do dim_id=1, dimf
        prod_comp=A_resc(dim_id, 1)*B_resc(dim_id, 1)
        exp1=exp(prod_comp)
        exp2=1.0/exp1
        cosh_val=(exp1+exp2)/2
        el_wders(1)=el_wders(1)*cosh_val
        if (calc_ders) then
            el_wders(2:kern_el_dim)=el_wders(2:kern_el_dim)+(A_resc(:, dim_id+1)+B_resc(:, dim_id+1))&
                    *prod_comp*(exp1-exp2)/2/cosh_val
        endif
    enddo
    if (calc_ders) then
        ! To go from logarithmic derivatives to normal derivatives.
        el_wders(2:kern_el_dim)=el_wders(2:kern_el_dim)*el_wders(1)
    endif

END SUBROUTINE


! Gaussian kernel for representation vectors restricted to being positive.
SUBROUTINE fgaussian_pos_restr_sym_kernel(A_rescaled_wsqrt, A_self_prods, sigmas,&
                                    nA, dimf, nsigmas, kern_el_dim, kernel_wders)
use fkernels_wders_module
implicit none
integer, intent(in):: nA, dimf, kern_el_dim, nsigmas
double precision, dimension(:), intent(in):: sigmas
double precision, dimension(:, :, :), intent(in):: A_rescaled_wsqrt
double precision, dimension(:, :), intent(in):: A_self_prods
double precision, dimension(:, :, :), intent(inout):: kernel_wders
integer:: i_A1, i_A2, lin_kern_el_dim
logical:: calc_ders, single_sigma
double precision:: inv_sq_sigma

calc_ders=(kern_el_dim/=1)

lin_kern_el_dim=kern_el_dim

inv_sq_sigma=sigmas(1)**(-2)

if (calc_ders) then
    lin_kern_el_dim=lin_kern_el_dim-1
    single_sigma=(nsigmas==2)
endif

!$OMP PARALLEL DO PRIVATE(i_A1, i_A2) SCHEDULE(DYNAMIC)
do i_A1=1, nA
    do i_A2=1, i_A1
        call gaussian_pos_restr_kernel_element(A_rescaled_wsqrt(:, :, i_A1), A_rescaled_wsqrt(:, :, i_A2),&
                            A_self_prods(:, i_A1), A_self_prods(:, i_A2), dimf, lin_kern_el_dim,&
                            kern_el_dim, calc_ders, single_sigma, inv_sq_sigma, kernel_wders(:, i_A2, i_A1))
        if (calc_ders) then
            kernel_wders(2, i_A2, i_A1)=kernel_wders(2, i_A2, i_A1)/sigmas(1)*2
            if (single_sigma) then
                kernel_wders(3:kern_el_dim, i_A2, i_A1)=-kernel_wders(3:kern_el_dim, i_A2, i_A1)/sigmas(2)
            else
                kernel_wders(3:kern_el_dim, i_A2, i_A1)=-kernel_wders(3:kern_el_dim, i_A2, i_A1)/sigmas(2:lin_kern_el_dim)
            endif
        endif
    enddo
enddo
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i_A1, i_A2) SCHEDULE(DYNAMIC)
do i_A1=1, nA
    do i_A2=1, i_A1
        kernel_wders(:, i_A1, i_A2)=kernel_wders(:, i_A2, i_A1)
    enddo
enddo
!$OMP END PARALLEL DO


END SUBROUTINE


SUBROUTINE fgaussian_pos_restr_input_init(A, sigmas, nsigmas, nA, dimf,&
                            kern_el_dim, calc_ders, A_rescaled_wsqrt, A_self_prods)
use fkernels_wders_module
implicit none
logical, intent(in):: calc_ders
integer, intent(in):: nA, dimf, kern_el_dim, nsigmas
double precision, intent(in), dimension(:, :):: A
double precision, dimension(:, :, :), intent(inout):: A_rescaled_wsqrt
double precision, dimension(:, :), intent(inout):: A_self_prods
double precision, dimension(:), intent(in):: sigmas
integer:: i_A
logical:: single_sigma

single_sigma=(nsigmas==2)

!$OMP PARALLEL DO PRIVATE(i_A)
    do i_A=1, nA
        if (single_sigma) then
             A_rescaled_wsqrt(1, :, i_A)=A(:, i_A)/sigmas(2)
        else
             A_rescaled_wsqrt(1, :, i_A)=A(:, i_A)/sigmas(2:nsigmas)
        endif
        A_rescaled_wsqrt(2, :, i_A)=sqrt(A_rescaled_wsqrt(1, :, i_A))
        call lin_pos_restr_kernel_element_self(A_rescaled_wsqrt(:, :, i_A),&
                    dimf, kern_el_dim, calc_ders, single_sigma, A_self_prods(:, i_A))
        A_self_prods(1, i_A)=sqrt(A_self_prods(1, i_A))
        if (calc_ders) A_self_prods(2:kern_el_dim, i_A)=A_self_prods(2:kern_el_dim, i_A)/2
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE

SUBROUTINE fgaussian_pos_restr_kernel(A_rescaled_wsqrt, B_rescaled_wsqrt, A_self_prods, B_self_prods, sigmas,&
                nA, nB, dimf, nsigmas, kern_el_dim, kernel_wders)
use fkernels_wders_module
implicit none
integer, intent(in):: nA, nB, dimf, kern_el_dim, nsigmas
double precision, dimension(:), intent(in):: sigmas
double precision, dimension(:, :, :), intent(in):: A_rescaled_wsqrt, B_rescaled_wsqrt
double precision, dimension(:, :), intent(in):: A_self_prods, B_self_prods
double precision, dimension(:, :, :), intent(inout):: kernel_wders
integer:: i_A, i_B, lin_kern_el_dim
logical:: calc_ders, single_sigma
double precision:: inv_sq_sigma

calc_ders=(kern_el_dim/=1)

lin_kern_el_dim=kern_el_dim

inv_sq_sigma=sigmas(1)**(-2)

if (calc_ders) then
    lin_kern_el_dim=lin_kern_el_dim-1
    single_sigma=(nsigmas==2)
endif

!$OMP PARALLEL DO PRIVATE(i_A, i_B)
do i_A=1, nA
    do i_B=1, nB
        call gaussian_pos_restr_kernel_element(A_rescaled_wsqrt(:, :, i_A), B_rescaled_wsqrt(:, :, i_B),&
                            A_self_prods(:, i_A), B_self_prods(:, i_B), dimf, lin_kern_el_dim,&
                            kern_el_dim, calc_ders, single_sigma, inv_sq_sigma, kernel_wders(:, i_B, i_A))
        if (calc_ders) then
            kernel_wders(2, i_B, i_A)=kernel_wders(2, i_B, i_A)/sigmas(1)*2
            if (single_sigma) then
                kernel_wders(3:kern_el_dim, i_B, i_A)=-kernel_wders(3:kern_el_dim, i_B, i_A)/sigmas(2)
            else
                kernel_wders(3:kern_el_dim, i_B, i_A)=-kernel_wders(3:kern_el_dim, i_B, i_A)/sigmas(2:lin_kern_el_dim)
            endif
        endif
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE

