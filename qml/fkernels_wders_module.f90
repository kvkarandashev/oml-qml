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

module fkernels_wders_module
implicit none
public

contains



PURE SUBROUTINE gaussian_pos_restr_kernel_element(A_rescaled_wsqrt,&
                    B_rescaled_wsqrt, A_self_prods, B_self_prods, dimf, lin_kern_el_dim, kern_el_dim, calc_ders,&
                    single_sigma_der, inv_sq_sigma, el_wders)
implicit none
integer, intent(in):: dimf, lin_kern_el_dim, kern_el_dim
logical, intent(in):: calc_ders, single_sigma_der
double precision, intent(in):: inv_sq_sigma
double precision, dimension(2, dimf), intent(in):: A_rescaled_wsqrt, B_rescaled_wsqrt
double precision, dimension(kern_el_dim), intent(inout):: el_wders
double precision, dimension(lin_kern_el_dim), intent(in):: A_self_prods, B_self_prods

call norm_lin_pos_restr_kernel_element(A_rescaled_wsqrt, B_rescaled_wsqrt,&
                    A_self_prods, B_self_prods, &
                    dimf, lin_kern_el_dim, calc_ders, single_sigma_der,&
                    el_wders(1:lin_kern_el_dim))

call linear_to_gauss_kernel(el_wders, inv_sq_sigma, kern_el_dim, calc_ders)

END SUBROUTINE

PURE SUBROUTINE linear_to_gauss_kernel(el_wders, inv_sq_sigma, kern_el_dim, calc_ders)
implicit none
integer, intent(in):: kern_el_dim
logical, intent(in):: calc_ders
double precision, intent(in):: inv_sq_sigma
double precision, intent(inout), dimension(kern_el_dim):: el_wders

if (calc_ders) then
    el_wders(3:kern_el_dim)=el_wders(2:kern_el_dim-1)*inv_sq_sigma
    el_wders(2)=(1.0-el_wders(1))*inv_sq_sigma
endif

el_wders(1)=exp(-inv_sq_sigma*(1.0-el_wders(1)))

if (calc_ders) el_wders(2:kern_el_dim)=el_wders(2:kern_el_dim)*el_wders(1)

END SUBROUTINE

PURE SUBROUTINE lin_pos_restr_kernel_element_self(A_rescaled_wsqrt,&
                    dimf, kern_el_dim, calc_ders, single_sigma, el_wders)
implicit none
integer, intent(in):: dimf, kern_el_dim
double precision, dimension(2, dimf), intent(in):: A_rescaled_wsqrt
logical, intent(in):: calc_ders, single_sigma
double precision, dimension(kern_el_dim), intent(inout):: el_wders

    call lin_pos_restr_kernel_element(A_rescaled_wsqrt,&
                    A_rescaled_wsqrt, dimf, kern_el_dim,&
                    calc_ders, single_sigma, el_wders)

END SUBROUTINE


PURE SUBROUTINE lin_pos_restr_kernel_element(A_rescaled_wsqrt,&
                    B_rescaled_wsqrt, dimf, kern_el_dim, calc_ders, single_sigma, el_wders)
implicit none
integer, intent(in):: dimf, kern_el_dim
logical, intent(in):: calc_ders, single_sigma
double precision, dimension(2, dimf), intent(in):: A_rescaled_wsqrt, B_rescaled_wsqrt
double precision, dimension(kern_el_dim), intent(inout):: el_wders
integer:: dim_id
double precision:: sqdist1, sqdist2, val_contribution, log_der_contribution

el_wders(1)=1.0
if (calc_ders) el_wders(2:kern_el_dim)=0.0
do dim_id=1, dimf
    sqdist1=A_rescaled_wsqrt(1, dim_id)+B_rescaled_wsqrt(1, dim_id)-2.0*A_rescaled_wsqrt(2, dim_id)*B_rescaled_wsqrt(2, dim_id)
    sqdist2=A_rescaled_wsqrt(1, dim_id)+B_rescaled_wsqrt(1, dim_id)+2.0*A_rescaled_wsqrt(2, dim_id)*B_rescaled_wsqrt(2, dim_id)
    val_contribution=(1.0+sqdist1)**(-2)+(1.0+sqdist2)**(-2)
    el_wders(1)=el_wders(1)*val_contribution
    if (calc_ders) then
        log_der_contribution=-2*(sqdist1*(1.0+sqdist1)**(-3)+sqdist2*(1.0+sqdist2)**(-3))/val_contribution
        if (single_sigma) then
            el_wders(2)=el_wders(2)+log_der_contribution
        else
            el_wders(1+dim_id)=log_der_contribution
        endif
    endif
enddo

END SUBROUTINE


PURE SUBROUTINE norm_lin_pos_restr_kernel_element(A_rescaled_wsqrt,&
                    B_rescaled_wsqrt, A_self_prods, B_self_prods, dimf, kern_el_dim, calc_ders, single_sigma_der,&
                    el_wders)
implicit none
integer, intent(in):: dimf, kern_el_dim
logical, intent(in):: calc_ders, single_sigma_der
double precision, dimension(2, dimf), intent(in):: A_rescaled_wsqrt, B_rescaled_wsqrt
double precision, dimension(kern_el_dim), intent(inout):: el_wders
double precision, dimension(kern_el_dim), intent(in):: A_self_prods, B_self_prods

call lin_pos_restr_kernel_element(A_rescaled_wsqrt, B_rescaled_wsqrt,&
                            dimf, kern_el_dim, calc_ders, single_sigma_der, el_wders)

el_wders(1)=el_wders(1)/A_self_prods(1)/B_self_prods(1)
if (calc_ders) then
    el_wders(2:kern_el_dim)=el_wders(2:kern_el_dim)&
                    -A_self_prods(2:kern_el_dim)-B_self_prods(2:kern_el_dim)
    el_wders(2:kern_el_dim)=el_wders(2:kern_el_dim)*el_wders(1)
endif
END SUBROUTINE


END MODULE
