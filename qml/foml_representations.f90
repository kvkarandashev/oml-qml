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

SUBROUTINE fgen_ibo_global_couplings(ibo_coeffs, ibo_id, angular_momenta, max_ang_mom,&
                            num_aos, coup_mats, num_coup_mats, scalar_reps)
implicit none
double precision, dimension(:, :), intent(in):: ibo_coeffs
integer, intent(in):: ibo_id
double precision, dimension(:, :, :), intent(in):: coup_mats
integer, dimension(:), intent(in):: angular_momenta
integer, intent(in):: max_ang_mom, num_aos
double precision, dimension(:), intent(inout):: scalar_reps
integer:: cur_array_position_other, ang_mom1, ang_mom2, index_to_add,&
                    mat_counter, num_coup_mats, num_ibos, other_ibo_id
logical:: self_term_present

    cur_array_position_other=1
    scalar_reps=0.0

    do mat_counter=1, num_coup_mats
        do ang_mom1=0, max_ang_mom
            do ang_mom2=0, max_ang_mom
                self_term_present=(ang_mom1<=ang_mom2)
                do other_ibo_id=1, num_ibos
                    if (other_ibo_id==ibo_id) then
                        if (self_term_present) then
                            index_to_add=cur_array_position_other+1
                        else
                            cycle
                        endif
                    else
                        index_to_add=cur_array_position_other
                    endif
                    call add_ibo_ibo_sq_coupling(ibo_coeffs(:, other_ibo_id), ibo_coeffs(:, ibo_id),&
                            coup_mats(:, :, mat_counter), angular_momenta, ang_mom1, ang_mom2, num_aos,&
                            scalar_reps(index_to_add))
                enddo
                if (self_term_present) then
                    cur_array_position_other=cur_array_position_other+2
                else
                    cur_array_position_other=cur_array_position_other+1
                endif
            enddo
        enddo
    enddo

END SUBROUTINE

SUBROUTINE add_ibo_ibo_sq_coupling(ibo_coeffs1, ibo_coeffs2, coup_mat, angular_momenta,&
                                ang_mom1, ang_mom2, num_aos, added_output)
integer, intent(in):: num_aos, ang_mom1, ang_mom2
integer, intent(in), dimension(num_aos):: angular_momenta
double precision, intent(in), dimension(num_aos):: ibo_coeffs1, ibo_coeffs2
double precision, intent(in), dimension(num_aos, num_aos):: coup_mat
double precision, intent(inout):: added_output
integer:: aos_id1, aos_id2
double precision:: coupling

    coupling=0.0
    do aos_id1=1, num_aos
        if (angular_momenta(aos_id1)/=ang_mom1) cycle
        do aos_id2=1, num_aos
            if (angular_momenta(aos_id2)/=ang_mom2) cycle
            coupling=coupling+ibo_coeffs1(aos_id1)*ibo_coeffs2(aos_id2)*coup_mat(aos_id1, aos_id2)
        enddo
    enddo
    added_output=added_output+coupling**2

END SUBROUTINE


SUBROUTINE fang_mom_descr(atom_ao_range, coeffs, angular_momenta, ovlp_mat, max_ang_mom, num_aos, scalar_reps, rho_val)
implicit none
integer, dimension(:), intent(in):: atom_ao_range
double precision, dimension(:), intent(in):: coeffs
integer, dimension(:), intent(in):: angular_momenta
double precision, dimension(:, :), intent(in):: ovlp_mat
integer, intent(in):: max_ang_mom, num_aos
double precision, intent(inout), dimension(:):: scalar_reps
double precision, dimension(1), intent(inout):: rho_val
integer:: ang_mom

    scalar_reps=0.0
    ! First generate the angular momentum distribution descriptors.
    do ang_mom=1, max_ang_mom
        call add_ibo_atom_coupling(scalar_reps(ang_mom),&
                atom_ao_range, 0, ang_mom, ang_mom,&
                coeffs, angular_momenta, ovlp_mat, num_aos)
    enddo
    rho_val=sum(scalar_reps)
    call add_ibo_atom_coupling(rho_val(1), atom_ao_range, 0, 0, 0, coeffs, angular_momenta, ovlp_mat, num_aos)

                                        
END SUBROUTINE

SUBROUTINE fgen_ibo_atom_scalar_rep(atom_ao_range, coeffs, angular_momenta, coup_mats,&
                                        num_coup_mats, max_ang_mom, num_aos, scalar_reps)
implicit none
integer, intent(in):: max_ang_mom, num_aos, num_coup_mats
integer, dimension(:, :), intent(in):: atom_ao_range
double precision, dimension(:), intent(in):: coeffs
integer, dimension(:), intent(in):: angular_momenta
double precision, dimension(:, :, :), intent(in):: coup_mats
double precision, intent(inout), dimension(:):: scalar_reps
integer:: ang_mom1, ang_mom2, mat_counter, cur_array_position, same_atom_check

    cur_array_position=1
    do mat_counter=1, num_coup_mats
        do same_atom_check=0, 1
            do ang_mom1=0, max_ang_mom
                do ang_mom2=0, max_ang_mom
                    if ((same_atom_check==0).and.(ang_mom1 > ang_mom2)) cycle
                    call add_ibo_atom_coupling(scalar_reps(cur_array_position),&
                                atom_ao_range, same_atom_check, ang_mom1, ang_mom2,&
                                coeffs, angular_momenta, coup_mats(:, :, mat_counter), num_aos)
                    cur_array_position=cur_array_position+1
                enddo
            enddo
        enddo
    enddo
END SUBROUTINE

SUBROUTINE add_ibo_atom_coupling(cur_coupling_val, atom_ao_range, same_atom, ang_mom1, ang_mom2,&
                                            coeffs, angular_momenta, mat, num_aos)
implicit none
integer, intent(in):: num_aos
double precision, intent(inout):: cur_coupling_val
integer, intent(in):: ang_mom1, ang_mom2, same_atom
integer, dimension(2), intent(in):: atom_ao_range
double precision, dimension(num_aos), intent(in):: coeffs
integer, dimension(num_aos), intent(in):: angular_momenta
double precision, dimension(num_aos, num_aos), intent(in):: mat
integer:: ao1


    do ao1=atom_ao_range(1)+1, atom_ao_range(2)
        if (angular_momenta(ao1)/=ang_mom1) cycle
        if (same_atom==0) then
            call add_aos_coupling(cur_coupling_val, ao1, mat, coeffs, angular_momenta,&
                                            ang_mom2, atom_ao_range(1)+1, atom_ao_range(2), num_aos)
        else
            call add_aos_coupling(cur_coupling_val, ao1, mat, coeffs, angular_momenta,&
                                                        ang_mom2, 1, atom_ao_range(1), num_aos)
            call add_aos_coupling(cur_coupling_val, ao1, mat, coeffs, angular_momenta,&
                                                ang_mom2, atom_ao_range(2)+1, num_aos, num_aos)
        endif
    enddo
END SUBROUTINE

SUBROUTINE add_aos_coupling(cur_coupling_val, ao1, mat, coeffs, angular_momenta, ang_mom2,&
                                         lower_index, upper_index, num_aos)
integer, intent(in):: num_aos
double precision, intent(inout):: cur_coupling_val
integer, intent(in):: ang_mom2, ao1
double precision, dimension(num_aos), intent(in):: coeffs
integer, dimension(num_aos), intent(in):: angular_momenta
double precision, dimension(num_aos, num_aos), intent(in):: mat
integer, intent(in):: lower_index, upper_index
integer:: ao2

        do ao2=lower_index, upper_index
            if (angular_momenta(ao2)/=ang_mom2) cycle
            cur_coupling_val=cur_coupling_val+mat(ao2, ao1)*coeffs(ao1)*coeffs(ao2)
        enddo

END SUBROUTINE

!   orb_coeffs - first index - basis function, second - molecular orbital.
SUBROUTINE fgen_ft_coup_mats(inv_orb_coeffs, orb_energies, prop_delta_t,&
                                    num_orbs, num_prop_times, ft_mats)
implicit none
integer, intent(in):: num_orbs, num_prop_times
double precision, intent(in), dimension(:, :):: inv_orb_coeffs
double precision, intent(in), dimension(:):: orb_energies
double precision, intent(in):: prop_delta_t
double precision, intent(inout), dimension(:, :, :):: ft_mats
double precision:: propagation_time
integer:: time_point_counter, cos_or_sin, mat_counter

    mat_counter=1
    do time_point_counter=1, num_prop_times
        propagation_time=time_point_counter*prop_delta_t
        do cos_or_sin=0, 1
            call fgen_ft_coup_mat(inv_orb_coeffs, orb_energies, propagation_time,&
                                   (cos_or_sin==0), num_orbs, ft_mats(:, :, mat_counter))
            mat_counter=mat_counter+1
        enddo
    enddo

END SUBROUTINE

SUBROUTINE fgen_ft_coup_mat(inv_orb_coeffs, orb_energies,&
                                    propagation_time, use_cos, num_orbs, ft_mat)
implicit none
integer, intent(in):: num_orbs
double precision, dimension(num_orbs, num_orbs), intent(in):: inv_orb_coeffs
double precision, dimension(num_orbs), intent(in):: orb_energies
double precision, intent(in):: propagation_time
logical, intent(in):: use_cos
double precision, dimension(num_orbs, num_orbs), intent(inout):: ft_mat
integer:: i1, i2, orb_counter
double precision, dimension(num_orbs):: propagator_coeffs

!$OMP PARALLEL DO
    do i1=1, num_orbs
        if (use_cos) then
            propagator_coeffs(i1)=cos(orb_energies(i1)*propagation_time)
        else
            propagator_coeffs(i1)=sin(orb_energies(i1)*propagation_time)
        endif
    enddo
!$OMP END PARALLEL DO
ft_mat=0.0
!$OMP PARALLEL DO
    do i2=1, num_orbs
        do i1=1, i2
            do orb_counter=1, num_orbs
                ft_mat(i1, i2)=ft_mat(i1, i2)+propagator_coeffs(orb_counter)*&
                    inv_orb_coeffs(orb_counter, i1)*inv_orb_coeffs(orb_counter, i2)
            enddo
        enddo
    enddo
!$OMP END PARALLEL DO
!$OMP PARALLEL DO
    do i2=1, num_orbs-1
        do i1=i2+1, num_orbs
            ft_mat(i1, i2)=ft_mat(i2, i1)
        enddo
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE



