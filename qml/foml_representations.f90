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


SUBROUTINE fgen_ibo_atom_scalar_rep(atom_id, atom_list, coeffs, atom_ao_ranges,&
                                        angular_momenta, ovlp_mat, coup_mats, num_ibo_atoms,&
                                        num_coup_mats, max_ang_mom, num_aos, tot_num_atoms,&
                                        scalar_reps, rho_val)
implicit none
integer, intent(in):: atom_id
integer, dimension(:), intent(in):: atom_list
double precision, dimension(:), intent(in):: coeffs
integer, dimension(:, :), intent(in):: atom_ao_ranges
integer, dimension(:), intent(in):: angular_momenta
double precision, dimension(:, :), intent(in):: ovlp_mat
double precision, dimension(:, :, :), intent(in):: coup_mats
integer, intent(in):: num_ibo_atoms, num_coup_mats, max_ang_mom, num_aos, tot_num_atoms
double precision, intent(inout), dimension(:):: scalar_reps
double precision, dimension(1), intent(inout):: rho_val
integer:: ang_mom1, ang_mom2, mat_counter, cur_array_position,&
            same_atom_check, other_atom_counter, other_atom_id,&
            true_atom_id

    true_atom_id=atom_id+1
    cur_array_position=1
    scalar_reps=0.0
    ! First generate the angular momentum distribution descriptors.
    do ang_mom1=1, max_ang_mom
        call add_ibo_atom_coupling(scalar_reps(cur_array_position),&
                true_atom_id, true_atom_id, ang_mom1, ang_mom1,&
                atom_ao_ranges, coeffs, angular_momenta,&
                ovlp_mat, tot_num_atoms, num_aos)
        cur_array_position=cur_array_position+1
    enddo
    rho_val=sum(scalar_reps)
    call add_ibo_atom_coupling(rho_val(1), true_atom_id, true_atom_id,&
                0, 0, atom_ao_ranges, coeffs, angular_momenta,&
                ovlp_mat, tot_num_atoms, num_aos)
    do mat_counter=1, num_coup_mats
        do same_atom_check=0, 1
            do ang_mom1=0, max_ang_mom
                do ang_mom2=0, max_ang_mom
                    if ((same_atom_check==0).and.(ang_mom1 > ang_mom2)) cycle
                    if (same_atom_check==0) then
                        call add_ibo_atom_coupling(scalar_reps(cur_array_position),&
                                    true_atom_id, true_atom_id, ang_mom1, ang_mom2,&
                                    atom_ao_ranges, coeffs, angular_momenta,&
                                    coup_mats(:, :, mat_counter), tot_num_atoms, num_aos)
                    else
                        do other_atom_counter=1, num_ibo_atoms
                            other_atom_id=atom_list(other_atom_counter)
                            if (other_atom_id /= atom_id) &
                                call add_ibo_atom_coupling(scalar_reps(cur_array_position), true_atom_id,&
                                    other_atom_id+1, ang_mom1, ang_mom2, atom_ao_ranges,&
                                    coeffs, angular_momenta, coup_mats(:, :, mat_counter),&
                                    tot_num_atoms, num_aos)
                            
                        enddo
                    endif
                    cur_array_position=cur_array_position+1
                enddo
            enddo
        enddo
    enddo

END SUBROUTINE

SUBROUTINE add_ibo_atom_coupling(cur_coupling_val, atom_id1, atom_id2, ang_mom1, ang_mom2,&
                                            atom_ao_ranges, coeffs, angular_momenta, mat,&
                                            tot_num_atoms, num_aos)
implicit none
integer, intent(in):: tot_num_atoms, num_aos
double precision, intent(inout):: cur_coupling_val
integer, intent(in):: atom_id1, atom_id2
integer, intent(in):: ang_mom1, ang_mom2
integer, dimension(2, tot_num_atoms), intent(in):: atom_ao_ranges
double precision, dimension(num_aos), intent(in):: coeffs
integer, dimension(num_aos), intent(in):: angular_momenta
double precision, dimension(num_aos, num_aos), intent(in):: mat
integer:: ao1, ao2

    do ao1=atom_ao_ranges(1, atom_id1)+1, atom_ao_ranges(2, atom_id1)
        if (angular_momenta(ao1)/=ang_mom1) cycle
        do ao2=atom_ao_ranges(1, atom_id2)+1, atom_ao_ranges(2, atom_id2)
            if (angular_momenta(ao2)/=ang_mom2) cycle
            cur_coupling_val=cur_coupling_val+mat(ao2, ao1)*coeffs(ao1)*coeffs(ao2)
        enddo
    enddo
END SUBROUTINE

!   hf_orb_coeffs - first index - basis function, second - molecular orbital.
SUBROUTINE fgen_fock_ft_coup_mats(inv_hf_orb_coeffs, hf_orb_energies, fbcm_delta_t,&
                                    num_orbs, num_fbcm_times, ft_fock_mats)
!   TO-DO why doesn't it work?
!use fconstants, only : pi
implicit none
double precision, parameter:: pi=3.14159265358979323846
integer, intent(in):: num_orbs, num_fbcm_times
double precision, intent(in), dimension(:, :):: inv_hf_orb_coeffs
double precision, intent(in), dimension(:):: hf_orb_energies
double precision, intent(in):: fbcm_delta_t
double precision, intent(inout), dimension(:, :, :):: ft_fock_mats
double precision:: propagation_time
integer:: freq_counter, cos_or_sin, mat_counter

    mat_counter=1
    do freq_counter=1, num_fbcm_times
        propagation_time=freq_counter*pi*fbcm_delta_t
        do cos_or_sin=0, 1
            call fgen_fock_ft_coup_mat(inv_hf_orb_coeffs, hf_orb_energies, propagation_time,&
                                            (cos_or_sin==0), num_orbs, ft_fock_mats(:, :, mat_counter))
            mat_counter=mat_counter+1
        enddo
    enddo

END SUBROUTINE

SUBROUTINE fgen_fock_ft_coup_mat(hf_orb_coeffs_transposed, hf_orb_energies,&
                                    propagation_time, use_cos, num_orbs, ft_fock_mat)
implicit none
integer, intent(in):: num_orbs
double precision, dimension(num_orbs, num_orbs), intent(in):: hf_orb_coeffs_transposed
double precision, dimension(num_orbs), intent(in):: hf_orb_energies
double precision, intent(in):: propagation_time
logical, intent(in):: use_cos
double precision, dimension(num_orbs, num_orbs), intent(inout):: ft_fock_mat
integer:: i1, i2
double precision, dimension(num_orbs):: propagator_coeffs

!$OMP PARALLEL DO
    do i1=1, num_orbs
!       Not %100 sure what option is best.
!        if (use_cos) then
!            propagator_coeffs(i1)=hf_orb_energies(i1)*cos(hf_orb_energies(i1)*characteristic_time)
!        else
!            propagator_coeffs(i1)=hf_orb_energies(i1)*sin(hf_orb_energies(i1)*characteristic_time)
!        endif
        if (use_cos) then
            propagator_coeffs(i1)=cos(hf_orb_energies(i1)*propagation_time)
        else
            propagator_coeffs(i1)=sin(hf_orb_energies(i1)*propagation_time)
        endif
    enddo
!$OMP END PARALLEL DO
!$OMP PARALLEL DO
    do i2=1, num_orbs
        do i1=1, i2
            call fgen_fock_ft_coup_el(hf_orb_coeffs_transposed(:, i1), hf_orb_coeffs_transposed(:, i2),&
                                        propagator_coeffs, num_orbs, ft_fock_mat(i1, i2))
        enddo
    enddo
!$OMP END PARALLEL DO
!$OMP PARALLEL DO
    do i2=1, num_orbs-1
        do i1=i2+1, num_orbs
            ft_fock_mat(i1, i2)=ft_fock_mat(i2, i1)
        enddo
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE

PURE SUBROUTINE fgen_fock_ft_coup_el(row1, row2, prop_coeffs, num_orbs, coup_el)
implicit none
integer, intent(in):: num_orbs
double precision, intent(in), dimension(num_orbs):: row1, row2, prop_coeffs
double precision, intent(inout):: coup_el
integer:: orb_counter

    coup_el=0.0
    do orb_counter=1, num_orbs
        coup_el=coup_el+row1(orb_counter)*row2(orb_counter)*prop_coeffs(orb_counter)
    enddo

END SUBROUTINE




