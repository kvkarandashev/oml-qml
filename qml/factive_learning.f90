

SUBROUTINE fmetadynamics_active_learning_order(sym_kernel_mat, num_samples, initial_ordered_size,&
            num_to_generate, output_indices)
implicit none
double precision, dimension(:, :), intent(in):: sym_kernel_mat
integer, intent(in):: initial_ordered_size, num_to_generate, num_samples
integer, intent(inout), dimension(:):: output_indices
integer, dimension(num_samples):: ordered_indices
double precision, dimension(num_samples):: metadynamics_potential
integer:: i_sample, cur_ordered_size, cur_vec_true_id, next_addition
logical:: adding_points

    cur_ordered_size=initial_ordered_size
    do i_sample=1, num_samples
        ordered_indices(i_sample)=i_sample
        metadynamics_potential(i_sample)=sym_kernel_mat(i_sample, i_sample)
    enddo

    if (cur_ordered_size == 0) then
        call pick_furthest_distance_points(sym_kernel_mat, ordered_indices, num_samples)
        cur_ordered_size=2
    endif

    adding_points=.false.
    do i_sample=1, num_to_generate
        if (adding_points) then
            call omp_min_shuffled_indices(metadynamics_potential, ordered_indices, cur_ordered_size, num_samples, next_addition)
            cur_ordered_size=cur_ordered_size+1
            call switch_positions(ordered_indices, cur_ordered_size, next_addition, num_samples)
        else
            adding_points=(i_sample==cur_ordered_size)
        endif
        cur_vec_true_id=ordered_indices(i_sample)
        metadynamics_potential=metadynamics_potential-sym_kernel_mat(:, cur_vec_true_id)**2&
                            /sym_kernel_mat(cur_vec_true_id, cur_vec_true_id)
    enddo

    output_indices=ordered_indices(1:num_to_generate)-1 ! The -1 is because we are going to use this in Python.

END SUBROUTINE fmetadynamics_active_learning_order

SUBROUTINE pick_furthest_distance_points(sym_kernel_mat, ordered_indices, num_samples)
implicit none
integer, intent(in):: num_samples
double precision, dimension(num_samples, num_samples), intent(in):: sym_kernel_mat
integer, dimension(num_samples), intent(inout):: ordered_indices
double precision:: cur_max_sqdistance, tot_max_sqdistance, cur_sqdistance
integer, dimension(2):: cur_max_dist_pair, tot_max_dist_pair
integer:: i_sample1, i_sample2, init_point_id

    tot_max_sqdistance=-1.0
    tot_max_dist_pair(1)=1
    tot_max_dist_pair(2)=2

!$OMP PARALLEL PRIVATE(cur_max_sqdistance, cur_sqdistance, cur_max_dist_pair)
    cur_max_sqdistance=-1.0
    cur_max_dist_pair=0

!$OMP DO
    do i_sample1=1, num_samples
        do i_sample2=1, i_sample1-1
            cur_sqdistance=sym_kernel_mat(i_sample1, i_sample1)+sym_kernel_mat(i_sample2, i_sample2)&
                        -2*sym_kernel_mat(i_sample1, i_sample2)
            if ((cur_sqdistance>cur_max_sqdistance).or.(cur_max_dist_pair(1)==0)) then
                cur_max_sqdistance=cur_sqdistance
                cur_max_dist_pair(1)=i_sample2
                cur_max_dist_pair(2)=i_sample1
            endif
        enddo
    enddo
!$OMP END DO
!$OMP CRITICAL
    if (cur_max_sqdistance>tot_max_sqdistance) then
        tot_max_sqdistance=cur_max_sqdistance
        tot_max_dist_pair=cur_max_dist_pair
    endif
!$OMP END CRITICAL

!$OMP END PARALLEL

    do init_point_id=1, 2
        call switch_positions(ordered_indices, init_point_id, tot_max_dist_pair(init_point_id), num_samples)
    enddo

END SUBROUTINE


SUBROUTINE switch_positions(partially_ordered_array, position1, position2, array_length)
implicit none
integer, intent(in):: array_length
integer, intent(inout), dimension(array_length):: partially_ordered_array
integer, intent(in):: position1, position2
integer:: temp_val

    if (position1 /= position2) then
        temp_val=partially_ordered_array(position1)
        partially_ordered_array(position1)=partially_ordered_array(position2)
        partially_ordered_array(position2)=temp_val
    endif

END SUBROUTINE

SUBROUTINE omp_min_shuffled_indices(metadynamics_potential, ordered_indices, ordered_size, num_samples, next_addition)
implicit none
integer, intent(in):: num_samples, ordered_size
double precision, intent(in), dimension(num_samples):: metadynamics_potential
integer, intent(in), dimension(num_samples):: ordered_indices
integer, intent(inout):: next_addition
double precision:: cur_pot, cur_max_pot, tot_max_pot
integer:: cur_max_pos
integer:: i_sample

    tot_max_pot=0.0
    next_addition=0
!$OMP PARALLEL PRIVATE (cur_pot, cur_max_pot, cur_max_pos, i_sample)
    cur_max_pot=0.0
    cur_max_pos=0
!$OMP DO
    do i_sample=ordered_size+1, num_samples
        cur_pot=metadynamics_potential(ordered_indices(i_sample))
        if ((cur_pot>cur_max_pot).or.(cur_max_pos==0)) then
            cur_max_pot=cur_pot
            cur_max_pos=i_sample
        endif
    enddo
!$OMP END DO
!$OMP CRITICAL
    if (((cur_max_pot>tot_max_pot).or.(next_addition==0)).and.(cur_max_pos/=0)) then
        tot_max_pot=cur_max_pot
        next_addition=cur_max_pos
    endif
!$OMP END CRITICAL
!$OMP END PARALLEL

END SUBROUTINE
