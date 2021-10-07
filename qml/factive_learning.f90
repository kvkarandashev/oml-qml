! Subroutines for the metadynamics-like approximation to active learning order.
SUBROUTINE fmetadynamics_active_learning_order(sym_kernel_mat, num_samples, initial_ordered_size,&
            num_to_generate, cov_num_tol, output_indices)
implicit none
double precision, dimension(:, :), intent(in):: sym_kernel_mat
integer, intent(in):: initial_ordered_size, num_to_generate, num_samples
double precision, intent(in):: cov_num_tol
integer, intent(inout), dimension(:):: output_indices
integer, dimension(num_samples):: ordered_indices
double precision, dimension(num_samples):: metadynamics_potential
integer:: i_sample, cur_ordered_size, cur_vec_true_id, next_addition
logical:: adding_points
double precision:: next_add_max_pot

    cur_ordered_size=initial_ordered_size
    if (cur_ordered_size/=0) ordered_indices(1:cur_ordered_size)=output_indices(1:cur_ordered_size)
    call fill_ordered_indices(ordered_indices, cur_ordered_size, num_samples)
    do i_sample=1, num_samples
        metadynamics_potential(i_sample)=sym_kernel_mat(i_sample, i_sample)
    enddo

    if (cur_ordered_size == 0) then
        call pick_least_covariant_points(sym_kernel_mat, ordered_indices, num_samples)
        cur_ordered_size=2
    endif

    adding_points=.false.
    do i_sample=1, num_to_generate
        if (adding_points) then
            call omp_max_shuffled_indices(metadynamics_potential, ordered_indices, cur_ordered_size,&
                        num_samples, next_addition, next_add_max_pot)
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

SUBROUTINE pick_least_covariant_points(sym_kernel_mat, ordered_indices, num_samples)
implicit none
integer, intent(in):: num_samples
double precision, dimension(num_samples, num_samples), intent(in):: sym_kernel_mat
integer, dimension(num_samples), intent(inout):: ordered_indices
double precision:: cur_min_covariance, tot_min_covariance, cur_covariance
integer, dimension(2):: cur_min_cov_pair, tot_min_cov_pair
integer:: i_sample1, i_sample2, init_point_id

    tot_min_covariance=-1.0
    tot_min_cov_pair=0

!$OMP PARALLEL PRIVATE(cur_min_cov_pair, cur_covariance, cur_min_covariance)
    cur_min_covariance=-1.0
    cur_min_cov_pair=0

!$OMP DO
    do i_sample2=1, num_samples
        do i_sample1=1, i_sample2-1
            cur_covariance=abs(sym_kernel_mat(i_sample1, i_sample2))
            if ((cur_covariance<cur_min_covariance).or.(cur_min_cov_pair(1)==0)) then
                cur_min_covariance=cur_covariance
                cur_min_cov_pair(1)=i_sample1
                cur_min_cov_pair(2)=i_sample2
            endif
        enddo
    enddo
!$OMP END DO
!$OMP CRITICAL
    if (((cur_min_covariance<tot_min_covariance).or.(tot_min_cov_pair(1)==0))&
                .and.(cur_min_cov_pair(1)/=0)) then
        tot_min_covariance=cur_min_covariance
        tot_min_cov_pair=cur_min_cov_pair
    endif
!$OMP END CRITICAL

!$OMP END PARALLEL
    do init_point_id=1, 2
        call switch_positions(ordered_indices, init_point_id, tot_min_cov_pair(init_point_id), num_samples)
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

SUBROUTINE omp_max_shuffled_indices(metadynamics_potential, ordered_indices,&
            ordered_size, num_samples, next_addition, next_add_pot)
implicit none
integer, intent(in):: num_samples, ordered_size
double precision, intent(in), dimension(num_samples):: metadynamics_potential
integer, intent(in), dimension(num_samples):: ordered_indices
integer, intent(inout):: next_addition
double precision, intent(inout):: next_add_pot
double precision:: cur_pot, cur_max_pot
integer:: cur_max_pos
integer:: i_sample

    next_add_pot=0.0
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
    if (((cur_max_pot>next_add_pot).or.(next_addition==0)).and.(cur_max_pos/=0)) then
        next_add_pot=cur_max_pot
        next_addition=cur_max_pos
    endif
!$OMP END CRITICAL
!$OMP END PARALLEL

END SUBROUTINE


! Subroutines for the feature distance-based learning order.
!   TO-DO generalizing to initial_ordered_size/=0 is required; should be possible with
!   Cholesky decomposition routines.

SUBROUTINE ffeature_distance_learning_order(sym_kernel_mat, num_samples, initial_ordered_size,&
                                            num_to_generate, cov_num_tol, orthog_sqnorm_tol, output_indices)
implicit none
integer, intent(in):: initial_ordered_size, num_to_generate, num_samples
double precision, dimension(:, :), intent(in):: sym_kernel_mat
double precision, intent(in):: cov_num_tol, orthog_sqnorm_tol
integer, intent(inout), dimension(:):: output_indices
integer, dimension(num_samples):: ordered_indices
double precision, dimension(num_samples):: residual_square_distances
double precision, dimension(num_to_generate, num_to_generate):: orthogonal_coord_coeffs
integer:: i_sample, cur_ordered_size, next_addition
logical:: adding_points, stop_orthogonalization, choosing_least_neg_cov
double precision, dimension(num_samples):: closest_neg_cov_vals
double precision:: next_add_least_neg_cov

    cur_ordered_size=initial_ordered_size

    if (cur_ordered_size/=0) ordered_indices(1:cur_ordered_size)=output_indices(1:cur_ordered_size)
    call fill_ordered_indices(ordered_indices, cur_ordered_size, num_samples)

    do i_sample=1, num_samples
        residual_square_distances(i_sample)=sym_kernel_mat(i_sample, i_sample)
    enddo
    closest_neg_cov_vals=0.0

    if (cur_ordered_size == 0) then
        call pick_least_covariant_points(sym_kernel_mat, ordered_indices, num_samples)
        cur_ordered_size=2
    endif
    orthogonal_coord_coeffs=0.0

    adding_points=.false.
    stop_orthogonalization=.false.
    choosing_least_neg_cov=.true.

    do i_sample=1, num_to_generate
        if (adding_points) then
            if (choosing_least_neg_cov) then
                call omp_max_shuffled_indices(closest_neg_cov_vals, ordered_indices, cur_ordered_size,&
                                                num_samples, next_addition, next_add_least_neg_cov)
                if (.not.stop_orthogonalization) &
                    choosing_least_neg_cov=(next_add_least_neg_cov>-cov_num_tol)
            endif
            if (.not.choosing_least_neg_cov) then
                call omp_max_shuffled_indices(residual_square_distances, ordered_indices, cur_ordered_size,&
                                        num_samples, next_addition, next_add_least_neg_cov)
            endif
            cur_ordered_size=cur_ordered_size+1
            call switch_positions(ordered_indices, cur_ordered_size, next_addition, num_samples)
        else
            adding_points=(i_sample==cur_ordered_size)
        endif
        if (.not.stop_orthogonalization) then
            call update_orthogonal_coord_coeffs(sym_kernel_mat, orthogonal_coord_coeffs,&
                                 i_sample, ordered_indices(1:i_sample), num_to_generate, num_samples,&
                                 orthog_sqnorm_tol, stop_orthogonalization)
            if (stop_orthogonalization) then
                choosing_least_neg_cov=.true.
            else
                call update_residual_square_distances(sym_kernel_mat, orthogonal_coord_coeffs(1:i_sample, i_sample),&
                    i_sample, ordered_indices, num_samples, residual_square_distances)
            endif
        endif
        call update_negative_closest_covariances(sym_kernel_mat, i_sample, ordered_indices, num_samples, closest_neg_cov_vals)
    enddo
    
    output_indices=ordered_indices(1:num_to_generate)-1

END SUBROUTINE

SUBROUTINE fill_ordered_indices(ordered_indices, initial_ordered_size, num_samples)
integer, intent(in):: num_samples, initial_ordered_size
integer, intent(inout), dimension(num_samples):: ordered_indices
integer:: ordered_part_walker, skipped_indices, cur_index

    ordered_part_walker=1
    skipped_indices=-initial_ordered_size
    do i_sample=initial_ordered_size+1, num_samples
        cur_index=i_sample+skipped_indices
        if ((cur_index==ordered_indices(ordered_part_walker)).and.&
                                (initial_ordered_size/=0)) then
            do
                ordered_part_walker=ordered_part_walker+1
                skipped_indices=skipped_indices+1
                cur_index=i_sample+skipped_indices
                if (ordered_part_walker>initial_ordered_size) then
                    ordered_part_walker=1
                    exit
                endif
                if (ordered_indices(ordered_part_walker)/=cur_index) exit
            enddo
        endif
        ordered_indices(i_sample)=cur_index
    enddo

END SUBROUTINE

SUBROUTINE update_orthogonal_coord_coeffs(sym_kernel_mat, orthogonal_coord_coeffs, cur_ordered_size,&
                            ordered_indices, num_to_generate, num_samples, sq_norm_tolerance, stop_orthogonalization)
implicit none
double precision, dimension(num_samples, num_samples):: sym_kernel_mat
integer, intent(in):: cur_ordered_size, num_samples, num_to_generate
double precision, dimension(num_to_generate, num_to_generate), intent(inout):: orthogonal_coord_coeffs
integer, dimension(cur_ordered_size), intent(in):: ordered_indices
double precision, intent(in):: sq_norm_tolerance
logical, intent(inout):: stop_orthogonalization
double precision, dimension(:), allocatable:: prev_coord_comps
integer:: latest_addition_true_id
integer:: i_coord, i_coord1
double precision:: sq_norm_coeff

    orthogonal_coord_coeffs(cur_ordered_size, cur_ordered_size)=1.0
    latest_addition_true_id=ordered_indices(cur_ordered_size)
    
    sq_norm_coeff=sym_kernel_mat(latest_addition_true_id, latest_addition_true_id)

    if (cur_ordered_size > 1) then
        allocate(prev_coord_comps(cur_ordered_size-1))
!$OMP PARALLEL DO PRIVATE(i_coord) SCHEDULE(DYNAMIC)
        do i_coord=1, cur_ordered_size-1
            call dot_product_with_orth_coord(sym_kernel_mat, orthogonal_coord_coeffs(1:i_coord, i_coord),&
                    ordered_indices(1:i_coord), i_coord, latest_addition_true_id, num_samples, prev_coord_comps(i_coord))
        enddo
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE (i_coord, i_coord1) SCHEDULE(DYNAMIC)
        do i_coord=1, cur_ordered_size-1
            do i_coord1=i_coord, cur_ordered_size-1
                orthogonal_coord_coeffs(i_coord, cur_ordered_size)=orthogonal_coord_coeffs(i_coord, cur_ordered_size)&
                                                    -prev_coord_comps(i_coord1)*orthogonal_coord_coeffs(i_coord, i_coord1)
            enddo
        enddo
!$OMP END PARALLEL DO
        sq_norm_coeff=sq_norm_coeff-sum(prev_coord_comps**2)
    endif
    if (sq_norm_coeff<sq_norm_tolerance) then
        print *, 'WARNING: negligible sq_norm_coeff during orthogonalization'
        stop_orthogonalization=.true.
    else
        orthogonal_coord_coeffs(1:cur_ordered_size, cur_ordered_size)=&
                orthogonal_coord_coeffs(1:cur_ordered_size, cur_ordered_size)/sqrt(sq_norm_coeff)
    endif

END SUBROUTINE

PURE SUBROUTINE dot_product_with_orth_coord(sym_kernel_mat, coord_coeffs, true_indices,&
                num_coeffs, true_other_id, num_samples, dot_prod)
implicit none
integer, intent(in):: num_samples, num_coeffs
double precision, intent(in), dimension(num_samples, num_samples):: sym_kernel_mat
double precision, intent(in), dimension(num_coeffs):: coord_coeffs
integer, intent(in), dimension(num_coeffs):: true_indices
integer, intent(in):: true_other_id
double precision, intent(inout):: dot_prod
integer:: i_coord

    dot_prod=0.0
    do i_coord=1, num_coeffs
        dot_prod=dot_prod+sym_kernel_mat(true_indices(i_coord), true_other_id)*coord_coeffs(i_coord)
    enddo

END SUBROUTINE

SUBROUTINE update_residual_square_distances(sym_kernel_mat, orthogonal_coord_coeffs, cur_ordered_size,&
                                    ordered_indices, num_samples, residual_square_distances)
implicit none
integer, intent(in):: cur_ordered_size, num_samples
double precision, dimension(num_samples, num_samples), intent(in):: sym_kernel_mat
double precision, dimension(cur_ordered_size), intent(in):: orthogonal_coord_coeffs
integer, dimension(num_samples), intent(in):: ordered_indices
double precision, dimension(num_samples), intent(inout):: residual_square_distances
double precision:: cur_dot_product
integer:: i_sample, true_sample_id

!$OMP PARALLEL DO PRIVATE (i_sample, true_sample_id, cur_dot_product) SCHEDULE(DYNAMIC)
    do i_sample=cur_ordered_size, num_samples
!       We don't need to do this for i_sample=cur_ordered_size, it's just to check during debugging
!       that residual_square_distances(ordered_indices(cur_ordered_size))==0.0 after the procedure.
        true_sample_id=ordered_indices(i_sample)
        call dot_product_with_orth_coord(sym_kernel_mat, orthogonal_coord_coeffs,&
                    ordered_indices(1:cur_ordered_size), cur_ordered_size, true_sample_id, num_samples, cur_dot_product)
        residual_square_distances(true_sample_id)=residual_square_distances(true_sample_id)-cur_dot_product**2
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE

                         
SUBROUTINE update_negative_closest_covariances(sym_kernel_mat, last_added_sample,&
                                    ordered_indices, num_samples, closest_neg_cov_vals)
implicit none
integer, intent(in):: num_samples, last_added_sample
double precision, intent(inout), dimension(num_samples):: closest_neg_cov_vals
double precision, intent(in), dimension(num_samples, num_samples):: sym_kernel_mat
integer, dimension(num_samples), intent(in):: ordered_indices
integer:: i_sample, true_sample_id, true_last_added_id
double precision:: added_neg_cov

    true_last_added_id=ordered_indices(last_added_sample)
!$OMP PARALLEL DO PRIVATE(i_sample, true_sample_id, added_neg_cov)
    do i_sample=last_added_sample, num_samples
        true_sample_id=ordered_indices(i_sample)
        added_neg_cov=-abs(sym_kernel_mat(true_sample_id, true_last_added_id))
        if (added_neg_cov<closest_neg_cov_vals(true_sample_id)) then
            closest_neg_cov_vals(true_sample_id)=added_neg_cov
        endif
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE

! Deleting linear dependent entries.

SUBROUTINE flinear_dependent_entries(sym_kernel_mat, num_elements, residue_tol_coeff, indices_to_ignore)
implicit none
integer, intent(in):: num_elements
double precision, intent(in), dimension(:, :):: sym_kernel_mat
double precision, intent(in):: residue_tol_coeff
integer, intent(inout), dimension(:):: indices_to_ignore
double precision, dimension(num_elements):: sqnorm_residue, residue_tolerance
logical, dimension(num_elements):: to_ignore, considered
double precision, dimension(num_elements, num_elements):: orthonormalized_vectors
integer:: i, j, cur_orth_id
double precision:: cur_norm, cur_product
integer, dimension(num_elements):: orthonorm_order
double precision:: proc_min_residue, global_min_residue
integer:: proc_min_res_id, global_min_res_id
logical:: proc_min_res_init, global_min_res_init
logical:: matrix_unstable
integer:: true_j

    matrix_unstable=.False.

    global_min_res_init=.False.

    orthonormalized_vectors=0.0
    considered=.False.
!$OMP PARALLEL PRIVATE(i, cur_norm, proc_min_res_id, proc_min_res_init, proc_min_residue)

    proc_min_res_init=.False.

!$OMP DO
    do i=1, num_elements
        orthonormalized_vectors(i, i)=1.0

        cur_norm=sym_kernel_mat(i, i)
        sqnorm_residue(i)=cur_norm
        residue_tolerance(i)=cur_norm*residue_tol_coeff

        if ((.not.proc_min_res_init).or.(proc_min_residue<cur_norm)) then
            proc_min_residue=cur_norm
            proc_min_res_id=i
            proc_min_res_init=.True.
        endif
    enddo
!$OMP END DO

    if (proc_min_res_init) then
!$OMP CRITICAL
        if ((.not.global_min_res_init).or.(global_min_residue<proc_min_residue)) then
            global_min_residue=proc_min_residue
            global_min_res_id=proc_min_res_id
            global_min_res_init=.True.
        endif
!$OMP END CRITICAL
    endif

!$OMP END PARALLEL

    cur_orth_id=1

    do
        orthonorm_order(cur_orth_id)=global_min_res_id
        considered(global_min_res_id)=.True.
        ! Normalize the vector.
        cur_norm=sqrt(global_min_residue)
!$OMP PARALLEL DO PRIVATE(j, true_j)
        do j=1, cur_orth_id
            true_j=orthonorm_order(j)
            orthonormalized_vectors(true_j, global_min_res_id)=&
                orthonormalized_vectors(true_j, global_min_res_id)/cur_norm
        enddo
!$OMP END PARALLEL DO
        ! Subtract projections of the normalized vector from all currently not orthonormalized vectors.
        ! Also check that their residue is above the corresponding threshold.

        global_min_res_init=.False.
!$OMP PARALLEL PRIVATE(i, j, cur_product, proc_min_res_id, proc_min_res_init, proc_min_residue, cur_norm, true_j)

        proc_min_res_init=.False.

!$OMP DO
        do i=1, num_elements
            if (.not.considered(i)) then
                cur_product=0.0
                do j=1, cur_orth_id
                    true_j=orthonorm_order(j)
                    cur_product=cur_product+sym_kernel_mat(true_j, i)&
                            *orthonormalized_vectors(true_j, global_min_res_id)
                enddo
                sqnorm_residue(i)=sqnorm_residue(i)-cur_product**2
                cur_norm=sqnorm_residue(i)
                if (cur_norm<residue_tolerance(i)) then
                    considered(i)=.True.
                    if (cur_norm<-residue_tolerance(i)) then
!$OMP CRITICAL
                        matrix_unstable=.True.
!$OMP END CRITICAL
                    endif
                else
                    do j=1, cur_orth_id
                        true_j=orthonorm_order(j)
                        orthonormalized_vectors(true_j, i)=&
                            orthonormalized_vectors(true_j, i)-cur_product*&
                            orthonormalized_vectors(true_j, global_min_res_id)
                    enddo
                    if ((.not.proc_min_res_init).or.(proc_min_residue<cur_norm)) then
                        proc_min_residue=cur_norm
                        proc_min_res_id=i
                        proc_min_res_init=.True.
                    endif
                endif
            endif
        enddo
!$OMP END DO

        if (proc_min_res_init) then
!$OMP CRITICAL
            if ((.not.global_min_res_init).or.(global_min_residue<proc_min_residue)) then
                global_min_residue=proc_min_residue
                global_min_res_id=proc_min_res_id
                global_min_res_init=.True.
            endif
!$OMP END CRITICAL
        endif

!$OMP END PARALLEL
        if (matrix_unstable) then
            indices_to_ignore(1)=-1
            return
        endif
        if (.not.global_min_res_init) exit
        cur_orth_id=cur_orth_id+1
    enddo


    to_ignore=.True.
!$OMP PARALLEL DO PRIVATE(i)
    do i=1, cur_orth_id
        to_ignore(orthonorm_order(i))=.False.
    enddo
!$OMP END PARALLEL DO
    indices_to_ignore=0
    ! Create a list with ignored indices.
    j=1
    do i=1, num_elements
        if (to_ignore(i)) then
            print *, 'Skipped:', i-1, 'residue:', sqnorm_residue(i) ! Subtracting 1 because later used in python scripts.
            indices_to_ignore(j)=i-1 ! Subtracting 1 because later used in python scripts.
            j=j+1
        endif
    enddo
END SUBROUTINE
