subroutine hru_output(ihru)
    use plant_module
    use plant_data_module
    use time_module
    use basin_module
    use output_landscape_module
    use hydrograph_module, only : sp_ob1, ob
    use organic_mineral_mass_module
    use soil_module
    use hru_module, only : hru
    use hdf5_module   ! This imports dataset_ids
    use mpi

    implicit none

    integer, intent(in) :: ihru
    integer :: idp, j, iob, ipl, mpi_rank, mpi_size, mpi_err
    real :: const, sw_init, sno_init
    integer :: hdferr  ! Error code for HDF5 operations
    integer(hid_t) :: dataset_id, dataspace_id, memspace_id, plist_id  ! Dataspace identifiers
    integer(hsize_t), dimension(2) :: start, count, max_dims  ! Dimensions and hyperslab information
    integer :: year_idx, i
    integer(hsize_t), dimension(1) :: dims   ! Dimension size

    ! Array for holding field values to write
    real(8), dimension(40) :: values_to_write

    ! Initialize MPI ranks
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, mpi_err)
    call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, mpi_err)

    j = ihru
    iob = sp_ob1%hru + j - 1   ! Added for new output write

    ! Monthly calculations...
    hwb_m(j) = hwb_m(j) + hwb_d(j)
    hnb_m(j) = hnb_m(j) + hnb_d(j)
    hls_m(j) = hls_m(j) + hls_d(j)
    hpw_m(j) = hpw_m(j) + hpw_d(j)

    hwb_d(j)%sw_final = soil(j)%sw
    hwb_d(j)%sw = (hwb_d(j)%sw_init + hwb_d(j)%sw_final) / 2.
    hwb_d(j)%sno_final = hru(j)%sno_mm
    hwb_d(j)%snopack = (hwb_d(j)%sno_init + hwb_d(j)%sno_final) / 2.

    hwb_d(j)%sw_init = hwb_d(j)%sw_final
    hwb_d(j)%sno_init = hwb_d(j)%sno_final

    ! End of month processing
    if (time%end_mo == 1) then
        hwb_y(j) = hwb_y(j) + hwb_m(j)
        hnb_y(j) = hnb_y(j) + hnb_m(j)
        hls_y(j) = hls_y(j) + hls_m(j)
        hpw_y(j) = hpw_y(j) + hpw_m(j)

        const = float(ndays(time%mo + 1) - ndays(time%mo))
        hwb_m(j) = hwb_m(j) / const
        hpw_m(j) = hpw_m(j) / const

        hwb_m(j)%sw_final = hwb_d(j)%sw_final
        hwb_m(j)%sno_final = hwb_d(j)%sno_final

        sw_init = hwb_m(j)%sw_final
        sno_init = hwb_m(j)%sno_final
        hwb_m(j)%sw_init = sw_init
        hwb_m(j)%sno_init = sno_init
        hnb_m(j) = hnbz
        hpw_m(j) = hpwz
        hls_m(j) = hlsz
    end if

    ! End of year processing
    if (time%end_yr == 1) then
        hwb_a(j) = hwb_a(j) + hwb_y(j)
        hnb_a(j) = hnb_a(j) + hnb_y(j)
        hls_a(j) = hls_a(j) + hls_y(j)
        hpw_a(j) = hpw_a(j) + hpw_y(j)

        const = time%day_end_yr
        hwb_y(j) = hwb_y(j) / const
        hpw_y(j) = hpw_y(j) / const

        hwb_y(j)%sw_final = hwb_d(j)%sw_final
        hwb_y(j)%sno_final = hwb_d(j)%sno_final

        ! Calculate the current year index relative to the start year
        year_idx = time%yrc - time%yrc_start + 1

        ! Load values into the array for writing
        values_to_write(1)  = hwb_y(j)%precip
        values_to_write(2)  = hwb_y(j)%snofall
        values_to_write(3)  = hwb_y(j)%snomlt
        values_to_write(4)  = hwb_y(j)%surq_gen
        values_to_write(5)  = hwb_y(j)%latq
        values_to_write(6)  = hwb_y(j)%wateryld
        values_to_write(7)  = hwb_y(j)%perc
        values_to_write(8)  = hwb_y(j)%et
        values_to_write(9)  = hwb_y(j)%ecanopy
        values_to_write(10) = hwb_y(j)%eplant
        values_to_write(11) = hwb_y(j)%esoil
        values_to_write(12) = hwb_y(j)%surq_cont
        values_to_write(13) = hwb_y(j)%cn
        values_to_write(14) = hwb_y(j)%sw_init
        values_to_write(15) = hwb_y(j)%sw_final
        values_to_write(16) = hwb_y(j)%sw
        values_to_write(17) = hwb_y(j)%sw_300
        values_to_write(18) = hwb_y(j)%sno_init
        values_to_write(19) = hwb_y(j)%sno_final
        values_to_write(20) = hwb_y(j)%snopack
        values_to_write(21) = hwb_y(j)%pet
        values_to_write(22) = hwb_y(j)%qtile
        values_to_write(23) = hwb_y(j)%irr
        values_to_write(24) = hwb_y(j)%surq_runon
        values_to_write(25) = hwb_y(j)%latq_runon
        values_to_write(26) = hwb_y(j)%overbank
        values_to_write(27) = hwb_y(j)%surq_cha
        values_to_write(28) = hwb_y(j)%surq_res
        values_to_write(29) = hwb_y(j)%surq_ls
        values_to_write(30) = hwb_y(j)%latq_cha
        values_to_write(31) = hwb_y(j)%latq_res
        values_to_write(32) = hwb_y(j)%latq_ls
        values_to_write(33) = hwb_y(j)%gwtran
        values_to_write(34) = hwb_y(j)%satex
        values_to_write(35) = hwb_y(j)%satex_chan
        values_to_write(36) = hwb_y(j)%delsw
        values_to_write(37) = hwb_y(j)%lagsurf
        values_to_write(38) = hwb_y(j)%laglatq
        values_to_write(39) = hwb_y(j)%lagsatex
        values_to_write(40) = hwb_y(j)%wet_out

        call MPI_Barrier(MPI_COMM_WORLD, mpi_err)

        ! Create property list for collective I/O
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Set the property list for collective I/O
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
        if (hdferr /= 0) then
            call h5pclose_f(plist_id, hdferr)
            return
        endif

        ! Loop over each dataset to write values
        do i = mpi_rank + 1, 40, mpi_size

            ! Retrieve the dataset's dataspace
            call h5dget_space_f(dataset_ids(i), dataspace_id, hdferr)
            if (hdferr /= 0) then
                cycle
            endif

            ! Set the start position in the dataset for this HRU and year
            start(1) = j - 1  ! HRU
            start(2) = year_idx - 1  ! Year index

            ! Set the count of elements to write
            count(1) = 1  ! Writing one element in HRU dimension
            count(2) = 1  ! Writing one element in Year dimension

            ! Select the hyperslab in the dataset where the data will be written
            call h5sselect_hyperslab_f(dataspace_id, H5S_SELECT_SET_F, start, count, hdferr)
            if (hdferr /= 0) then
                call h5sclose_f(dataspace_id, hdferr)
                cycle
            endif

            ! Create a memory dataspace for the data to be written
            call h5screate_simple_f(2, count, memspace_id, hdferr)
            if (hdferr /= 0) then
                call h5sclose_f(dataspace_id, hdferr)
                cycle
            endif

            ! Write the data using collective I/O
            call h5dwrite_f(dataset_ids(i), H5T_NATIVE_REAL, values_to_write(i), count, hdferr, memspace_id, dataspace_id, plist_id)
            if (hdferr /= 0) then
                call h5sclose_f(dataspace_id, hdferr)
                call h5sclose_f(memspace_id, hdferr)
                cycle
            endif

            ! Close the memory dataspace
            call h5sclose_f(memspace_id, hdferr)

            ! Close the dataset's dataspace
            call h5sclose_f(dataspace_id, hdferr)
        end do

        ! Close the property list for collective I/O
        call h5pclose_f(plist_id, hdferr)

    end if

end subroutine hru_output



















































































































