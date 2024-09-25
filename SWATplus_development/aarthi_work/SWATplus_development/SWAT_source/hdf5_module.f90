module hdf5_module
    use hdf5
    use mpi
    use time_module
    use hydrograph_module
    implicit none

    ! Declare shared variables
    integer(hid_t) :: file_id, hru_wb_group_id, yr_group_id, metadata_group_id, year_dataset_id
    integer(hid_t), dimension(40) :: dataset_ids  ! Array to store IDs for each dataset
    integer :: my_mpi_comm  ! Renamed to avoid conflict with mpi_comm from mpi module

    ! Declare and initialize dataset names directly in the module
    character(len=30), public :: dataset_names(40) = (/ 'precip', 'snofall', 'snomlt', 'surq_gen', 'latq', 'wateryld', &
                                                       'perc', 'et', 'ecanopy', 'eplant', 'esoil', 'surq_cont', 'cn', &
                                                       'sw_init', 'sw_final', 'sw_ave', 'sw_300', 'sno_init', 'sno_final', &
                                                       'snopack', 'pet', 'qtile', 'irr', 'surq_runon', 'latq_runon', &
                                                       'overbank', 'surq_cha', 'surq_res', 'surq_ls', 'latq_cha', &
                                                       'latq_res', 'latq_ls', 'gwtranq', 'satex', 'satex_chan', &
                                                       'sw_change', 'lagsurf', 'laglatq', 'lagsatex', 'wet_out' /)

    ! Make the subroutines and variables accessible outside the module
    public :: create_hdf5_output, close_hdf5_resources
    public :: file_id, dataset_ids, hru_wb_group_id, yr_group_id

contains

    subroutine create_hdf5_output()
        ! Declare the subroutine variables
        integer(hid_t) :: dataspace_id, access_prop_id, create_prop_id, dataspace_id_yr
        integer :: hdferr, mpierr
        integer(hsize_t), dimension(2) :: dims
        integer, parameter :: rank = 2
        integer, parameter :: rank_yr = 1
        integer :: i, mpi_rank, mpi_size
        integer, allocatable :: year_data(:)
        character(len=*), parameter :: filename = "output.h5"

        ! Initialize HDF5 library
        call h5open_f(hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Get the rank and size of the process
        call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, mpierr)
        call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, mpierr)

        ! Create the file creation property list
        call H5Pcreate_f(H5P_FILE_CREATE_F, create_prop_id, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Create the file access property list (collective)
        call H5Pcreate_f(H5P_FILE_ACCESS_F, access_prop_id, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Set MPI-IO access on the access property list (collective)
        call H5Pset_fapl_mpio_f(access_prop_id, MPI_COMM_WORLD, MPI_INFO_NULL, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Enable collective metadata operations
        call H5Pset_all_coll_metadata_ops_f(access_prop_id, .true., hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Synchronize all processes before file creation
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Collectively create the HDF5 file in parallel using both property lists
        call H5Fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, hdferr, create_prop_id, access_prop_id)
        if (hdferr /= 0) then
            call H5Pclose_f(access_prop_id, hdferr)
            call H5Pclose_f(create_prop_id, hdferr)
            call h5close_f(hdferr)
            return
        endif

        ! Close the file creation and access property lists
        call H5Pclose_f(create_prop_id, hdferr)
        call H5Pclose_f(access_prop_id, hdferr)

        ! Synchronize all processes after file creation
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Collective group creation and opening for all ranks
        call h5gcreate_f(file_id, 'hru_wb', hru_wb_group_id, hdferr)
        if (hdferr /= 0) then
            call H5Fclose_f(file_id, hdferr)
            call h5close_f(hdferr)
            return
        endif

        ! Synchronize all processes after group creation
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Collective group creation for 'yr' group
        call h5gcreate_f(hru_wb_group_id, 'yr', yr_group_id, hdferr)
        if (hdferr /= 0) then
            call H5Gclose_f(hru_wb_group_id, hdferr)
            call H5Fclose_f(file_id, hdferr)
            call h5close_f(hdferr)
            return
        endif

        ! Synchronize all processes before dataset creation
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Create the metadata group
        call h5gcreate_f(file_id, 'metadata', metadata_group_id, hdferr)
        if (hdferr /= 0) then
            call H5Fclose_f(file_id, hdferr)
            return
        endif

        ! Define the dimensions for the 'year' dataset
        dims(1) = time%yrc_end - time%yrc_start + 1

        ! Allocate memory for year data
        allocate(year_data(dims(1)))
        do i = 1, dims(1)
            year_data(i) = time%yrc_start + i - 1
        end do

        ! Create the dataspace for the 'year' dataset
        call h5screate_simple_f(rank_yr, dims, dataspace_id_yr, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Create the 'year' dataset inside the 'metadata' group
        call H5Dcreate_f(metadata_group_id, "year", H5T_NATIVE_INTEGER, dataspace_id_yr, year_dataset_id, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Write the year data to the dataset
        call H5Dwrite_f(year_dataset_id, H5T_NATIVE_INTEGER, year_data, dims, hdferr)
        if (hdferr /= 0) then
            return
        endif

        ! Close the dataspace and free year data
        call H5Sclose_f(dataspace_id_yr, hdferr)
        deallocate(year_data)
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Collective dataset creation by all ranks
        dims(1) = sp_ob%hru
        dims(2) = time%yrc_end - time%yrc_start + 1

        do i = 1, 40
            call h5screate_simple_f(rank, dims, dataspace_id, hdferr)
            if (hdferr /= 0) then
                return
            endif

            ! Create dataset and store its ID (collectively on all ranks)
            call H5Dcreate_f(yr_group_id, trim(dataset_names(i)), H5T_NATIVE_DOUBLE, dataspace_id, dataset_ids(i), hdferr)
            if (hdferr /= 0) then
                return
            endif

            ! Close dataspace after dataset creation
            call H5Sclose_f(dataspace_id, hdferr)
        end do

        ! Synchronize all processes after dataset creation
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Close datasets (first close after writing)
        do i = 1, 40
            if (dataset_ids(i) /= 0_hid_t) then
                call H5Dclose_f(dataset_ids(i), hdferr)
                if (hdferr /= 0) then
                    return
                endif
            endif
        end do

        ! Synchronize all processes after dataset closure
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Reopen datasets on all ranks
        do i = 1, 40
            call H5Dopen_f(yr_group_id, trim(dataset_names(i)), dataset_ids(i), hdferr)
            if (hdferr /= 0) then
                return
            endif
        end do

        ! Synchronize after reopening datasets
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)
    end subroutine create_hdf5_output

    subroutine close_hdf5_resources()
        integer :: hdferr, mpierr
        integer :: i

        ! Synchronize MPI processes before closing resources
        call MPI_Barrier(MPI_COMM_WORLD, mpierr)

        ! Close all datasets
        do i = 40, 1, -1
            if (dataset_ids(i) /= 0_hid_t) then
                call H5Dclose_f(dataset_ids(i), hdferr)
            endif
        end do

        ! Close the 'year' dataset
        if (year_dataset_id /= 0_hid_t) then
            call H5Dclose_f(year_dataset_id, hdferr)
        endif

        ! Close the 'yr' group
        if (yr_group_id /= 0_hid_t) then
            call H5Gclose_f(yr_group_id, hdferr)
        endif

        ! Close the 'hru_wb' group
        if (hru_wb_group_id /= 0_hid_t) then
            call H5Gclose_f(hru_wb_group_id, hdferr)
        endif

        ! Close the 'metadata' group
        if (metadata_group_id /= 0_hid_t) then
            call H5Gclose_f(metadata_group_id, hdferr)
        endif

        ! Close the HDF5 file
        if (file_id /= 0_hid_t) then
            call H5Fclose_f(file_id, hdferr)
        endif

        ! Close the HDF5 library
        call h5close_f(hdferr)
    end subroutine close_hdf5_resources

end module hdf5_module



















































































