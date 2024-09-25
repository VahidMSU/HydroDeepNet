module hdf5_operations
    use hdf5
    use mpi
    use iso_c_binding, only: c_int_fast32_t  ! Correct type name

    implicit none

contains

    subroutine write_hdf5_file(database_name, level, dataset_name, data)
        character(len=*), intent(in) :: database_name, level, dataset_name
        real(c_float), dimension(:, :), intent(in) :: data
        ! add printing 
        integer(c_int_fast32_t) :: file_id, group_id, dataspace_id, dataset_id, plist_id
        integer :: error_code, mpi_error_code, mpi_rank, mpi_size
        integer(HSIZE_T), dimension(2) :: dims
        integer :: mpi_info
        logical :: dataset_exists
        ! get nrows and ncols from data
        integer :: nrows, ncols
        nrows = size(data, 1)
        ncols = size(data, 2)
        print *, 'Writing HDF5 file with dataset name: ', dataset_name

        ! Get MPI rank and size
        call MPI_COMM_RANK(MPI_COMM_WORLD, mpi_rank, mpi_error_code)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, mpi_size, mpi_error_code)

        ! Create MPI info object
        call MPI_INFO_CREATE(mpi_info, mpi_error_code)
        print *, 'MPI rank: ', mpi_rank, 'MPI size: ', mpi_size

        ! Set dimensions
        dims = [nrows, ncols]

        ! Initialize HDF5 library
        call h5open_f(error_code)

        ! Create a file access property list with MPI-IO driver
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error_code)
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, mpi_info, error_code)
        print *, 'Error code after setting MPI-IO driver: ', error_code
        ! Check if the file exists by attempting to open it in read-only mode
        call h5fopen_f(database_name, H5F_ACC_RDONLY_F, file_id, error_code, access_prp=plist_id)
        print *, 'Error code after opening file: ', error_code
        if (error_code /= 0) then
            ! File does not exist, create a new file
            call h5fcreate_f(database_name, H5F_ACC_TRUNC_F, file_id, error_code, access_prp=plist_id)
            if (error_code /= 0) then
                print *, 'Error creating file'
                return
            end if
        else
            ! File exists, open it in read-write mode
            call h5fclose_f(file_id, error_code)
            call h5fopen_f(database_name, H5F_ACC_RDWR_F, file_id, error_code, access_prp=plist_id)
            if (error_code /= 0) then
                print *, 'Error opening file'
                return
            end if
        end if
        print *, 'Error code after opening file: ', error_code
        ! Close the file access property list
        call h5pclose_f(plist_id, error_code)

        ! Check if the group (level) exists and create it if it doesn't
        call h5gopen_f(file_id, level, group_id, error_code)
        if (error_code /= 0) then
            ! Group does not exist, create a new group
            call h5gcreate_f(file_id, level, group_id, error_code)
            if (error_code /= 0) then
                print *, 'Error creating or opening group'
                call h5fclose_f(file_id, error_code)
                return
            end if
        end if

        ! Check if the dataset exists within the group
        call h5lexists_f(group_id, dataset_name, dataset_exists, error_code)

        if (.not. dataset_exists) then
            ! Create the data space for the dataset
            call h5screate_simple_f(2, dims, dataspace_id, error_code)
            if (error_code /= 0) then
                print *, 'Error creating dataspace'
                call h5gclose_f(group_id, error_code)
                call h5fclose_f(file_id, error_code)
                return
            end if

            ! Create the dataset within the group
            call h5dcreate_f(group_id, dataset_name, H5T_NATIVE_INTEGER, dataspace_id, dataset_id, error_code)
            if (error_code /= 0) then
                print *, 'Error creating dataset'
                call h5sclose_f(dataspace_id, error_code)
                call h5gclose_f(group_id, error_code)
                call h5fclose_f(file_id, error_code)
                return
            end if

            ! Close the dataspace (it is no longer needed)
            call h5sclose_f(dataspace_id, error_code)
        else
            ! Open existing dataset within the group
            call h5dopen_f(group_id, dataset_name, dataset_id, error_code)
            if (error_code /= 0) then
                print *, 'Error opening dataset'
                call h5gclose_f(group_id, error_code)
                call h5fclose_f(file_id, error_code)
                return
            end if
        end if

        ! Write the dataset
        call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, data, dims, error_code)
        if (error_code /= 0) then
            print *, 'Error writing data'
        else
            print *, 'Data written successfully'
        end if

        ! Close/release resources
        call h5dclose_f(dataset_id, error_code)
        call h5gclose_f(group_id, error_code)
        call h5fclose_f(file_id, error_code)

        ! Close the HDF5 library
        call h5close_f(error_code)

        ! Finalize MPI info object
        call MPI_INFO_FREE(mpi_info, mpi_error_code)

        print *, 'HDF5 file created with dataset name: ', dataset_name

    end subroutine write_hdf5_file

end module hdf5_operations
