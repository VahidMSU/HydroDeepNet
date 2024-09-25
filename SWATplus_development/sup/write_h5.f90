subroutine write_to_h5dataset(filename, dataset_name, data, headers)
    use hdf5
    implicit none
    character(len=*), intent(in) :: filename, dataset_name
    integer, dimension(:,:,:), intent(in) :: data
	character(len=*), dimension(:), intent(in) :: headers
    integer(HID_T) :: file_id, dataset_id, dataspace_id
    integer(HSIZE_T), dimension(3) :: dims
    integer :: error

    ! Initialize HDF5 library
    call h5open_f(error)

    ! Open an existing HDF5 file
    call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error)
    if (error /= 0) then
        print*, "Failed to open file: ", filename
        return
    end if

    ! Open an existing dataset
    call h5dopen_f(file_id, dataset_name, dataset_id, error)
    if (error /= 0) then
        print*, "Failed to open dataset: ", dataset_name
        call h5fclose_f(file_id, error)
        return
    end if

    ! Get dataspace of the dataset
    call h5dget_space_f(dataset_id, dataspace_id, error)
    if (error /= 0) then
        print*, "Failed to get dataspace for dataset: ", dataset_name
        call h5dclose_f(dataset_id, error)
        call h5fclose_f(file_id, error)
        return
    end if

    ! Read dimensions of the dataset dataspace
    call h5sget_simple_extent_dims_f(dataspace_id, dims, error)
    if (error /= 0) then
        print*, "Failed to get dimensions for dataspace."
        call h5dclose_f(dataset_id, error)
        call h5sclose_f(dataspace_id, error)
        call h5fclose_f(file_id, error)
        return
    end if

    ! Write the data to the dataset
    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, data, dims, error)
    if (error /= 0) then
        print*, "Failed to write data to dataset: ", dataset_name
    end if

	! Get the dimensions of the data
	dims = shape(data)

	! Create a dataspace for the dataset
	call h5screate_simple_f(size(dims), dims, dataspace_id, error)

	! Write the data to the dataset
	call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, data, dims, error)

	! Close the dataspace, dataset, and file
	call h5sclose_f(dataspace_id, error)
	call h5dclose_f(dataset_id, error)
	call h5fclose_f(file_id, error)

	! Close the HDF5 library
	call h5close_f(error)
end subroutine write_to_dataset
