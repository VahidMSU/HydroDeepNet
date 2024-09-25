module hdf5_module
    use hdf5
    implicit none
    private
    public :: write_hdf5_data
	contains
	subroutine write_hdf5_data(file_id, dataset_name, data, index)
		integer(HID_T), intent(in) :: file_id
		character(len=*), intent(in) :: dataset_name
		real, dimension(:), intent(in) :: data
		integer, intent(in) :: index
		integer(HID_T) :: dataset_id
		integer(HSIZE_T), dimension(1) :: start, count
		integer :: error
		start = [index - 1]  ! Adjust based on your actual data structure
		count = [size(data)] ! Number of elements to write
		! Open the dataset
		call h5dopen_f(file_id, dataset_name, dataset_id, error)
		! Write the data
		call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, data, count, error, start=start)
		! Close the dataset
		call h5dclose_f(dataset_id, error)
	end subroutine write_hdf5_data
end module hdf5_module