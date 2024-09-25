program write_hdf5
	use hdf5
	implicit none
	integer :: i
	integer, dimension(4, 6) :: data = reshape([(i, i=1,24)], [4, 6])
	integer(HID_T) :: file_id, dataspace_id, dataset_id
	integer(HSIZE_T), dimension(2) :: dims = [4, 6]
	integer :: error

	! Initialize HDF5 library
	call h5open_f(error)

	! Create a new file using default properties.
	call h5fcreate_f("/home/rafieiva/MyDataBase/SWATplus_development/sup/build/dset.h5", H5F_ACC_TRUNC_F, file_id, error)

	! Create the data space for the dataset.
	call h5screate_simple_f(2, dims, dataspace_id, error)

	! Create the dataset.
	call h5dcreate_f(file_id, "dset", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, error)

	! Write the dataset.
	call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, data, dims, error)

	! Close/release resources.
	call h5dclose_f(dataset_id, error)
	call h5sclose_f(dataspace_id, error)
	call h5fclose_f(file_id, error)

	call h5close_f(error)
end program write_hdf5