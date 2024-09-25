subroutine hru_lte_h5_output(isd)
	use hdf5_module
	use hdf5
	use time_module
	use basin_module
	use output_landscape_module
	use hydrograph_module, only: sp_ob1, ob
	implicit none
	integer, intent(in) :: isd
	integer :: iob, error
	integer(HID_T) :: file_id, dataset_id
	character(len=128) :: dataset_path
	real, dimension(:), allocatable :: buffer

	iob = sp_ob1%hru_lte + isd - 1

	! Initialize HDF5 library
	call h5open_f(error)
	! Open existing HDF5 file
	call h5fopen_f('output.h5', H5F_ACC_RDWR_F, file_id, error)

	! Add daily, monthly, yearly, and annual outputs
	if (pco%day_print == "y" .and. pco%int_day_cur == pco%int_day) then
		if (pco%wb_sd%d == "y") then
			dataset_path = 'hru_wb/day'
			call write_hdf5_data(file_id, dataset_path, hltwb_d(isd), isd)
		endif
		! Additional conditions for ls and pw can be added here
	endif

	! Similar checks for month, year, and simulation end
	! Close HDF5 file
	call h5fclose_f(file_id, error)
	call h5close_f(error)
end subroutine hru_lte_h5_output
