	subroutine channel_h5_output(jrch, file_id)
		use time_module
		use basin_module
		use hydrograph_module, only: ob, sp_ob1
		use channel_module
		use climate_module
		use hdf5_module, only: H5Fcreate_f, H5Gcreate_f, H5Dcreate_f, H5Dwrite_f, H5Sclose_f, H5Dclose_f, H5Gclose_f
		implicit none
		integer, intent(in) :: jrch
		integer, intent(in) :: file_id
		integer :: iob
		integer :: status
		integer(hid_t) :: group_id, dataset_id, dataspace_id
		integer(hsize_t) :: dims(2)
		
		iob = sp_ob1%chan + jrch - 1
		
		ch_m(jrch) = ch_m(jrch) + ch_d(jrch)
		
		if (pco%day_print == "y" .and. pco%int_day_cur == pco%int_day) then
			if (pco%chan%d == "y") then
				dims = [8, 1]
				dataspace_id = H5Screate_simple_f(2, dims, status)
				dataset_id = H5Dcreate_f(file_id, "day_print", H5T_NATIVE_REAL, dataspace_id, status)
				call H5Dwrite_f(dataset_id, H5T_NATIVE_REAL, ch_d(jrch), dims, status)
				call H5Dclose_f(dataset_id, status)
				call H5Sclose_f(dataspace_id, status)
			end if
		end if

		!!!! monthly print
		if (time%end_mo == 1) then
			ch_y(jrch) = ch_y(jrch) + ch_m(jrch)
			if (pco%chan%m == "y") then
				dims = [8, 1]
				dataspace_id = H5Screate_simple_f(2, dims, status)
				dataset_id = H5Dcreate_f(file_id, "monthly_print", H5T_NATIVE_REAL, dataspace_id, status)
				call H5Dwrite_f(dataset_id, H5T_NATIVE_REAL, ch_m(jrch), dims, status)
				call H5Dclose_f(dataset_id, status)
				call H5Sclose_f(dataspace_id, status)
			end if
			ch_m(jrch) = chz
		end if

		!!!! yearly print
		if (time%end_yr == 1) then
			ch_a(jrch) = ch_a(jrch) + ch_y(jrch)
			if (pco%chan%y == "y") then
				dims = [8, 1]
				dataspace_id = H5Screate_simple_f(2, dims, status)
				dataset_id = H5Dcreate_f(file_id, "yearly_print", H5T_NATIVE_REAL, dataspace_id, status)
				call H5Dwrite_f(dataset_id, H5T_NATIVE_REAL, ch_y(jrch), dims, status)
				call H5Dclose_f(dataset_id, status)
				call H5Sclose_f(dataspace_id, status)
			end if
			ch_y(jrch) = chz
		end if

		!!!! average annual print
		if (time%end_sim == 1 .and. pco%chan%a == "y") then
			ch_a(jrch) = ch_a(jrch) / time%yrs_prt
			dims = [8, 1]
			dataspace_id = H5Screate_simple_f(2, dims, status)
			dataset_id = H5Dcreate_f(file_id, "average_annual_print", H5T_NATIVE_REAL, dataspace_id, status)
			call H5Dwrite_f(dataset_id, H5T_NATIVE_REAL, ch_a(jrch), dims, status)
			call H5Dclose_f(dataset_id, status)
			call H5Sclose_f(dataspace_id, status)
		end if

		return
	end subroutine channel_output