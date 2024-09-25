! test_hdf5.f90
program test_hdf5
   use hdf5
   implicit none
   integer(hid_t) :: file_id
   integer :: error

   ! Create a new file using default properties.
   call h5open_f(error)
   call h5fcreate_f("testfile.h5", H5F_ACC_TRUNC_F, file_id, error)
   call h5fclose_f(file_id, error)
   call h5close_f(error)

   if (error == 0) then
      print *, "HDF5 test file created successfully."
   else
      print *, "Error creating HDF5 test file."
   end if
end program test_hdf5