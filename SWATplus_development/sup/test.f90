program write_hdf5_parallel
    use hdf5
    use mpi
    implicit none
    integer :: i, j, mpierr, rank, nprocs, error
    integer, dimension(3600, 1) :: data
    integer(HID_T) :: file_id, dataspace_id, dataset_id
    integer(HSIZE_T), dimension(2) :: dims = [4, 20]
    character(len=10) :: dataset_name

    ! Initialize MPI
    call MPI_INIT(mpierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, mpierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, mpierr)

    ! Initialize HDF5 library
    call h5open_f(error)
	! name of the file is parallel.hf
	! the file contains the following datasets:
	! hru_wb_day, channel_sd_day, hru_wb_mon, channel_sd_mon
	! each dataset has several variables for different HRUs
	! the variables are: hru_id, hru_area, hru_lat, hru_lon, hru_elev, hru_slope
	!  hru_wb_day and channel_sd_mon have following variables and identifiers: 
	!jday   mon   day    yr    unit  gis_id  name                  precip     snofall      snomlt    surq_gen        latq    wateryld        perc          et     ecanopy      eplant       esoil   surq_cont          cn     sw_init    sw_final      sw_ave      sw_300    sno_init   sno_final     snopack         pet       qtile         irr  surq_runon  latq_runon    overbank    surq_cha    surq_res     surq_ls    latq_cha    latq_res     latq_ls     gwtranq       satex  satex_chan   sw_change     lagsurf     laglatq    lagsatex     wet_out
    !                                                              mm            mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm         ---          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm          mm

	! channel_sd_day and channel_sd_mon variables and identifiers: 
	!jday   mon   day    yr     unit   gis_id   name                  area         precip           evap           seep          flo_stor       sed_stor      orgn_stor      sedp_stor       no3_stor      solp_stor      chla_stor       nh3_stor       no2_stor      cbod_stor       dox_stor       san_stor       sil_stor       cla_stor       sag_stor       lag_stor       grv_stor           null         flo_in         sed_in        orgn_in        sedp_in         no3_in        solp_in        chla_in         nh3_in         no2_in        cbod_in         dox_in         san_in         sil_in         cla_in         sag_in         lag_in         grv_in           null        flo_out        sed_out       orgn_out       sedp_out        no3_out       solp_out       chla_out        nh3_out        no2_out       cbod_out        dox_out        san_out        sil_out        cla_out        sag_out        lag_out        grv_out           null     water_temp
	!																 ha            m^3            m^3            m^3             m^3           tons            kgN            kgP            kgN            kgP             kg            kgN            kgN             kg             kg           tons           tons           tons           tons           tons           tons                         m^3/s           tons            kgN            kgP            kgN            kgP             kg            kgN            kgN             kg             kg           tons           tons           tons           tons           tons           tons                         m^3/s           tons            kgN            kgP            kgN            kgP             kg            kgN            kgN             kg             kg           tons           tons           tons           tons           tons           tons                          degc
	!!! note that the subroutine that write on the file sends identifiers seperately plus an array of data for one 
	if (rank == 0) then
		! create the h5 file with the above datasets names and variables
		call h5fcreate_f("parallel.h5", H5F_ACC_TRUNC_F, file_id, error)
    end if
    call MPI_BARRIER(MPI_COMM_WORLD, mpierr)

    ! All ranks open the file
    call h5fopen_f("parallel.h5", H5F_ACC_RDWR_F, file_id, error)

    ! Create the data space for the dataset.
    call h5screate_simple_f(2, dims, dataspace_id, error)

    ! Each MPI process creates and writes its dataset
    write(dataset_name, '(A5,I1)') "dset", rank + 1
	
    ! random data
	do i = 1, 3600
		do j = 1, 1
			data(i, j) = i + j
		end do
	end do

    ! Create the dataset.
    call h5dcreate_f(file_id, dataset_name, H5T_NATIVE_INTEGER, dataspace_id, dataset_id, error)
    ! Write the dataset.
    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, data, dims, error)
    ! Close the dataset.
    call h5dclose_f(dataset_id, error)
    ! Close/release resources.
    call h5sclose_f(dataspace_id, error)
    call h5fclose_f(file_id, error)
    call h5close_f(error)
    call MPI_FINALIZE(mpierr)

end program write_hdf5_parallel
