      subroutine snowdb_read_snodas
      !! this subroutine reads the snow.sno database
      use input_file_module
      use maximum_data_module
      use hru_module
      
      implicit none

      character (len=80) :: titldum   !           |title of file
      character (len=80) :: header    !           |header of file
      character (len=13) :: file      !           |
      integer :: eof                  !           |end of file
      integer :: imax                 !none       |determine max number for array (imax) and total number in file
      integer :: snowmax
      logical :: i_exist              !none       |check to determine if file exists
      integer :: i                    !none       |counter
      integer :: j
      integer :: msno                 !           |
      integer :: isno                 !none       |counter
      integer :: ioerr
      integer :: dum1, dum2, dum3, dum4
      integer  date_time(8) 
      character*10 b(3) 
      msno = 0
      eof = 0
      imax = 0
            

      !! read snow database data from snow.sno
      inquire (file=in_parmdb%snow, exist=i_exist)                  

      if (.not. i_exist .or. in_parmdb%snow == "null") then          !!!! checking whether this file exists

        allocate (snodb(0:0))
      
      else  
        do 
          open (107,file=in_parmdb%snow)
          read (107,*,iostat=eof) titldum  
          read (107,*,iostat=eof) titldum          
          if (eof < 0) exit
            
            do while (eof == 0)
              read (107,*,iostat=eof) titldum
              if (eof < 0) exit
        
              imax = imax + 1
              
            end do

          rewind (107)

          read (107,*,iostat=eof) titldum
          if (eof < 0) exit
          read (107,*,iostat=eof) header
          if (eof < 0) exit
          
          allocate (snodb(0:imax))
      
          do isno = 1, imax
            read (107,*,iostat=eof) snodb(isno) 
            write(9001,*) 'snodb', snodb(isno)
            if (eof < 0) exit
          end do

        end do

        close (107)
        db_mx%sno = imax

      end if
  
      allocate(snow_data_array(imax))

      write(9001, *) 'number of snow.sno:', imax

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  now we start reading the SNODAS files one by one
      call DATE_AND_TIME (b(1), b(2), b(3), date_time)
      
      write (*,111) "reading from SNODAS files          ", date_time(5), date_time(6), date_time(7)
      
      do i = 1, imax

        !  write(9001, *) 'reading the SNDAS file:', trim(snodb(i)%name)
        
          open(unit=1080, file=trim(snodb(i)%name), status='old', iostat=ioerr)
      
          if (ioerr /= 0) then
              ! Handle error
              exit
          endif

          ! Skip headers
          do j = 1, 4
              read (1080,*,iostat=eof) titldum
              if (eof /= 0) exit
          end do

          ! Count the number of data lines
          snowmax = 0
  
          do
              read(1080,*,iostat=eof) titldum
              if (eof /= 0) exit
              snowmax = snowmax + 1
          end do

        !  write(9001, *) 'number of snowmax:', snowmax

          ! Allocate the snow_data_array for the current file
          allocate( snow_data_array(i)%year(snowmax)  , snow_data_array(i)%day(snowmax), &    !!! this is the line i recieve error
                    snow_data_array(i)%SA(snowmax)    , snow_data_array(i)%NSA(snowmax), &
                    snow_data_array(i)%MR(snowmax)    , snow_data_array(i)%SWE(snowmax), &
                    snow_data_array(i)%BSSR(snowmax)  , snow_data_array(i)%SLT(snowmax), &
                    snow_data_array(i)%SSR(snowmax)   , snow_data_array(i)%SAT(snowmax)  )


          ! Rewind and read the actual data

          rewind (1080)
          do j = 1, 4
              read (1080,*)  ! Skip headers again
          end do
          
         ! write(9001, *) 'start reading the SNOFAS content:'
          do j = 1, snowmax 
      
              read (1080, *, iostat=eof) snow_data_array(i)%year(j), snow_data_array(i)%day(j), &
                      snow_data_array(i)%SA(j), snow_data_array(i)%NSA(j), snow_data_array(i)%MR(j), &
                      snow_data_array(i)%SWE(j), snow_data_array(i)%BSSR(j), snow_data_array(i)%SLT(j), &
                      snow_data_array(i)%SSR(j)

      
              if (eof /= 0) exit
      
          end do
          
          close (1080)

      end do
      return

111   format(1x,a, 5x,"Time",2x,i2,":",i2,":",i2)

      end subroutine snowdb_read_snodas