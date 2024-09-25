      subroutine sq_snom_snodas
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine predicts daily snom melt when the average air
!!    temperature exceeds 0 degrees Celcius, using SNODAS database and current SWAT+ module where no SNODAS data is availble

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ihru         |none          |HRU number
!!    snocov1      |none          |1st shape parameter for snow cover equation
!!                                |This parameter is determined by solving the
!!                                |equation for 50% snow cover
!!    snocov2      |none          |2nd shape parameter for snow cover equation
!!                                |This parameter is determined by solving the
!!                                |equation for 95% snow cover
!!    snotmp       |deg C         |temperature of snow pack in HRU
!!    snow_data_array             | SNODAS database linked to HRUs containing snow melt (MR), 

!!   snow_data_array%SAT          |  Deg C        Modeled snowpack average temperature        SWE-weighted average of snow layers   (no values: -99.0)         
!!   snow_data_array%BSSR         |  mm           Modeled blowing snow sublimation rate       24-hour total                         (no values: -99.0)
!!   snow_data_array%MR           |  mm           Modeled melt rate, bottom of snow layers    24-hour total                         (no values: -99.0)
!!   snow_data_array%SLT          |  mm           Modeled snow layer thickness                total of snow layer                   (no values: -99.0)
!!   snow_data_array%SWE          |  mm           Modeled snow water equivalent               total of snow layers                  (no values: -99.0)
!!   snow_data_array%SSR          |  mm           Modeled snowpack sublimation rate           24-hour total                         (no values: -99.0)
!!   snow_data_array%NSA          |  kg/sqm       Non-snow accumulation                       24-hour total                         (no values: -99.0)
!!   snow_data_array%SA           |  kg/sqm       Snow accumulation                           24-hour total                         (no values: -99.0)
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    wst(:)%weat%ts(:)  |mm H2O        |precipitation for the time step during day
!!    snofall      |mm H2O        |amount of precipitation falling as freezing rain/snow on day
!!    snomlt       |mm H2O        |amount of water in snow melt for the day
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Real, Sin, Exp

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use time_module
      use hydrograph_module
      use hru_module, only : hru, ihru, precip_eff, snocov1, snocov2,  &
        snofall, snomlt, hru_db, snodb
      use input_file_module, only : snow_data_array
      use climate_module, only: wst, w
      use output_landscape_module 

      implicit none
      integer :: snow_data_index
      
      integer :: j          !none       |HRU number
      real :: smfac         !           |
      real :: rto_sno  = 0. !none       |ratio of current day's snow water to minimum amount needed to
      integer :: i                      !           |cover ground completely 
      real :: snocov = 0.   !none       |fraction of HRU area covered with snow
      real ::   snotmp = 0.        !deg C      |temperature of snow pack
      integer :: ii         !none       |counter
      real ::  snosub    !mm        | snow sublimation, total 24 hr
      real ::  snotik    !mm        | snow thickness 
      real ::  snowe    !mm        | snow water equivalent 
      real ::  snoac    !kg/m2        | snow accumulation 
      real ::  nsnoac    !kg/m2        | non snow accumulation 
      integer :: start_index, mid_index, end_index
      real :: total_melt_rate
      integer :: count_valid_rates

      snomlt    = -99

      j = ihru  
        ! Find the index of the current day in snow_data_array
        snow_data_index = -1
        ii = hru_db(j)%dbs%snow   
       ! write(9001, *) 'snodb%MR_Jan', snodb(ii)MMR(1)
        ! Find the index of the current day in the snow data array using binary search
        if (size(snow_data_array)>0) then

          ! setting start/end index for binary search
          start_index = 1
          end_index = size(snow_data_array(ii)%day)
          snow_data_index = -1

          ! binary search
          do while (start_index <= end_index)
              mid_index = (start_index + end_index) / 2
              if (snow_data_array(ii)%year(mid_index) == time%yrc .and. &              
                  snow_data_array(ii)%day(mid_index) == time%day) then
                  snow_data_index = mid_index
                  exit
              elseif (snow_data_array(ii)%year(mid_index) < time%yrc .or. &
                      (snow_data_array(ii)%year(mid_index) == time%yrc .and. &
                      snow_data_array(ii)%day(mid_index) < time%day)) then
                  start_index = mid_index + 1
              else
                  end_index = mid_index - 1
              endif
          end do

          ! if the index is found, then use the SNODAS data
          snomlt = snow_data_array(ii)%MR(snow_data_index) 
          snosub = snow_data_array(ii)%SSR(snow_data_index)
          snotik = snow_data_array(ii)%SLT(snow_data_index) 
          nsnoac = snow_data_array(ii)%NSA(snow_data_index)    
          snoac  = snow_data_array(ii)%SA(snow_data_index)
          snowe  = snow_data_array(ii)%SWE(snow_data_index)

        endif

        if (snoac >= 0) then
          !! calculate snow fall
          snofall = snoac
          hru(j)%sno_mm = hru(j)%sno_mm + snofall
        else
          snofall = 0
        end if

        if (nsnoac >= 0) then
          !! calculate snow fall
          precip_eff =  precip_eff + nsnoac
        end if

        if (snomlt < 0) then
          snomlt = 0
        end if

        
        if (snomlt > hru(j)%sno_mm) then
          snomlt = hru(j)%sno_mm
          hru(j)%sno_mm = 0.
        else
          hru(j)%sno_mm = hru(j)%sno_mm - snomlt
        end if

      return
      end subroutine sq_snom_snodas

