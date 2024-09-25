subroutine linear_search(sch, max, cfind, iseq)

    implicit none

    character(len=50), intent(in) :: cfind                  ! Target item to find
    integer, intent(in) :: max                              ! Size of the input array
    character(len=50), dimension(max), intent(in) :: sch    ! Input array to search
    integer, intent(out) :: iseq                            ! Index of the found item

    integer :: i                                            ! Loop variable

    iseq = 0  ! Initialize as not found

    ! Loop through each element of the array
    do i = 1, max
        if (trim(sch(i)) == trim(cfind)) then
            iseq = i
            return
        end if
    end do

end subroutine linear_search
