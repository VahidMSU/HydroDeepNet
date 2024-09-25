subroutine initialize_decision_table(dtbl)
    use conditional_module
    type(decision_table), intent(inout) :: dtbl
    integer :: i, j

    ! Initialize conditions
    do i = 1, size(dtbl%cond)
        dtbl%cond(i)%var = ' '
        dtbl%cond(i)%ob = ' '
        dtbl%cond(i)%ob_num = 0
        dtbl%cond(i)%lim_var = ' '
        dtbl%cond(i)%lim_op = ' '
        dtbl%cond(i)%lim_const = -999.9
    end do

    ! Initialize con_act
    dtbl%con_act = 0

    ! Initialize alternatives (alt)
    do i = 1, size(dtbl%alt, dim=1)
        do j = 1, size(dtbl%alt, dim=2)
            dtbl%alt(i, j) = ' '
        end do
    end do

    ! Initialize act_hit
    dtbl%act_hit = 'n'  

    ! Initialize actions (act)
    do i = 1, size(dtbl%act)
        dtbl%act(i)%typ = ' '
        dtbl%act(i)%ob = ' '
        dtbl%act(i)%ob_num = 0
        dtbl%act(i)%name = ' '
        dtbl%act(i)%option = ' '
        dtbl%act(i)%const = -999.9
        dtbl%act(i)%const2 = -999.9
        dtbl%act(i)%file_pointer = ' '
    end do

    ! Initialize act_typ and act_app
    dtbl%act_typ = 0
    dtbl%act_app = 0

    ! Initialize act_outcomes
    do i = 1, size(dtbl%act_outcomes, dim=1)
        do j = 1, size(dtbl%act_outcomes, dim=2)
            dtbl%act_outcomes(i, j) = 'n'  
        end do
    end do
end subroutine initialize_decision_table