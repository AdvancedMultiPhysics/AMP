       module AMP_INTERFACE
         use, intrinsic :: iso_c_binding, only: C_INT, C_PTR, C_BOOL
         contains
         subroutine AMP_STARTUP(argc, argv)
           use, intrinsic :: iso_c_binding, only: C_INT, C_PTR
           integer(c_int), value :: argc
           type(c_ptr), intent(in) :: argv(*)
           interface
             subroutine amp_startup_f(argc, argv) bind(c)
               use, intrinsic :: iso_c_binding, only: C_INT, C_PTR
               implicit none
               integer(c_int), value :: argc
               type(c_ptr), intent(in) :: argv(*)
             end subroutine amp_startup_f
           end interface
           call amp_startup_f(argc, argv)
         end subroutine    
         subroutine AMP_SHUTDOWN( )
           interface
             subroutine amp_shutdown_f( ) bind(c)
             end subroutine amp_shutdown_f
           end interface
           call amp_shutdown_f( )
         end subroutine
         logical(c_bool) function AMP_INITIALIZED() RESULT(c_result)
           use, intrinsic :: iso_c_binding, only: c_bool
           interface
             function amp_initialized_f( ) bind(c)
               use, intrinsic :: iso_c_binding, only: c_bool
               implicit none
               logical(c_bool) :: amp_initialized_f 
             end function amp_initialized_f
           end interface
           c_result = amp_initialized_f()
         end function
         logical(c_bool) function AMP_FINALIZED() RESULT(c_result)
           use, intrinsic :: iso_c_binding, only: c_bool
           interface
             function amp_finalized_f( ) bind(c)
               use, intrinsic :: iso_c_binding, only: c_bool
               implicit none
               logical(c_bool) :: amp_finalized_f 
             end function amp_finalized_f
           end interface
           c_result = amp_finalized_f()
         end function
       end module
