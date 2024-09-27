       module AMP_INTERFACE
         use iso_c_binding, only: C_INT, C_PTR
         contains
         subroutine AMP_STARTUP(argc, argv)
           use iso_c_binding, only: C_INT, C_PTR
           integer(c_int), value :: argc
           type(c_ptr), intent(in) :: argv(*)
           interface
             subroutine amp_startup_f(argc, argv) bind(c)
               use iso_c_binding, only: C_INT, C_PTR
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
       end module
