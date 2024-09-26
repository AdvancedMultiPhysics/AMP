       module AMP_INTERFACE
         use iso_c_binding, only: C_INT, C_PTR
         contains
         subroutine STARTUP(argc, argv)
           use iso_c_binding, only: C_INT, C_PTR
           integer(c_int), value :: argc
           type(c_ptr), value :: argv
           interface
             subroutine amp_startup(argc, argv) bind(c)
               use iso_c_binding, only: C_INT, C_PTR
               implicit none
               integer(c_int), value :: argc
               type(c_ptr), value :: argv
             end subroutine amp_startup
           end interface
           call amp_startup(argc, argv)
         end subroutine    
         subroutine SHUTDOWN( )
           interface
             subroutine amp_shutdown( ) bind(c)
             end subroutine amp_shutdown
           end interface
           call amp_shutdown( )
         end subroutine
       end module
