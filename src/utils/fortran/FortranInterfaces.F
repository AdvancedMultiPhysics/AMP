module amp_interface
  use iso_c_binding, only: c_int, c_ptr
  implicit none

  subroutine AMP_STARTUP(argc, argv)
      integer(c_int), value :: argc
      type(c_ptr), value :: argv
      interface
         subroutine amp_startup(argc, argv) bind(c, name="amp_startup")
          implicit none
          integer(c_int), value :: argc
          type(c_ptr), value :: argv
        end subroutine amp_startup
      end interface
      call amp_startup(argc, argv)
  end subroutine    

  subroutine AMP_SHUTDOWN( )
      interface
         subroutine amp_shutdown(argc, argv) bind(c, name="amp_shutdown")
          implicit none
        end subroutine amp_shutdown
      end interface
      call amp_shutdown( )
  end subroutine    
end module
