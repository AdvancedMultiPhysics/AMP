      program hello_world
        use AMP_INTERFACE
        use, intrinsic :: iso_c_binding

        ! check if initialized
        if ( amp_initialized() ) then
           print *, 'amp initialized before calling startup'
        else
           print *, 'amp not initialized before startup'
        endif

        ! startup
        call amp_startup_basic_f()

        ! check if initialized
        if ( amp_initialized() ) then
           print *, 'amp initialized after calling startup'
        else
           print *, 'amp not initialized after startup'
        endif

        ! test initialized

        ! check if finalized
        if ( amp_finalized() ) then
           print *, 'amp finalized before shutdown'
        else
           print *, 'amp not finalized before shutdown'
        endif

        ! shutdown
        call amp_shutdown()
        ! check if finalized
        if ( amp_finalized() ) then
           print *, 'amp finalized after shutdown'
        else
           print *, 'amp not finalized after shutdown'
        endif
        ! Finished
        print *, 'Finished'
      end program


