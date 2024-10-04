      program hello_world
        use AMP_INTERFACE
        use, intrinsic :: iso_c_binding
        ! following https://community.intel.com/t5/Intel-Fortran-Compiler/passing-an-array-of-strings-from-Fortran-to-C/m-p/1211858?profile.language=fr&countrylabel=Latin%20America
        integer(c_int) :: argc
        type(c_ptr), dimension(:), allocatable :: argv
        ! this is the array of names, max 40 characters including null termination 
        character(len=40,kind=c_char), allocatable, target :: args(:) ! must be target in order to get the starting C address of each name, which is a C char*

        argc = command_argument_count()
        allocate(args(num_args))  ! I've omitted checking the return status of the allocation 

        do ix = 1, num_args
           call get_command_argument(ix,args(ix))
           ! now parse the argument as you wish
        end do

        allocate(argv(argc))     ! allocate the storage for what the C char** points to
        argv(1) = c_loc(args(1)) ! get the starting C address of each arg
        argv(2) = c_loc(args(2))  

        ! startup
        call amp_startup( argc, argv )

        deallocate (argv) ! deallocate the C char** array
        deallocate (args) ! deallocate the args        
        ! test initialized

        ! shutdown
        call amp_shutdown()
        ! Finished
        print *, 'Finished'
      end program


