      program hello_world
        use AMP_INTERFACE
        use, intrinsic :: iso_c_binding
        ! following https://community.intel.com/t5/Intel-Fortran-Compiler/passing-an-array-of-strings-from-Fortran-to-C/m-p/1211858?profile.language=fr&countrylabel=Latin%20America
        integer(c_int) :: argc
        type(c_ptr), dimension(:), allocatable :: argv
        ! this is the array of names, max 40 characters including null termination 
        character(len=40,kind=c_char), allocatable, target :: args(:) ! must be target in order to get the starting C address of each name, which is a C char*

        argc = 2
        allocate(args(argc))

        ! You must have //C_NULL_CHAR termination on dynamically allocated strings
        ! Otherwise on the C side there's no termination and this is very unpleasant
        args(1) = "hello_world"//c_null_char
        args(2) = "dummy first"//c_null_char

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


