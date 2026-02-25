# Install executables
SET( CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE )
ADD_AMP_EXECUTABLE( run_system_tpool )

# Add thread pool tests
FILE( COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/commands.txt" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" )
ADD_AMP_TEST( test_atomic RUN_SERIAL 1 )
ADD_AMP_TEST( test_atomic_list )
ADD_AMP_TEST( run_system_tpool ARGS 4 commands.txt THREADS 4 )
ADD_AMP_TEST( test_thread_pool PROCS 1 THREADS 4 )
ADD_AMP_TEST_1_2_4( test_lock_MPI_Mutex THREADS 4 )

# Add affinity tests
COPY_TEST_FILE( input_print_affinity_1 )
ADD_AMP_TEST( print_thread_affinities input_print_affinity_1 )
ADD_AMP_TEST( test_async_communication )
