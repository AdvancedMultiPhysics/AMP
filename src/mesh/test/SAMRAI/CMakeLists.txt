# Copy some basic input files for the tests
RENAME_TEST_FILE( input_3d/input_patch_1d          input_3d_patch_1d   )
RENAME_TEST_FILE( input_3d/input_patch_2level      input_3d_2level     )
RENAME_TEST_FILE( input_3d/input_patch_2level_2    input_3d_2level_2   )
RENAME_TEST_FILE( input_3d/input_patch_2level_3    input_3d_2level_3   )
RENAME_TEST_FILE( input_3d/input_patch_32_2level   input_3d_32_2level  )
RENAME_TEST_FILE( input_3d/input_patch_128_2level  input_3d_128_2level )
RENAME_TEST_FILE( input_3d/input_bug1              input_3d_bug1       )
RENAME_TEST_FILE( input_3d/input_bug2              input_3d_bug2       )
RENAME_TEST_FILE( input_3d/input_wall              input_3d_wall       )
RENAME_TEST_FILE( input_3d/input_PeriodicPatchHierarchy-1 input_PeriodicPatchHierarchy-1 )
RENAME_TEST_FILE( input_3d/input_nrdf_16b4l        input_3d_nrdf_16b4l )
RENAME_TEST_FILE( input_3d/input_3level_coarsen_bug  input_3d_3level_coarsen_bug )
RENAME_TEST_FILE( input_3d/input_nrdf_4b7l           input_3d_7level_coarsen_bug )
RENAME_TEST_FILE( input_MultipleHierarchies-1 input_MultipleHierarchies-1 )
RENAME_TEST_FILE( input_MultipleHierarchies-2 input_MultipleHierarchies-2 )

# Add interpolation test
ADD_AMP_TEST_1_2_4( test_coarsenHierarchy )

# tests for multiple hierarchies
ADD_AMP_TEST( test_multipleHierarchies input_MultipleHierarchies-1 log.MultipleHierarchies-1 PROCS 2 )
ADD_AMP_TEST( test_multipleHierarchies input_MultipleHierarchies-2 log.MultipleHierarchies-2 PROCS 4 )
ADD_AMP_TEST( test_PeriodicPatchHierarchy input_PeriodicPatchHierarchy-1 log_PeriodicPatchHierarchy-1 PROCS 1 )

