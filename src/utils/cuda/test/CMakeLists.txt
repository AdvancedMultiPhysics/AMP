ADD_AMP_TEST( deviceQuery GPU )
ADD_AMP_TEST( test_foo GPU )
ADD_AMP_TEST( testGPUAllocators GPU )
ADD_AMP_TEST( testGPUFunctionTable GPU )
ADD_AMP_TEST( testHelperCuda GPU )
SET_SOURCE_FILES_PROPERTIES( testHelperCuda.cpp PROPERTIES LANGUAGE CUDA )

