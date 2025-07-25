# Copy test inputs
COPY_TEST_FILE( input_linearDiffusion-TUI-Soret-1 )
COPY_TEST_FILE( input_linearDiffusion-TUI-Thermal-1 )
COPY_TEST_FILE( input_linearDiffusion-UO2MSRZC09-Fick-1 )
COPY_TEST_FILE( input_linearDiffusion-UO2MSRZC09-Soret-1 )
COPY_TEST_FILE( input_linearDiffusion-UO2MSRZC09-Thermal-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Fick-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Soret-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Thermal-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Fick-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Soret-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Thermal-DirichletMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Fick-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Soret-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-TUI-Thermal-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Fick-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Soret-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_linearBVPOperator-Diffusion-UO2MSRZC09-Thermal-RobinMatrixCorrection-1 )
COPY_TEST_FILE( input_nonlinearBVP-Thermal-Oxygen-UO2MSRZC09-1 )
COPY_TEST_FILE( input_linearBVP-Thermal+Fick+Dirichlet-TUI-1 )
COPY_TEST_FILE( input_linearBVP-Thermal+Fick+Dirichlet-UO2MSRZC09-1 )
COPY_TEST_FILE( input_linearBVP-Thermal+Fick+Robin-TUI-1 )
COPY_TEST_FILE( input_linearBVP-Thermal+Fick+Robin-UO2MSRZC09-1 )
ADD_AMP_TEST_1_2_4 ( testLinearOperatorFromMatrix-1 )

COPY_TEST_FILE( input_Diffusion-TUI-Thermal-1  )
COPY_TEST_FILE( input_Diffusion-TUI-Fick-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Soret-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Thermal-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Fick-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Soret-1 )
COPY_TEST_FILE( input_Diffusion-TUI-TensorFick-1 )
COPY_TEST_FILE( input_Diffusion-CylindricalFick-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Thermal-2  )
COPY_TEST_FILE( input_Diffusion-TUI-Fick-2 )
COPY_TEST_FILE( input_Diffusion-TUI-Soret-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Thermal-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Fick-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Soret-2 )
COPY_TEST_FILE( input_Diffusion-TUI-Thermal-1  )
COPY_TEST_FILE( input_Diffusion-TUI-Fick-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Soret-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Thermal-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Fick-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Soret-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Thermal-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Fick-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Soret-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Thermal-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Fick-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Soret-ActiveTemperatureAndConcentration-1 )
COPY_TEST_FILE( input_Diffusion-TUI-TensorFick-1 )
COPY_TEST_FILE( input_Diffusion-CylindricalFick-1 )
COPY_TEST_FILE( input_Diffusion-TUI-Thermal-1  )
COPY_TEST_FILE( input_Diffusion-TUI-Thermal-2  )
COPY_TEST_FILE( input_Diffusion-TUI-Fick-2 )
COPY_TEST_FILE( input_Diffusion-TUI-Soret-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Thermal-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Fick-2 )
COPY_TEST_FILE( input_Diffusion-UO2MSRZC09-Soret-2 )
COPY_TEST_FILE( input_Diffusion-Fick-OxMSRZC09-MMS-1 )
COPY_TEST_FILE( input_Diffusion-Fick-TUI-MMS-1 )
COPY_TEST_FILE( input_TensorDiffusion-Fick-MMS-1 )
COPY_TEST_FILE( input_TensorDiffusion-Fick-MMS-2 )
COPY_TEST_FILE( input_linearThermalDiffusion+FickDiffusion-TUI-1 )
COPY_TEST_FILE( input_linearThermalDiffusion+FickDiffusion-UO2MSRZC09-1 )


# Add the tests
INITIALIZE_TESTBUILDER()
# We need to sort out which tests rely on AMP_DATA
IF ( AMP_DATA )
    ADD_TB_TEST_1_2_4( testNonlinearThermalOxygenDiffusion_1 )
    ADD_TB_TEST_1_2_4( testLinearDiffusion_1 )
    ADD_TB_TEST_1_2_4( testLinearDiffusion_2 )
    ADD_TB_TEST_1_2_4( testNonlinearDiffusion_1 )
    ADD_TB_TEST_1_2_4( testNonlinearDiffusion_2 )
    ADD_TB_TEST_1_2_4( testDiffusionBVP_1 )
    ADD_TB_TEST_1_2_4( testDiffusionManufacturedSolution_1 )
    ADD_TB_TEST_1_2_4( testTensorDiffusionManufacturedSolution_1 )
    ADD_TB_PROVISIONAL_TEST( testDiffusionMMSForward )
ENDIF()
FINALIZE_TESTBUILDER()


