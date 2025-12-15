#include "AMP/operators/mechanics/MechanicsLinearUpdatedLagrangianElement.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

void MechanicsLinearUpdatedLagrangianElement::computeStressAndStrain(
    const std::vector<double> &dispVec,
    std::vector<double> &stressVec,
    std::vector<double> &strainVec )
{
    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    d_fe->reinit( d_elem );

    d_materialModel->preLinearElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preLinearGaussPointOperation();

        double *constitutiveMatrix;

        d_materialModel->getConstitutiveMatrix( constitutiveMatrix );

        /* Compute Strain From Given Displacement */
        double dudx = 0;
        double dudy = 0;
        double dudz = 0;
        double dvdx = 0;
        double dvdy = 0;
        double dvdz = 0;
        double dwdx = 0;
        double dwdy = 0;
        double dwdz = 0;
        for ( unsigned int k = 0; k < num_nodes; k++ ) {
            dudx += dispVec[( 3 * k ) + 0] * dphi[k][qp]( 0 );
            dudy += dispVec[( 3 * k ) + 0] * dphi[k][qp]( 1 );
            dudz += dispVec[( 3 * k ) + 0] * dphi[k][qp]( 2 );

            dvdx += dispVec[( 3 * k ) + 1] * dphi[k][qp]( 0 );
            dvdy += dispVec[( 3 * k ) + 1] * dphi[k][qp]( 1 );
            dvdz += dispVec[( 3 * k ) + 1] * dphi[k][qp]( 2 );

            dwdx += dispVec[( 3 * k ) + 2] * dphi[k][qp]( 0 );
            dwdy += dispVec[( 3 * k ) + 2] * dphi[k][qp]( 1 );
            dwdz += dispVec[( 3 * k ) + 2] * dphi[k][qp]( 2 );
        } // end for k

        double uStrain[6];
        uStrain[0] = dudx;
        uStrain[1] = dvdy;
        uStrain[2] = dwdz;
        uStrain[3] = 0.5 * ( dvdz + dwdy );
        uStrain[4] = 0.5 * ( dudz + dwdx );
        uStrain[5] = 0.5 * ( dudy + dvdx );

        double uStress[6];
        for ( int r = 0; r < 6; r++ ) {
            uStress[r] = 0;
            for ( int c = 0; c < 6; c++ ) {
                uStress[r] += ( constitutiveMatrix[( 6 * r ) + c] * uStrain[c] );
            } // end for c
        }     // end for r

        for ( int i = 0; i < 6; i++ ) {
            stressVec[( 6 * qp ) + i] = uStress[i];
            strainVec[( 6 * qp ) + i] = uStrain[i];
        } // end for i

        d_materialModel->postLinearGaussPointOperation();
    } // end for qp

    d_materialModel->postLinearElementOperation();
}

void MechanicsLinearUpdatedLagrangianElement::printStressAndStrain(
    FILE *fp, const std::vector<double> &dispVec )
{
    size_t N = d_qrule->n_points();
    std::vector<double> stressVec( N * 6 ), strainVec( N * 6 );
    computeStressAndStrain( dispVec, stressVec, strainVec );
    const auto &xyz = *d_xyz;
    for ( size_t i = 0; i < N; i++ ) {
        auto stress = &stressVec[6 * i];
        auto strain = &strainVec[6 * i];
        fprintf( fp, "%.12f %.12f %.12f \n", xyz[i]( 0 ), xyz[i]( 1 ), xyz[i]( 2 ) );
        fprintf( fp,
                 "%.12f %.12f %.12f %.12f %.12f %.12f \n",
                 stress[0],
                 stress[1],
                 stress[2],
                 stress[3],
                 stress[4],
                 stress[5] );
        fprintf( fp,
                 "%.12f %.12f %.12f %.12f %.12f %.12f \n\n",
                 strain[0],
                 strain[1],
                 strain[2],
                 strain[3],
                 strain[4],
                 strain[5] );
    }
}

void MechanicsLinearUpdatedLagrangianElement::apply_Reduced()
{
    // AMP_INSIST((d_useJaumannRate == true), "Reduced integration has only been implemented for
    // Jaumann rate.");

    auto &elementStiffnessMatrix = ( *d_elementStiffnessMatrix );

    const auto &elementInputVectors = d_elementInputVectors;

    std::vector<libMesh::Point> xyz, xyz_np1;

    d_fe->reinit( d_elem );

    d_materialModel->preLinearElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize( num_nodes );
    xyz_np1.resize( num_nodes );

    for ( unsigned int ijk = 0; ijk < num_nodes; ijk++ ) {
        const auto &p1 = d_elem->point( ijk );
        xyz[ijk]       = p1;
    }

    double currX[8], currY[8], currZ[8], dNdx[8], dNdy[8], dNdz[8], detJ[1];
    double rsq3              = ( 1.0 / std::sqrt( 3.0 ) );
    const double currXi[8]   = { -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3 };
    const double currEta[8]  = { -rsq3, -rsq3, rsq3, rsq3, -rsq3, -rsq3, rsq3, rsq3 };
    const double currZeta[8] = { -rsq3, -rsq3, -rsq3, -rsq3, rsq3, rsq3, rsq3, rsq3 };
    double Bl_np1_bar[6][24], sum_detJ, Bl_center[6][24];
    double materialStiffness[24][24], materialStiffnessTemp[6][24];
    double dNdX[8], dNdY[8], dNdZ[8], refX[8], refY[8], refZ[8], detJ_0[1];

    for ( unsigned int ijk = 0; ijk < num_nodes; ijk++ ) {
        refX[ijk] = xyz[ijk]( 0 );
        refY[ijk] = xyz[ijk]( 1 );
        refZ[ijk] = xyz[ijk]( 2 );

        currX[ijk] = xyz_np1[ijk]( 0 ) =
            xyz[ijk]( 0 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 0];
        currY[ijk] = xyz_np1[ijk]( 1 ) =
            xyz[ijk]( 1 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 1];
        currZ[ijk] = xyz_np1[ijk]( 2 ) =
            xyz[ijk]( 2 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 2];
    }

    sum_detJ = 0.0;
    for ( unsigned int i = 0; i < 6; i++ ) {
        for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
            Bl_np1_bar[i][j] = 0.0;
            Bl_center[i][j]  = 0.0;
        }
    }

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        constructShapeFunctionDerivatives(
            dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ );
        sum_detJ += detJ[0];

        for ( unsigned int i = 0; i < 8; i++ ) {
            Bl_np1_bar[0][( 3 * i ) + 0] += ( dNdx[i] * detJ[0] );
            Bl_np1_bar[1][( 3 * i ) + 0] += ( dNdx[i] * detJ[0] );
            Bl_np1_bar[2][( 3 * i ) + 0] += ( dNdx[i] * detJ[0] );

            Bl_np1_bar[0][( 3 * i ) + 1] += ( dNdy[i] * detJ[0] );
            Bl_np1_bar[1][( 3 * i ) + 1] += ( dNdy[i] * detJ[0] );
            Bl_np1_bar[2][( 3 * i ) + 1] += ( dNdy[i] * detJ[0] );

            Bl_np1_bar[0][( 3 * i ) + 2] += ( dNdz[i] * detJ[0] );
            Bl_np1_bar[1][( 3 * i ) + 2] += ( dNdz[i] * detJ[0] );
            Bl_np1_bar[2][( 3 * i ) + 2] += ( dNdz[i] * detJ[0] );
        }
    }

    // std::cout<<"sum_detJ="<<sum_detJ<<std::endl;

    double one3TimesSumDetJ = 1.0 / ( 3.0 * sum_detJ );
    for ( auto &elem : Bl_np1_bar ) {
        for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
            elem[j] = elem[j] * one3TimesSumDetJ;
            // std::cout<<"Bl_np1_bar["<<i<<"]["<<j<<"]="<<Bl_np1_bar[i][j]<<std::endl;
        }
    }

    constructShapeFunctionDerivatives( dNdx, dNdy, dNdz, currX, currY, currZ, 0.0, 0.0, 0.0, detJ );
    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        Bl_center[0][( 3 * qp ) + 0] += ( dNdx[qp] );
        Bl_center[1][( 3 * qp ) + 1] += ( dNdy[qp] );
        Bl_center[2][( 3 * qp ) + 2] += ( dNdz[qp] );
        Bl_center[3][( 3 * qp ) + 1] += ( dNdz[qp] );
        Bl_center[3][( 3 * qp ) + 2] += ( dNdy[qp] );
        Bl_center[4][( 3 * qp ) + 0] += ( dNdz[qp] );
        Bl_center[4][( 3 * qp ) + 2] += ( dNdx[qp] );
        Bl_center[5][( 3 * qp ) + 0] += ( dNdy[qp] );
        Bl_center[5][( 3 * qp ) + 1] += ( dNdx[qp] );
    }

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearAssemblyGaussPointOperation();

        double Bl_np1[6][24], Bl_dil[6][24], Bnl_np1[9][24];

        // Calculate the derivatives of the shape functions at the current coordinate.
        constructShapeFunctionDerivatives(
            dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ );

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                Bl_np1[i][j] = 0.0;
                Bl_dil[i][j] = 0.0;
            }
        }

        for ( auto &elem : Bnl_np1 ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                elem[j] = 0.0;
            }
        }

        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            Bl_np1[0][( 3 * i ) + 0] = dNdx[i];
            Bl_np1[1][( 3 * i ) + 1] = dNdy[i];
            Bl_np1[2][( 3 * i ) + 2] = dNdz[i];
            Bl_np1[3][( 3 * i ) + 1] = dNdz[i];
            Bl_np1[3][( 3 * i ) + 2] = dNdy[i];
            Bl_np1[4][( 3 * i ) + 0] = dNdz[i];
            Bl_np1[4][( 3 * i ) + 2] = dNdx[i];
            Bl_np1[5][( 3 * i ) + 0] = dNdy[i];
            Bl_np1[5][( 3 * i ) + 1] = dNdx[i];

            double one3 = 1.0 / 3.0;
            for ( int j = 0; j < 3; j++ ) {
                Bl_dil[j][( 3 * i ) + 0] = dNdx[i] * one3;
                Bl_dil[j][( 3 * i ) + 1] = dNdy[i] * one3;
                Bl_dil[j][( 3 * i ) + 2] = dNdz[i] * one3;
            }
        }

        /*      for(int i = 0; i < 3; i++) {
                for(int j = 0; j < (3 * num_nodes); j++) {
                  Bl_np1[i][j] = Bl_np1[i][j] - Bl_dil[i][j] + Bl_center[i][j];
                }
              }
        */
        for ( int i = 0; i < 6; i++ ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                Bl_np1[i][j] = Bl_np1[i][j] - Bl_dil[i][j] + Bl_np1_bar[i][j];
                // std::cout<<"Bl_np1["<<i<<"]["<<j<<"]="<<Bl_np1[i][j]<<std::endl;
            }
        }

        if ( d_onePointShearIntegration ) {
            for ( int i = 3; i < 6; i++ ) {
                for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                    Bl_np1[i][j] = Bl_center[i][j];
                }
            }
        }

        for ( unsigned int i = 0; i < ( num_nodes ); i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                Bnl_np1[( 3 * j ) + 0][( 3 * i ) + j] = dNdx[i];
                Bnl_np1[( 3 * j ) + 1][( 3 * i ) + j] = dNdy[i];
                Bnl_np1[( 3 * j ) + 2][( 3 * i ) + j] = dNdz[i];
            }
        }

        double constitutiveMatrix[6][6];
        double currentStress[6], materialMatrix[6][6], R_np1[3][3];

        for ( unsigned int i = 0; i < 3; i++ ) {
            for ( unsigned int j = 0; j < 3; j++ ) {
                R_np1[i][j] = 0.0;
            }
            R_np1[i][i] = 1.0;
        }

        if ( d_useJaumannRate == false ) {
            double F_np1[3][3];
            constructShapeFunctionDerivatives(
                dNdX, dNdY, dNdZ, refX, refY, refZ, currXi[qp], currEta[qp], currZeta[qp], detJ_0 );
            // computeDeformationGradientLin(dphi, *xyz_np1, num_nodes, qp, F_np1);
            computeGradient( dNdX, dNdY, dNdZ, currX, currY, currZ, num_nodes, F_np1 );
            if ( d_useFlanaganTaylorElem == false ) {
                double U_np1[3][3];
                polarDecompositionFeqRU_Simo( F_np1, R_np1, U_np1 );
            }
        }

        d_materialModel->getConstitutiveMatrixUpdatedLagrangian( constitutiveMatrix, R_np1 );
        d_materialModel->getStressForUpdatedLagrangian( currentStress );

        double stressMatrix[3][3], bigStressMatrix[9][9];
        stressMatrix[0][0] = currentStress[0];
        stressMatrix[1][1] = currentStress[1];
        stressMatrix[2][2] = currentStress[2];
        stressMatrix[1][2] = stressMatrix[2][1] = currentStress[3];
        stressMatrix[0][2] = stressMatrix[2][0] = currentStress[4];
        stressMatrix[0][1] = stressMatrix[1][0] = currentStress[5];

        for ( auto &elem : bigStressMatrix ) {
            for ( double &j : elem ) {
                j = 0.0;
            }
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                materialMatrix[i][j] = constitutiveMatrix[i][j];
            }
        }

        if ( d_useJaumannRate == true ) {
            materialMatrix[0][0] -= ( 2.0 * currentStress[0] );
            materialMatrix[1][1] -= ( 2.0 * currentStress[1] );
            materialMatrix[2][2] -= ( 2.0 * currentStress[2] );
            materialMatrix[3][1] = materialMatrix[3][2] = materialMatrix[1][3] =
                materialMatrix[2][3] -= currentStress[3];
            materialMatrix[4][0] = materialMatrix[4][2] = materialMatrix[0][4] =
                materialMatrix[2][4] -= currentStress[4];
            materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[1][5] =
                materialMatrix[0][5] -= currentStress[5];
            materialMatrix[3][3] -= ( 0.5 * ( currentStress[1] + currentStress[2] ) );
            materialMatrix[4][4] -= ( 0.5 * ( currentStress[0] + currentStress[2] ) );
            materialMatrix[5][5] -= ( 0.5 * ( currentStress[0] + currentStress[1] ) );
            materialMatrix[3][4] = materialMatrix[4][3] -= ( 0.5 * currentStress[5] );
            materialMatrix[3][5] = materialMatrix[5][3] -= ( 0.5 * currentStress[4] );
            materialMatrix[4][5] = materialMatrix[5][4] -= ( 0.5 * currentStress[3] );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                materialStiffnessTemp[i][j] = 0.0;
                for ( int k = 0; k < 6; k++ ) {
                    // materialStiffnessTemp[i][j] += (materialMatrix[i][k] * Bl[k][j]);
                    materialStiffnessTemp[i][j] += ( materialMatrix[i][k] * Bl_np1[k][j] );
                }
            }
        }
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                materialStiffness[i][j] = 0.0;
                for ( int k = 0; k < 6; k++ ) {
                    // materialStiffness[i][j] += (Bl[k][i] * materialStiffnessTemp[k][j]);
                    materialStiffness[i][j] += ( Bl_np1[k][i] * materialStiffnessTemp[k][j] );
                }
            }
        }

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                bigStressMatrix[i][j]         = stressMatrix[i][j];
                bigStressMatrix[i + 3][j + 3] = stressMatrix[i][j];
                bigStressMatrix[i + 6][j + 6] = stressMatrix[i][j];
            }
        }

        // Computing the stress stiffness matrix.
        double tempMatrix[24][9];
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 9; j++ ) {
                tempMatrix[i][j] = 0.0;
                for ( int k = 0; k < 9; k++ ) {
                    // tempMatrix[i][j] += (Bnl[k][i] * bigStressMatrix[k][j]);
                    tempMatrix[i][j] += ( Bnl_np1[k][i] * bigStressMatrix[k][j] );
                }
            }
        }

        double tempStiffness[24][24];
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                tempStiffness[i][j] = 0.0;
                for ( int k = 0; k < 9; k++ ) {
                    tempStiffness[i][j] += ( tempMatrix[i][k] * Bnl_np1[k][j] );
                    // tempStiffness[i][j] += (tempMatrix[i][k] * Bnl[k][j]);
                }
            }
        }

        for ( unsigned int j = 0; j < num_nodes; j++ ) {
            for ( int d1 = 0; d1 < 3; d1++ ) {
                for ( unsigned int k = 0; k < num_nodes; k++ ) {
                    for ( int d2 = 0; d2 < 3; d2++ ) {

                        elementStiffnessMatrix[( 3 * k ) + d2][( 3 * j ) + d1] +=
                            ( detJ[0] * materialStiffness[( 3 * k ) + d2][( 3 * j ) + d1] );

                    } // end for d2
                }     // end for k
            }         // end for d1
        }             // end for j

        // Adding the stress stiffness matrix to the element stiffness matrix.
        for ( unsigned int i = 0; i < ( 3 * num_nodes ); i++ ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                // elementStiffnessMatrix[i][j] += (detJ_n[0] * tempStiffness[i][j]);
                elementStiffnessMatrix[i][j] += ( detJ[0] * tempStiffness[i][j] );
            }
        }

        d_materialModel->postLinearGaussPointOperation();
    } // end for qp

    d_materialModel->postLinearElementOperation();
}

void MechanicsLinearUpdatedLagrangianElement::apply_Normal()
{
    std::vector<std::vector<double>> &elementStiffnessMatrix = ( *d_elementStiffnessMatrix );

    std::vector<std::vector<double>> &elementInputVectors = d_elementInputVectors;

    std::vector<libMesh::Point> xyz, xyz_np1;
    double Bl_np1[6][24], Bnl_np1[9][24];
    double currX[8], currY[8], currZ[8];
    double rsq3              = ( 1.0 / std::sqrt( 3.0 ) );
    const double currXi[8]   = { -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3 };
    const double currEta[8]  = { -rsq3, -rsq3, rsq3, rsq3, -rsq3, -rsq3, rsq3, rsq3 };
    const double currZeta[8] = { -rsq3, -rsq3, -rsq3, -rsq3, rsq3, rsq3, rsq3, rsq3 };
    double dNdX[8], dNdY[8], dNdZ[8], refX[8], refY[8], refZ[8], detJ_0[1];
    // double prevX[8], prevY[8], prevZ[8];

    d_fe->reinit( d_elem );

    d_materialModel->preLinearElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize( num_nodes );
    xyz_np1.resize( num_nodes );

    for ( unsigned int ijk = 0; ijk < num_nodes; ijk++ ) {
        [[maybe_unused]] auto p1 = d_elem->point( ijk );
        // xyz[ijk] = p1;
        xyz[ijk]( 0 ) = d_elementRefXYZ[( 3 * ijk ) + 0];
        xyz[ijk]( 1 ) = d_elementRefXYZ[( 3 * ijk ) + 1];
        xyz[ijk]( 2 ) = d_elementRefXYZ[( 3 * ijk ) + 2];
    }

    for ( unsigned int ijk = 0; ijk < num_nodes; ijk++ ) {
        refX[ijk] = xyz[ijk]( 0 );
        refY[ijk] = xyz[ijk]( 1 );
        refZ[ijk] = xyz[ijk]( 2 );

        currX[ijk] = xyz_np1[ijk]( 0 ) =
            xyz[ijk]( 0 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 0];
        currY[ijk] = xyz_np1[ijk]( 1 ) =
            xyz[ijk]( 1 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 1];
        currZ[ijk] = xyz_np1[ijk]( 2 ) =
            xyz[ijk]( 2 ) + elementInputVectors[Mechanics::DISPLACEMENT][( 3 * ijk ) + 2];

        // prevX[ijk] = xyz[ijk](0);
        // prevY[ijk] = xyz[ijk](1);
        // prevZ[ijk] = xyz[ijk](2);
    }

    for ( int i = 0; i < 24; i++ ) {
        for ( int j = 0; j < 24; j++ ) {
            if ( elementStiffnessMatrix[i][j] > 1.0 ) {
                std::cout << "Initia case 2 elementStiffnessMatrix = "
                          << elementStiffnessMatrix[i][j] << std::endl;
            }
            elementStiffnessMatrix[i][j] = 0.0;
        }
    }

    /*    for(int i = 0; i < num_nodes; i++) {
          std::cout<<"currX["<<i<<"]="<<currX[i]<<"currY["<<i<<"]="<<currY[i]<<"currZ["<<i<<"]="<<currZ[i]<<std::endl;
        }
    */
    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preLinearGaussPointOperation();

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case prev-3 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<" qp =
           "<<qp<<std::endl;
                  }
                }
              }
        */
        double dNdx[8], dNdy[8], dNdz[8], detJ[1];

        constructShapeFunctionDerivatives(
            dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ );
        // constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, prevX, prevY, prevZ, currXi[qp],
        // currEta[qp],
        // currZeta[qp], detJ);

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case 3 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<std::endl;
                  }
                }
              }
        */
        for ( auto &elem : Bnl_np1 ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                elem[j] = 0.0;
            }
        }

        for ( auto &elem : Bl_np1 ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                elem[j] = 0.0;
            }
        }

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case 4 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<std::endl;
                  }
                }
              }
        */
        for ( unsigned int i = 0; i < ( num_nodes ); i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                Bnl_np1[( 3 * j ) + 0][( 3 * i ) + j] = dNdx[i];
                Bnl_np1[( 3 * j ) + 1][( 3 * i ) + j] = dNdy[i];
                Bnl_np1[( 3 * j ) + 2][( 3 * i ) + j] = dNdz[i];
            }
        }

        /*      for(int i = 0; i < num_nodes; i++) {
                std::cout<<"Linear-->dN_dxn["<<i<<"]="<<dN_dxn[i]<<" dN_dyn["<<i<<"]="<<dN_dyn[i]<<"
           dN_dzn["<<i<<"]="<<dN_dzn[i]<<std::endl;
              }
        */
        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            Bl_np1[0][( 3 * i ) + 0] = dNdx[i];
            Bl_np1[1][( 3 * i ) + 1] = dNdy[i];
            Bl_np1[2][( 3 * i ) + 2] = dNdz[i];
            Bl_np1[3][( 3 * i ) + 1] = dNdz[i];
            Bl_np1[3][( 3 * i ) + 2] = dNdy[i];
            Bl_np1[4][( 3 * i ) + 0] = dNdz[i];
            Bl_np1[4][( 3 * i ) + 2] = dNdx[i];
            Bl_np1[5][( 3 * i ) + 0] = dNdy[i];
            Bl_np1[5][( 3 * i ) + 1] = dNdx[i];
        }

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case 5 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<std::endl;
                  }
                }
              }
        */
        double constitutiveMatrix[6][6];
        double currentStress[6], materialMatrix[6][6], R_np1[3][3];

        for ( unsigned int i = 0; i < 3; i++ ) {
            for ( unsigned int j = 0; j < 3; j++ ) {
                R_np1[i][j] = 0.0;
            }
            R_np1[i][i] = 1.0;
        }

        if ( d_useJaumannRate == false ) {
            double F_np1[3][3];
            constructShapeFunctionDerivatives(
                dNdX, dNdY, dNdZ, refX, refY, refZ, currXi[qp], currEta[qp], currZeta[qp], detJ_0 );
            // computeDeformationGradientLin(dphi, *xyz_np1, num_nodes, qp, F_np1);
            computeGradient( dNdX, dNdY, dNdZ, currX, currY, currZ, num_nodes, F_np1 );
            if ( d_useFlanaganTaylorElem == false ) {
                double U_np1[3][3];
                polarDecompositionFeqRU_Simo( F_np1, R_np1, U_np1 );
            } else {
                for ( int i = 0; i < 3; i++ ) {
                    for ( int j = 0; j < 3; j++ ) {
                        R_np1[i][j] = 0.0;
                        if ( i == j )
                            R_np1[i][j] = 1.0;
                    }
                }
            }
        }

        d_materialModel->getConstitutiveMatrixUpdatedLagrangian( constitutiveMatrix, R_np1 );
        d_materialModel->getStressForUpdatedLagrangian( currentStress );
        /*      for(int i = 0; i < 6; i++) {
                std::cout<<"currentStress["<<i<<"]="<<currentStress[i]<<std::endl;
              }
        */

        double stressMatrix[3][3], bigStressMatrix[9][9];
        stressMatrix[0][0] = currentStress[0];
        stressMatrix[1][1] = currentStress[1];
        stressMatrix[2][2] = currentStress[2];
        stressMatrix[1][2] = stressMatrix[2][1] = currentStress[3];
        stressMatrix[0][2] = stressMatrix[2][0] = currentStress[4];
        stressMatrix[0][1] = stressMatrix[1][0] = currentStress[5];

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                materialMatrix[i][j] = constitutiveMatrix[i][j];
            }
        }

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case 5 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<std::endl;
                  }
                }
              }
        */
        if ( d_useJaumannRate == true ) {
            materialMatrix[0][0] -= ( 2.0 * currentStress[0] );
            materialMatrix[1][1] -= ( 2.0 * currentStress[1] );
            materialMatrix[2][2] -= ( 2.0 * currentStress[2] );
            materialMatrix[3][1] = materialMatrix[3][2] = materialMatrix[1][3] =
                materialMatrix[2][3] -= currentStress[3];
            materialMatrix[4][0] = materialMatrix[4][2] = materialMatrix[0][4] =
                materialMatrix[2][4] -= currentStress[4];
            materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[1][5] =
                materialMatrix[0][5] -= currentStress[5];
            materialMatrix[3][3] -= ( 0.5 * ( currentStress[1] + currentStress[2] ) );
            materialMatrix[4][4] -= ( 0.5 * ( currentStress[0] + currentStress[2] ) );
            materialMatrix[5][5] -= ( 0.5 * ( currentStress[0] + currentStress[1] ) );
            materialMatrix[3][4] = materialMatrix[4][3] -= ( 0.5 * currentStress[5] );
            materialMatrix[3][5] = materialMatrix[5][3] -= ( 0.5 * currentStress[4] );
            materialMatrix[4][5] = materialMatrix[5][4] -= ( 0.5 * currentStress[3] );
        }

        double materialStiffnessTemp[6][24];
        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                materialStiffnessTemp[i][j] = 0.0;
                for ( int k = 0; k < 6; k++ ) {
                    materialStiffnessTemp[i][j] += ( materialMatrix[i][k] * Bl_np1[k][j] );
                }
            }
        }
        double materialStiffness[24][24];
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                materialStiffness[i][j] = 0.0;
                for ( int k = 0; k < 6; k++ ) {
                    materialStiffness[i][j] += ( Bl_np1[k][i] * materialStiffnessTemp[k][j] );
                }
            }
        }

        for ( auto &elem : bigStressMatrix ) {
            for ( double &j : elem ) {
                j = 0.0;
            }
        }

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                bigStressMatrix[i][j]         = stressMatrix[i][j];
                bigStressMatrix[i + 3][j + 3] = stressMatrix[i][j];
                bigStressMatrix[i + 6][j + 6] = stressMatrix[i][j];
            }
        }

        // Computing the stress stiffness matrix.
        double tempMatrix[24][9];
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 9; j++ ) {
                tempMatrix[i][j] = 0.0;
                for ( int k = 0; k < 9; k++ ) {
                    tempMatrix[i][j] += ( Bnl_np1[k][i] * bigStressMatrix[k][j] );
                }
            }
        }

        double tempStiffness[24][24];
        for ( int i = 0; i < 24; i++ ) {
            for ( int j = 0; j < 24; j++ ) {
                tempStiffness[i][j] = 0.0;
                for ( int k = 0; k < 9; k++ ) {
                    tempStiffness[i][j] += ( tempMatrix[i][k] * Bnl_np1[k][j] );
                }
            }
        }

        /*      for(int i = 0; i < 24; i++) {
                for(int j = 0; j < 24; j++) {
                  if(elementStiffnessMatrix[i][j] > 100.0) {
                    std::cout<<"Initia case 6 elementStiffnessMatrix =
           "<<elementStiffnessMatrix[i][j]<<std::endl;
                  }
                }
              }
        */
        for ( unsigned int j = 0; j < num_nodes; j++ ) {
            for ( int d1 = 0; d1 < 3; d1++ ) {

                for ( unsigned int k = 0; k < num_nodes; k++ ) {
                    for ( int d2 = 0; d2 < 3; d2++ ) {

                        // elementStiffnessMatrix[(3*k) + d2][(3*j) + d1] += (detJ[0]*tmp);
                        // elementStiffnessMatrix[(3*k) + d2][(3*j) + d1] += (detJ_n[0]*tmp);
                        // elementStiffnessMatrix[(3 * k) + d2][(3 * j) + d1] += (detJ_n[0] *
                        // materialStiffness[(3 * k)
                        // + d2][(3 * j) + d1]);
                        elementStiffnessMatrix[( 3 * k ) + d2][( 3 * j ) + d1] +=
                            ( detJ[0] * materialStiffness[( 3 * k ) + d2][( 3 * j ) + d1] );
                        // AMP::pout<<"elementStiffnessMatrix["<<(3 * k) + d2<<"]["<<(3 * j) +
                        // d1<<"]="<<elementStiffnessMatrix[(3 * k) + d2][(3 * j) + d1]<<std::endl;

                    } // end for d2
                }     // end for k
            }         // end for d1
        }             // end for j

        // Adding the stress stiffness matrix to the element stiffness matrix.
        for ( unsigned int i = 0; i < ( 3 * num_nodes ); i++ ) {
            for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                // elementStiffnessMatrix[i][j] += (detJ_n[0] * tempStiffness[i][j]);
                elementStiffnessMatrix[i][j] += ( detJ[0] * tempStiffness[i][j] );
                // AMP::pout<<"elementStiffnessMatrix["<<i<<"]["<<j<<"] =
                // "<<elementStiffnessMatrix[i][j]<<std::endl;
            }
        }

        if ( d_iDebugPrintInfoLevel > 11 ) {
            for ( unsigned int i = 0; i < ( 3 * num_nodes ); i++ ) {
                for ( unsigned int j = 0; j < ( 3 * num_nodes ); j++ ) {
                    AMP::pout << "elementStiffnessMatrix[" << i << "][" << j
                              << "] = " << elementStiffnessMatrix[i][j] << std::endl;
                }
            }
        }

        d_materialModel->postLinearGaussPointOperation();
    } // end for qp

    d_materialModel->postLinearElementOperation();
}

/*
  void MechanicsLinearUpdatedLagrangianElement :: computeDeformationGradientLin(const
  std::vector<std::vector<libMesh::RealGradient> > & dphi,
      const std::vector<Point> & xyz, unsigned int num_nodes, unsigned int qp, double F[3][3])
  {
    for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
        F[i][j] = 0.0;
      }
    }

    for(unsigned int k = 0; k < num_nodes; k++) {
      F[0][0] += (xyz[k](0) * dphi[k][qp](0));
      F[0][1] += (xyz[k](0) * dphi[k][qp](1));
      F[0][2] += (xyz[k](0) * dphi[k][qp](2));
      F[1][0] += (xyz[k](1) * dphi[k][qp](0));
      F[1][1] += (xyz[k](1) * dphi[k][qp](1));
      F[1][2] += (xyz[k](1) * dphi[k][qp](2));
      F[2][0] += (xyz[k](2) * dphi[k][qp](0));
      F[2][1] += (xyz[k](2) * dphi[k][qp](1));
      F[2][2] += (xyz[k](2) * dphi[k][qp](2));
    }
  }
*/

void MechanicsLinearUpdatedLagrangianElement::initializeReferenceXYZ(
    std::vector<double> &elementRefXYZ )
{
    std::vector<libMesh::Point> xyz;

    d_fe->reinit( d_elem );

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize( num_nodes );

    for ( unsigned int ijk = 0; ijk < num_nodes; ijk++ ) {
        auto p1  = d_elem->point( ijk );
        xyz[ijk] = p1;

        elementRefXYZ[( 3 * ijk ) + 0] = xyz[ijk]( 0 );
        elementRefXYZ[( 3 * ijk ) + 1] = xyz[ijk]( 1 );
        elementRefXYZ[( 3 * ijk ) + 2] = xyz[ijk]( 2 );
    } // end of ijk.
}
} // namespace AMP::Operator
