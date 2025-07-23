#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

void MechanicsNonlinearElement::computeStressAndStrain(
    const std::vector<std::vector<double>> &elementInputVectors,
    std::vector<double> &stressVec,
    std::vector<double> &strainVec )
{
    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    d_fe->reinit( d_elem );

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearAssemblyGaussPointOperation();

        /*      // Compute Strain From Given Displacement

              double dudx = 0;
              double dudy = 0;
              double dudz = 0;
              double dvdx = 0;
              double dvdy = 0;
              double dvdz = 0;
              double dwdx = 0;
              double dwdy = 0;
              double dwdz = 0;

              for(unsigned int k = 0; k < num_nodes; k++) {
                dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
                dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
                dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

                dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
                dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
                dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

                dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
                dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
                dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
              } //end for k

              std::vector<std::vector<double> >
           fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES);

              //Strain
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));
        */
        double Bl_np1[6][24], el_np1[6];

        for ( int i = 0; i < 6; i++ ) {
            el_np1[i] = 0.0;
            for ( int j = 0; j < 24; j++ ) {
                Bl_np1[i][j] = 0.0;
            }
        }

        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            Bl_np1[0][( 3 * i ) + 0] = dphi[i][qp]( 0 );
            Bl_np1[1][( 3 * i ) + 1] = dphi[i][qp]( 1 );
            Bl_np1[2][( 3 * i ) + 2] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 1] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 2] = dphi[i][qp]( 1 );
            Bl_np1[4][( 3 * i ) + 0] = dphi[i][qp]( 2 );
            Bl_np1[4][( 3 * i ) + 2] = dphi[i][qp]( 0 );
            Bl_np1[5][( 3 * i ) + 0] = dphi[i][qp]( 1 );
            Bl_np1[5][( 3 * i ) + 1] = dphi[i][qp]( 0 );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( unsigned int j = 0; j < num_nodes; j++ ) {
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 0] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 0] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 1] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 1] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 2] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 2] );
            }
        }

        std::vector<std::vector<double>> fieldsAtGaussPt( Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[0] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[1] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[2] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[3] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[4] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[5] );

        if ( !( elementInputVectors[Mechanics::TEMPERATURE].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::TEMPERATURE][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::BURNUP].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::BURNUP][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::BURNUP].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt +=
                    ( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::LHGR].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::LHGR][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::LHGR].push_back( valAtGaussPt );
        }

        double *currStress;

        /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */

        d_materialModel->getInternalStress( fieldsAtGaussPt, currStress );

        for ( int i = 0; i < 6; i++ ) {
            stressVec[( 6 * qp ) + i] = currStress[i];
            strainVec[( 6 * qp ) + i] = fieldsAtGaussPt[Mechanics::DISPLACEMENT][i];
        } // end for i

        d_materialModel->postNonlinearAssemblyGaussPointOperation();
    } // end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
}


void MechanicsNonlinearElement::printStressAndStrain(
    FILE *fp, const std::vector<std::vector<double>> &elementInputVectors )
{
    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    const auto &xyz = ( *d_xyz );

    d_fe->reinit( d_elem );

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearAssemblyGaussPointOperation();

        /*      // Compute Strain From Given Displacement

              double dudx = 0;
              double dudy = 0;
              double dudz = 0;
              double dvdx = 0;
              double dvdy = 0;
              double dvdz = 0;
              double dwdx = 0;
              double dwdy = 0;
              double dwdz = 0;

              for(unsigned int k = 0; k < num_nodes; k++) {
                dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
                dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
                dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

                dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
                dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
                dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

                dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
                dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
                dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
              } //end for k

              std::vector<std::vector<double> >
           fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES);

              //Strain
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));
        */
        double Bl_np1[6][24], el_np1[6];

        for ( int i = 0; i < 6; i++ ) {
            el_np1[i] = 0.0;
            for ( int j = 0; j < 24; j++ ) {
                Bl_np1[i][j] = 0.0;
            }
        }

        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            Bl_np1[0][( 3 * i ) + 0] = dphi[i][qp]( 0 );
            Bl_np1[1][( 3 * i ) + 1] = dphi[i][qp]( 1 );
            Bl_np1[2][( 3 * i ) + 2] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 1] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 2] = dphi[i][qp]( 1 );
            Bl_np1[4][( 3 * i ) + 0] = dphi[i][qp]( 2 );
            Bl_np1[4][( 3 * i ) + 2] = dphi[i][qp]( 0 );
            Bl_np1[5][( 3 * i ) + 0] = dphi[i][qp]( 1 );
            Bl_np1[5][( 3 * i ) + 1] = dphi[i][qp]( 0 );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( unsigned int j = 0; j < num_nodes; j++ ) {
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 0] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 0] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 1] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 1] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 2] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 2] );
            }
        }

        std::vector<std::vector<double>> fieldsAtGaussPt( Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[0] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[1] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[2] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[3] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[4] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[5] );

        if ( !( elementInputVectors[Mechanics::TEMPERATURE].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::TEMPERATURE][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::BURNUP].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::BURNUP][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::BURNUP].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt +=
                    ( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::LHGR].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::LHGR][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::LHGR].push_back( valAtGaussPt );
        }

        double *currStress;

        /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */

        d_materialModel->getInternalStress( fieldsAtGaussPt, currStress );

        fprintf( fp, "%.12f %.12f %.12f \n", xyz[qp]( 0 ), xyz[qp]( 1 ), xyz[qp]( 2 ) );
        fprintf( fp,
                 "%.12f %.12f %.12f %.12f %.12f %.12f \n",
                 currStress[0],
                 currStress[1],
                 currStress[2],
                 currStress[3],
                 currStress[4],
                 currStress[5] );
        fprintf( fp,
                 "%.12f %.12f %.12f %.12f %.12f %.12f \n",
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][0],
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][1],
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][2],
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][3],
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][4],
                 fieldsAtGaussPt[Mechanics::DISPLACEMENT][5] );

        d_materialModel->postNonlinearAssemblyGaussPointOperation();
    } // end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
}

void MechanicsNonlinearElement::initMaterialModel( const std::vector<double> &initTempVector )
{
    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    d_fe->reinit( d_elem );

    d_materialModel->preNonlinearInitElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearInitGaussPointOperation();

        double tempAtGaussPt = 0;
        if ( !( initTempVector.empty() ) ) {
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                tempAtGaussPt += ( initTempVector[k] * phi[k][qp] );
            } // end for k
        }

        d_materialModel->nonlinearInitGaussPointOperation( tempAtGaussPt );

        d_materialModel->postNonlinearInitGaussPointOperation();
    } // end for qp

    d_materialModel->postNonlinearInitElementOperation();
}

void MechanicsNonlinearElement::apply_Normal()
{
    const std::vector<libMesh::Real> &JxW = ( *d_JxW );

    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    std::vector<std::vector<double>> &elementInputVectors = d_elementInputVectors;

    std::vector<double> &elementOutputVector = ( *d_elementOutputVector );

    std::vector<libMesh::Point> xyz;

    d_fe->reinit( d_elem );

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize( num_nodes );

    /*
        Point p1;
        for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
          p1 = d_elem->point(ijk);
          xyz[ijk] = p1;
        }
    */

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearAssemblyGaussPointOperation();

        // The value of radius is very important.
        // double radius = std::sqrt((xyz[qp](0) * xyz[qp](0)) + (xyz[qp](1) * xyz[qp](1)));

        /* Compute Strain From Given Displacement */

        /*      double dudx = 0;
              double dudy = 0;
              double dudz = 0;
              double dvdx = 0;
              double dvdy = 0;
              double dvdz = 0;
              double dwdx = 0;
              double dwdy = 0;
              double dwdz = 0;

              for(unsigned int k = 0; k < num_nodes; k++) {
                dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
                dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
                dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

                dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
                dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
                dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

                dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
                dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
                dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
              } //end for k

              std::vector<std::vector<double> >
           fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES);

              //Strain
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz);
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));
        */
        double Bl_np1[6][24], el_np1[6];

        for ( int i = 0; i < 6; i++ ) {
            el_np1[i] = 0.0;
            for ( int j = 0; j < 24; j++ ) {
                Bl_np1[i][j] = 0.0;
            }
        }

        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            Bl_np1[0][( 3 * i ) + 0] = dphi[i][qp]( 0 );
            Bl_np1[1][( 3 * i ) + 1] = dphi[i][qp]( 1 );
            Bl_np1[2][( 3 * i ) + 2] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 1] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 2] = dphi[i][qp]( 1 );
            Bl_np1[4][( 3 * i ) + 0] = dphi[i][qp]( 2 );
            Bl_np1[4][( 3 * i ) + 2] = dphi[i][qp]( 0 );
            Bl_np1[5][( 3 * i ) + 0] = dphi[i][qp]( 1 );
            Bl_np1[5][( 3 * i ) + 1] = dphi[i][qp]( 0 );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( unsigned int j = 0; j < num_nodes; j++ ) {
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 0] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 0] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 1] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 1] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 2] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 2] );
            }
        }

        std::vector<std::vector<double>> fieldsAtGaussPt( Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[0] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[1] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[2] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[3] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[4] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[5] );

        if ( !( elementInputVectors[Mechanics::TEMPERATURE].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::TEMPERATURE][k] * phi[k][qp] );
            } // end for k
            // valAtGaussPt = ((valAtGaussPt - 301.0) * (1.25 - (radius * radius))) + 301.0;
            /*        if(radius <= 0.0025527) {
          valAtGaussPt = 791.1018 - (26950800.0 * radius * radius);
        }
        if((radius > 0.0025527) && (radius <= 0.003175)) {
          valAtGaussPt = 49.3983 - (94.8119 * log(radius));
        }
        if((radius > 0.003175) && (radius < 0.00319405)) {
          valAtGaussPt = 532.0;
        }
        if((radius >= 0.00319405) && (radius <= 0.00465709)) {
          valAtGaussPt = (-122.0998 * log(radius)) - 225.3706;
        }
        if((radius > 0.00465709) && (radius <= 0.004665626)) {
          valAtGaussPt = (-9392.5217 * log(radius)) - 50001.047;
        }
        if(radius > 0.004665626) {
          valAtGaussPt = 414.0;
        }
*/ // This part of the code is used to assign a
            // temperature distribution by
            // hand.
            /*        if(radius <= 0.0025527) {
          valAtGaussPt = 772.8084 - (17315315.440195 * radius * radius);
        }
        if((radius > 0.0025527) && (radius <= 0.003175)) {
          valAtGaussPt = 17.5302 - (107.6016 * log(radius));
        }
        if((radius > 0.003175) && (radius < 0.00319405)) {
          valAtGaussPt = 569.2452;
        }
        if((radius >= 0.00319405) && (radius <= 0.00465709)) {
          valAtGaussPt = (-138.5565 * log(radius)) - 294.2228;
        }
        if((radius > 0.00465709) && (radius <= 0.004665626)) {
          valAtGaussPt = (-10658.1931 * log(radius)) - 56777.9863;
        }
        if(radius > 0.004665626) {
          valAtGaussPt = 430.22;
        }
*/ // This part of the code is used to assign a
            // temperature distribution by
            // hand.
            fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::BURNUP].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::BURNUP][k] * phi[k][qp] );
            } // end for k
            // valAtGaussPt = (valAtGaussPt / 2.0) * exp(radius);
            fieldsAtGaussPt[Mechanics::BURNUP].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt +=
                    ( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::LHGR].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::LHGR][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::LHGR].push_back( valAtGaussPt );
        }

        double *currStress;

        /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */

        d_materialModel->getInternalStress( fieldsAtGaussPt, currStress );

        /* Convert Voigt Notation into Matrix Notation */
        /*
              double T[3][3];

              T[0][0] = currStress[0]; //xx
              T[0][1] = currStress[5]; //xy
              T[0][2] = currStress[4]; //xz

              T[1][0] = currStress[5]; //yx
              T[1][1] = currStress[1]; //yy
              T[1][2] = currStress[3]; //yz

              T[2][0] = currStress[4]; //zx
              T[2][1] = currStress[3]; //zy
              T[2][2] = currStress[2]; //zz
        */
        for ( unsigned int j = 0; j < num_nodes; j++ ) {
            for ( unsigned int d = 0; d < 3; d++ ) {

                double tmp = 0;
                for ( unsigned int m = 0; m < 6; m++ ) {
                    tmp += ( Bl_np1[m][( 3 * j ) + d] * currStress[m] );
                }


                /*          for(int d1 = 0; d1 < 3; d1++) {
                            tmp += (T[d1][d]*dphi[j][qp](d1));
                          }//end for d1
                */
                /*

                //Strain-Displacement Relation (Voigt Notation)

                double tmpStrain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

                if(d == 0) {
                tmpStrain[0] = dphi[j][qp](0); //xx
                tmpStrain[5] = (0.5*dphi[j][qp](1)); //xy
                tmpStrain[4] = (0.5*dphi[j][qp](2)); //xz
                }

                if(d == 1) {
                tmpStrain[5] = (0.5*dphi[j][qp](0)); //yx
                tmpStrain[1] = dphi[j][qp](1); //yy
                tmpStrain[3] = (0.5*dphi[j][qp](2)); //yz
                }

                if(d == 2) {
                tmpStrain[4] = (0.5*dphi[j][qp](0)); //zx
                tmpStrain[3] = (0.5*dphi[j][qp](1)); //zy
                tmpStrain[2] = dphi[j][qp](2); //zz
                }

                //Convert Voigt Notation to Matrix Notation

                double S[3][3];

                S[0][0] = tmpStrain[0]; //xx
                S[0][1] = tmpStrain[5]; //xy
                S[0][2] = tmpStrain[4]; //xz

                S[1][0] = tmpStrain[5]; //yx
                S[1][1] = tmpStrain[1]; //yy
                S[1][2] = tmpStrain[3]; //yz

                S[2][0] = tmpStrain[4]; //zx
                S[2][1] = tmpStrain[3]; //zy
                S[2][2] = tmpStrain[2]; //zz

                //tr{TS}
                double tmp = 0;
                for(int d1 = 0; d1 < 3; d1++) {
                for(int d2 = 0; d2 < 3; d2++) {
                tmp += (T[d1][d2]*S[d2][d1]);
                }//end for d2
                }//end for d1

      */

                elementOutputVector[( 3 * j ) + d] += ( JxW[qp] * tmp );

            } // end for d
        }     // end for j

        d_materialModel->postNonlinearAssemblyGaussPointOperation();
    } // end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
}

void MechanicsNonlinearElement::apply_Reduced()
{
    const std::vector<libMesh::Real> &JxW = ( *d_JxW );

    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    std::vector<std::vector<double>> &elementInputVectors = d_elementInputVectors;

    std::vector<double> &elementOutputVector = ( *d_elementOutputVector );

    d_fe->reinit( d_elem );

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    std::vector<double> avgTraceTerm( num_nodes * 3 );
    for ( unsigned int i = 0; i < ( 3 * num_nodes ); i++ ) {
        avgTraceTerm[i] = 0.0;
    } // end for i

    double sumJxW = 0.0;
    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        sumJxW += JxW[qp];

        for ( unsigned int j = 0; j < num_nodes; j++ ) {
            for ( int d = 0; d < 3; d++ ) {
                avgTraceTerm[( 3 * j ) + d] += ( JxW[qp] * dphi[j][qp]( d ) );
            } // end for d
        }     // end for j
    }         // end for qp

    for ( unsigned int i = 0; i < ( 3 * num_nodes ); i++ ) {
        avgTraceTerm[i] = avgTraceTerm[i] / sumJxW;
    } // end for i

    /*
        double avgDivDisp = 0.0;
        for(unsigned int j = 0; j < num_nodes; j++) {
          for(int d = 0; d < 3; d++) {
            avgDivDisp += (avgTraceTerm[(3*j) +
       d]*elementInputVectors[Mechanics::DISPLACEMENT][(3*j) + d]);
          }//end for d
        }//end for j
    */

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_materialModel->preNonlinearAssemblyGaussPointOperation();

        double Bl_np1[6][24], el_np1[6];

        for ( int i = 0; i < 6; i++ ) {
            el_np1[i] = 0.0;
            for ( int j = 0; j < 24; j++ ) {
                Bl_np1[i][j] = 0.0;
            }
        }

        for ( unsigned int i = 0; i < num_nodes; i++ ) {
            double one3 = 1.0 / 3.0;
            Bl_np1[0][( 3 * i ) + 0] =
                dphi[i][qp]( 0 ) + ( one3 * ( avgTraceTerm[( 3 * i ) + 0] - dphi[i][qp]( 0 ) ) );
            Bl_np1[1][( 3 * i ) + 0] = one3 * ( avgTraceTerm[( 3 * i ) + 0] - dphi[i][qp]( 0 ) );
            Bl_np1[2][( 3 * i ) + 0] = one3 * ( avgTraceTerm[( 3 * i ) + 0] - dphi[i][qp]( 0 ) );
            Bl_np1[0][( 3 * i ) + 1] = one3 * ( avgTraceTerm[( 3 * i ) + 1] - dphi[i][qp]( 1 ) );
            Bl_np1[1][( 3 * i ) + 1] =
                dphi[i][qp]( 1 ) + ( one3 * ( avgTraceTerm[( 3 * i ) + 1] - dphi[i][qp]( 1 ) ) );
            Bl_np1[2][( 3 * i ) + 1] = one3 * ( avgTraceTerm[( 3 * i ) + 1] - dphi[i][qp]( 1 ) );
            Bl_np1[0][( 3 * i ) + 2] = one3 * ( avgTraceTerm[( 3 * i ) + 2] - dphi[i][qp]( 2 ) );
            Bl_np1[1][( 3 * i ) + 2] = one3 * ( avgTraceTerm[( 3 * i ) + 2] - dphi[i][qp]( 2 ) );
            Bl_np1[2][( 3 * i ) + 2] =
                dphi[i][qp]( 2 ) + ( one3 * ( avgTraceTerm[( 3 * i ) + 2] - dphi[i][qp]( 2 ) ) );
            Bl_np1[3][( 3 * i ) + 1] = dphi[i][qp]( 2 );
            Bl_np1[3][( 3 * i ) + 2] = dphi[i][qp]( 1 );
            Bl_np1[4][( 3 * i ) + 0] = dphi[i][qp]( 2 );
            Bl_np1[4][( 3 * i ) + 2] = dphi[i][qp]( 0 );
            Bl_np1[5][( 3 * i ) + 0] = dphi[i][qp]( 1 );
            Bl_np1[5][( 3 * i ) + 1] = dphi[i][qp]( 0 );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( unsigned int j = 0; j < num_nodes; j++ ) {
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 0] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 0] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 1] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 1] );
                el_np1[i] += ( Bl_np1[i][( 3 * j ) + 2] *
                               elementInputVectors[Mechanics::DISPLACEMENT][( 3 * j ) + 2] );
            }
        }

        std::vector<std::vector<double>> fieldsAtGaussPt( Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[0] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[1] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( el_np1[2] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[3] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[4] );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 1.0 * el_np1[5] );

        // Compute Strain From Given Displacement

        /*      double dudx = 0;
              double dudy = 0;
              double dudz = 0;
              double dvdx = 0;
              double dvdy = 0;
              double dvdz = 0;
              double dwdx = 0;
              double dwdy = 0;
              double dwdz = 0;

              for(unsigned int k = 0; k < num_nodes; k++) {
                dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
                dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
                dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

                dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
                dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
                dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

                dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
                dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
                dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
              } //end for k

              double divDisp = (dudx + dvdy + dwdz);

              std::vector<std::vector<double> >
           fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES);

              //Strain
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx + ((avgDivDisp -
           divDisp)/3.0));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy + ((avgDivDisp -
           divDisp)/3.0));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz + ((avgDivDisp -
           divDisp)/3.0));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
              fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));
        */
        if ( !( elementInputVectors[Mechanics::TEMPERATURE].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::TEMPERATURE][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::BURNUP].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::BURNUP][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::BURNUP].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt +=
                    ( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::LHGR].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::LHGR][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::LHGR].push_back( valAtGaussPt );
        }

        double *currStress;

        // Compute Stress Corresponding to Given Strain, Temperature, Burnup, ...

        d_materialModel->getInternalStress( fieldsAtGaussPt, currStress );

        // Convert Voigt Notation into Matrix Notation
        /*
              double T[3][3];

              T[0][0] = currStress[0]; //xx
              T[0][1] = currStress[5]; //xy
              T[0][2] = currStress[4]; //xz

              T[1][0] = currStress[5]; //yx
              T[1][1] = currStress[1]; //yy
              T[1][2] = currStress[3]; //yz

              T[2][0] = currStress[4]; //zx
              T[2][1] = currStress[3]; //zy
              T[2][2] = currStress[2]; //zz
        */
        for ( unsigned int j = 0; j < num_nodes; j++ ) {
            for ( unsigned int d = 0; d < 3; d++ ) {

                double tmp = 0;
                for ( unsigned int m = 0; m < 6; m++ ) {
                    tmp += ( Bl_np1[m][( 3 * j ) + d] * currStress[m] );
                }


                /*          // Strain-Displacement Relation (Voigt Notation)

                          double tmpStrain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

                          if(d == 0) {
                            tmpStrain[0] = dphi[j][qp](0); //xx
                            tmpStrain[5] = (0.5*dphi[j][qp](1)); //xy
                            tmpStrain[4] = (0.5*dphi[j][qp](2)); //xz
                          }

                          if(d == 1) {
                            tmpStrain[5] = (0.5*dphi[j][qp](0)); //yx
                            tmpStrain[1] = dphi[j][qp](1); //yy
                            tmpStrain[3] = (0.5*dphi[j][qp](2)); //yz
                          }

                          if(d == 2) {
                            tmpStrain[4] = (0.5*dphi[j][qp](0)); //zx
                            tmpStrain[3] = (0.5*dphi[j][qp](1)); //zy
                            tmpStrain[2] = dphi[j][qp](2); //zz
                          }

                          for(int c = 0; c < 3; c++) {
                            tmpStrain[c] = tmpStrain[c] + ((avgTraceTerm[(3*j) + d] -
                   dphi[j][qp](d))/3.0);
                          }//end for c

                          // Convert Voigt Notation to Matrix Notation

                          double S[3][3];

                          S[0][0] = tmpStrain[0]; //xx
                          S[0][1] = tmpStrain[5]; //xy
                          S[0][2] = tmpStrain[4]; //xz

                          S[1][0] = tmpStrain[5]; //yx
                          S[1][1] = tmpStrain[1]; //yy
                          S[1][2] = tmpStrain[3]; //yz

                          S[2][0] = tmpStrain[4]; //zx
                          S[2][1] = tmpStrain[3]; //zy
                          S[2][2] = tmpStrain[2]; //zz

                          // tr{TS}
                          double tmp = 0;
                          for(int d1 = 0; d1 < 3; d1++) {
                            for(int d2 = 0; d2 < 3; d2++) {
                              tmp += (T[d1][d2]*S[d2][d1]);
                            }
                          }
                */
                elementOutputVector[( 3 * j ) + d] += ( JxW[qp] * tmp );

            } // end for d
        }     // end for j

        d_materialModel->postNonlinearAssemblyGaussPointOperation();
    } // end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
}

void MechanicsNonlinearElement::materialModelPreNonlinearElementOperation( MaterialUpdateType type )
{
    if ( type == MaterialUpdateType::RESET )
        d_materialModel->preNonlinearResetElementOperation();
    else if ( type == MaterialUpdateType::JACOBIAN )
        d_materialModel->preNonlinearJacobianElementOperation();
}

void MechanicsNonlinearElement::materialModelPreNonlinearGaussPointOperation(
    MaterialUpdateType type )
{
    if ( type == MaterialUpdateType::RESET )
        d_materialModel->preNonlinearResetGaussPointOperation();
    else if ( type == MaterialUpdateType::JACOBIAN )
        d_materialModel->preNonlinearJacobianGaussPointOperation();
}

void MechanicsNonlinearElement::materialModelPostNonlinearElementOperation(
    MaterialUpdateType type )
{
    if ( type == MaterialUpdateType::RESET )
        d_materialModel->postNonlinearResetElementOperation();
    else if ( type == MaterialUpdateType::JACOBIAN )
        d_materialModel->postNonlinearJacobianElementOperation();
}

void MechanicsNonlinearElement::materialModelPostNonlinearGaussPointOperation(
    MaterialUpdateType type )
{
    if ( type == MaterialUpdateType::RESET )
        d_materialModel->postNonlinearResetGaussPointOperation();
    else if ( type == MaterialUpdateType::JACOBIAN )
        d_materialModel->postNonlinearJacobianGaussPointOperation();
}

void MechanicsNonlinearElement::materialModelNonlinearGaussPointOperation(
    MaterialUpdateType type, const std::vector<std::vector<double>> &strain )
{
    if ( type == MaterialUpdateType::RESET )
        d_materialModel->nonlinearResetGaussPointOperation( strain );
    else if ( type == MaterialUpdateType::JACOBIAN )
        d_materialModel->nonlinearJacobianGaussPointOperation( strain );
}

void MechanicsNonlinearElement::updateMaterialModel(
    MaterialUpdateType type, const std::vector<std::vector<double>> &elementInputVectors )
{
    const std::vector<std::vector<libMesh::RealGradient>> &dphi = ( *d_dphi );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    d_fe->reinit( d_elem );

    materialModelPreNonlinearElementOperation( type );

    const unsigned int num_nodes = d_elem->n_nodes();

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        materialModelPreNonlinearGaussPointOperation( type );

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
            dudx +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 0] * dphi[k][qp]( 0 ) );
            dudy +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 0] * dphi[k][qp]( 1 ) );
            dudz +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 0] * dphi[k][qp]( 2 ) );

            dvdx +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 1] * dphi[k][qp]( 0 ) );
            dvdy +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 1] * dphi[k][qp]( 1 ) );
            dvdz +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 1] * dphi[k][qp]( 2 ) );

            dwdx +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 2] * dphi[k][qp]( 0 ) );
            dwdy +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 2] * dphi[k][qp]( 1 ) );
            dwdz +=
                ( elementInputVectors[Mechanics::DISPLACEMENT][( 3 * k ) + 2] * dphi[k][qp]( 2 ) );
        } // end for k

        std::vector<std::vector<double>> fieldsAtGaussPt( Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        // Strain
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( dudx );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( dvdy );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( dwdz );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 0.5 * ( dvdz + dwdy ) );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 0.5 * ( dudz + dwdx ) );
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back( 0.5 * ( dudy + dvdx ) );

        if ( !( elementInputVectors[Mechanics::TEMPERATURE].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::TEMPERATURE][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::BURNUP].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::BURNUP][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::BURNUP].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt +=
                    ( elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back( valAtGaussPt );
        }

        if ( !( elementInputVectors[Mechanics::LHGR].empty() ) ) {
            double valAtGaussPt = 0;
            for ( unsigned int k = 0; k < num_nodes; k++ ) {
                valAtGaussPt += ( elementInputVectors[Mechanics::LHGR][k] * phi[k][qp] );
            } // end for k
            fieldsAtGaussPt[Mechanics::LHGR].push_back( valAtGaussPt );
        }

        materialModelNonlinearGaussPointOperation( type, fieldsAtGaussPt );

        materialModelPostNonlinearGaussPointOperation( type );
    } // end for qp

    materialModelPostNonlinearElementOperation( type );
}


} // namespace AMP::Operator
