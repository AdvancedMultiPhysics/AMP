#include "AMP/operators/mechanics/ConstructLinearMechanicsRHSVector.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/auto_ptr.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


void computeTemperatureRhsVector( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                  std::shared_ptr<AMP::Database> input_db,
                                  std::shared_ptr<AMP::LinearAlgebra::Variable>,
                                  std::shared_ptr<AMP::LinearAlgebra::Variable> displacementVar,
                                  std::shared_ptr<AMP::LinearAlgebra::Vector> currTemperatureVec,
                                  std::shared_ptr<AMP::LinearAlgebra::Vector> prevTemperatureVec,
                                  AMP::LinearAlgebra::Vector::shared_ptr rhsVec )
{
    currTemperatureVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    prevTemperatureVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    auto rInternal = rhsVec->subsetVectorForVariable( displacementVar );
    rInternal->zero();

    auto elementRhsDatabase    = input_db->getDatabase( "RhsElements" );
    auto materialModelDatabase = input_db->getDatabase( "RhsMaterialModel" );

    std::shared_ptr<libMesh::FEType> feType;
    std::shared_ptr<libMesh::FEBase> fe;
    std::shared_ptr<libMesh::QBase> qrule;

    auto feTypeOrderName = elementRhsDatabase->getWithDefault<std::string>( "FE_ORDER", "FIRST" );
    auto feTypeOrder     = libMesh::Utility::string_to_enum<libMeshEnums::Order>( feTypeOrderName );

    auto feFamilyName = elementRhsDatabase->getWithDefault<std::string>( "FE_FAMILY", "LAGRANGE" );
    auto feFamily     = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( feFamilyName );

    auto qruleTypeName = elementRhsDatabase->getWithDefault<std::string>( "QRULE_TYPE", "QGAUSS" );
    auto qruleType =
        libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( qruleTypeName );

    const unsigned int dimension = 3;

    feType.reset( new libMesh::FEType( feTypeOrder, feFamily ) );
    fe.reset( ( libMesh::FEBase::build( dimension, ( *feType ) ) ).release() );

    auto qruleOrderName =
        elementRhsDatabase->getWithDefault<std::string>( "QRULE_ORDER", "DEFAULT" );
    libMeshEnums::Order qruleOrder;
    if ( qruleOrderName == "DEFAULT" ) {
        qruleOrder = feType->default_quadrature_order();
    } else {
        qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( qruleOrderName );
    }

    qrule.reset( ( libMesh::QBase::build( qruleType, dimension, qruleOrder ) ).release() );
    fe->attach_quadrature_rule( qrule.get() );

    const auto &JxW  = ( fe->get_JxW() );
    const auto &dphi = ( fe->get_dphi() );
    const auto &phi  = ( fe->get_phi() );

    std::shared_ptr<AMP::Materials::Material> material;
    double youngsModulus = 1.0e10, poissonsRatio = 0.33, thermalExpansionCoefficient = 2.0e-6;
    double default_BURNUP, default_OXYGEN_CONCENTRATION;

    bool useMaterialsLibrary =
        materialModelDatabase->getWithDefault<bool>( "USE_MATERIALS_LIBRARY", false );
    if ( useMaterialsLibrary == true ) {
        AMP_INSIST( ( materialModelDatabase->keyExists( "Material" ) ),
                    "Key ''Material'' is missing!" );
        auto matname = materialModelDatabase->getString( "Material" );
        material     = AMP::Materials::getMaterial( matname );
    }

    if ( useMaterialsLibrary == false ) {
        AMP_INSIST( materialModelDatabase->keyExists( "THERMAL_EXPANSION_COEFFICIENT" ),
                    "Missing key: THERMAL_EXPANSION_COEFFICIENT" );
        AMP_INSIST( materialModelDatabase->keyExists( "Youngs_Modulus" ),
                    "Missing key: Youngs_Modulus" );
        AMP_INSIST( materialModelDatabase->keyExists( "Poissons_Ratio" ),
                    "Missing key: Poissons_Ratio" );

        thermalExpansionCoefficient =
            materialModelDatabase->getScalar<double>( "THERMAL_EXPANSION_COEFFICIENT" );
        youngsModulus = materialModelDatabase->getScalar<double>( "Youngs_Modulus" );
        poissonsRatio = materialModelDatabase->getScalar<double>( "Poissons_Ratio" );
    }

    default_BURNUP = materialModelDatabase->getWithDefault<double>( "Default_Burnup", 0.0 );
    default_OXYGEN_CONCENTRATION =
        materialModelDatabase->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    auto dof_map_0 = rInternal->getDOFManager();
    auto dof_map_1 = currTemperatureVec->getDOFManager();

    auto el     = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end_el = el.end();

    for ( ; el != end_el; ++el ) {
        auto currNodes            = el->getElements( AMP::Mesh::GeomType::Vertex );
        size_t numNodesInCurrElem = currNodes.size();

        std::vector<std::vector<size_t>> type0DofIndices( currNodes.size() );
        std::vector<std::vector<size_t>> type1DofIndices( currNodes.size() );
        for ( size_t j = 0; j < currNodes.size(); ++j ) {
            dof_map_0->getDOFs( currNodes[j]->globalID(), type0DofIndices[j] );
            dof_map_1->getDOFs( currNodes[j]->globalID(), type1DofIndices[j] );
        } // end j

        std::vector<double> elementForceVector( ( 3 * numNodesInCurrElem ), 0.0 );
        std::vector<double> currElementTemperatureVector( numNodesInCurrElem );
        std::vector<double> prevElementTemperatureVector( numNodesInCurrElem );

        for ( size_t r = 0; r < numNodesInCurrElem; ++r ) {
            currElementTemperatureVector[r] =
                currTemperatureVec->getValueByGlobalID( type1DofIndices[r][0] );
            prevElementTemperatureVector[r] =
                prevTemperatureVec->getValueByGlobalID( type1DofIndices[r][0] );
        } // end r

        libMesh::Elem *elem = new libMesh::Hex8;
        for ( size_t j = 0; j < currNodes.size(); ++j ) {
            auto pt             = currNodes[j]->coord();
            elem->set_node( j ) = new libMesh::Node( pt[0], pt[1], pt[2], j );
        } // end j

        fe->reinit( elem );

        for ( unsigned int qp = 0; qp < qrule->n_points(); ++qp ) {
            double Bl_np1[6][24];

            for ( auto &x : Bl_np1 ) {
                for ( size_t j = 0; j < ( 3 * numNodesInCurrElem ); ++j ) {
                    x[j] = 0.0;
                } // end j
            }     // end i

            for ( size_t i = 0; i < numNodesInCurrElem; ++i ) {
                Bl_np1[0][( 3 * i ) + 0] = dphi[i][qp]( 0 );
                Bl_np1[1][( 3 * i ) + 1] = dphi[i][qp]( 1 );
                Bl_np1[2][( 3 * i ) + 2] = dphi[i][qp]( 2 );
                Bl_np1[3][( 3 * i ) + 1] = dphi[i][qp]( 2 );
                Bl_np1[3][( 3 * i ) + 2] = dphi[i][qp]( 1 );
                Bl_np1[4][( 3 * i ) + 0] = dphi[i][qp]( 2 );
                Bl_np1[4][( 3 * i ) + 2] = dphi[i][qp]( 0 );
                Bl_np1[5][( 3 * i ) + 0] = dphi[i][qp]( 1 );
                Bl_np1[5][( 3 * i ) + 1] = dphi[i][qp]( 0 );
            } // end i

            double currTemperatureAtGaussPoint = 0.0;
            double prevTemperatureAtGaussPoint = 0.0;
            for ( unsigned int k = 0; k < numNodesInCurrElem; k++ ) {
                currTemperatureAtGaussPoint += ( currElementTemperatureVector[k] * phi[k][qp] );
                prevTemperatureAtGaussPoint += ( prevElementTemperatureVector[k] * phi[k][qp] );
            } // end k

            if ( useMaterialsLibrary == true ) {

                std::vector<double> tempVec;
                std::vector<double> burnupVec;
                std::vector<double> oxygenVec;

                tempVec.push_back( currTemperatureAtGaussPoint );
                burnupVec.push_back( default_BURNUP );
                oxygenVec.push_back( default_OXYGEN_CONCENTRATION );

                std::vector<double> YM( 1 );
                std::vector<double> PR( 1 );
                std::vector<double> TEC( 1 );

                std::string ymString  = "YoungsModulus";
                std::string prString  = "PoissonRatio";
                std::string tecString = "ThermalExpansion";

                std::map<std::string, std::vector<double> &> args = { { "temperature", tempVec },
                                                                      { "concentration",
                                                                        oxygenVec },
                                                                      { "burnup", burnupVec } };

                material->property( ymString )->evalv( YM, {}, args );
                material->property( prString )->evalv( PR, {}, args );
                material->property( tecString )->evalv( TEC, {}, args );

                youngsModulus               = YM[0];
                poissonsRatio               = PR[0];
                thermalExpansionCoefficient = TEC[0];
            }

            double d_thermalStress[6], d_thermalStrain[6], d_constitutiveMatrix[6][6];
            for ( unsigned int i = 0; i < 6; i++ ) {
                d_thermalStrain[i] = 0.0;
                d_thermalStress[i] = 0.0;
                for ( unsigned int j = 0; j < 6; j++ )
                    d_constitutiveMatrix[i][j] = 0.0;
            }

            double E  = youngsModulus;
            double nu = poissonsRatio;
            double K  = E / ( 3.0 * ( 1.0 - ( 2.0 * nu ) ) );
            double G  = E / ( 2.0 * ( 1.0 + nu ) );

            for ( unsigned int i = 0; i < 3; i++ )
                d_constitutiveMatrix[i][i] += ( 2.0 * G );

            for ( unsigned int i = 3; i < 6; i++ )
                d_constitutiveMatrix[i][i] += G;

            for ( unsigned int i = 0; i < 3; i++ ) {
                for ( unsigned int j = 0; j < 3; j++ ) {
                    d_constitutiveMatrix[i][j] += ( K - ( ( 2.0 * G ) / 3.0 ) );
                }
            }

            for ( unsigned int i = 0; i < 3; i++ ) {
                d_thermalStrain[i] = thermalExpansionCoefficient *
                                     ( currTemperatureAtGaussPoint - prevTemperatureAtGaussPoint );
            }

            for ( unsigned int i = 0; i < 6; i++ ) {
                for ( unsigned int j = 0; j < 6; j++ ) {
                    d_thermalStress[i] += ( d_constitutiveMatrix[i][j] * d_thermalStrain[j] );
                }
            }

            for ( unsigned int j = 0; j < numNodesInCurrElem; j++ ) {
                for ( unsigned int d = 0; d < 3; d++ ) {

                    double tmp = 0;
                    for ( unsigned int m = 0; m < 6; m++ ) {
                        tmp += ( Bl_np1[m][( 3 * j ) + d] * d_thermalStress[m] );
                    }

                    elementForceVector[( 3 * j ) + d] += ( JxW[qp] * tmp );
                } // end d
            }     // end j
        }         // end qp

        for ( unsigned int r = 0; r < numNodesInCurrElem; r++ ) {
            for ( unsigned int d = 0; d < 3; d++ ) {
                rInternal->addValuesByGlobalID(
                    1, &type0DofIndices[r][d], &elementForceVector[( 3 * r ) + d] );
            } // end d
        }     // end r

        for ( size_t j = 0; j < elem->n_nodes(); ++j ) {
            delete ( elem->node_ptr( j ) );
            elem->set_node( j ) = nullptr;
        } // end j
        delete elem;
        elem = nullptr;
    } // end el

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}
