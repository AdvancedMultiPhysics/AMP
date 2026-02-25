#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/boundary/DirichletMatrixCorrectionParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"


using AMP::Utilities::stringf;


namespace AMP::Operator {


static std::shared_ptr<const DirichletVectorCorrectionParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const DirichletVectorCorrectionParameters>( params ) )
        return std::dynamic_pointer_cast<const DirichletVectorCorrectionParameters>( params );
    auto myparams    = std::make_shared<DirichletVectorCorrectionParameters>( params->d_db );
    myparams->d_Mesh = params->d_Mesh;
    auto boundary    = std::dynamic_pointer_cast<const BoundaryOperatorParameters>( params );
    if ( boundary ) {
        myparams->d_variable = boundary->d_volumeOperator->getOutputVariable();
    }
    return myparams;
}


DirichletVectorCorrection::DirichletVectorCorrection(
    std::shared_ptr<const OperatorParameters> params )
    : BoundaryOperator( params )
{

    AMP_ASSERT( params );
    auto myparams                = convert( params );
    d_variable                   = myparams->d_variable;
    d_isAttachedToVolumeOperator = false;
    d_setResidual                = false;
    d_valuesType                 = 0;
    d_scalingFactor              = 0.0;
    reset( params );
}

void DirichletVectorCorrection::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_INSIST( params, "NULL parameters" );
    AMP_INSIST( params->d_db, "NULL database" );

    d_skipParams = params->d_db->getWithDefault<bool>( "skip_params", false );

    if ( !d_skipParams ) {
        d_scalingFactor = params->d_db->getWithDefault<double>( "SCALING_FACTOR", 1.0 );
        d_setResidual   = params->d_db->getWithDefault<bool>( "setResidual", false );
        d_isAttachedToVolumeOperator =
            params->d_db->getWithDefault<bool>( "isAttachedToVolumeOperator", false );
        d_valuesType = params->d_db->getWithDefault<int>( "valuesType", 1 );
        AMP_INSIST( ( d_valuesType == 1 ) || ( d_valuesType == 2 ), "Wrong value." );

        int numIds = params->d_db->getScalar<int>( "number_of_ids" );

        d_boundaryIds.resize( numIds );
        d_dofIds.resize( numIds );

        if ( d_valuesType == 1 )
            d_dirichletValues1.resize( numIds );

        for ( int j = 0; j < numIds; j++ ) {
            d_boundaryIds[j] = params->d_db->getScalar<int>( stringf( "id_%d", j ) );

            int numDofIds = params->d_db->getScalar<int>( stringf( "number_of_dofs_%d", j ) );

            d_dofIds[j].resize( numDofIds );

            if ( d_valuesType == 1 ) {
                d_dirichletValues1[j].resize( numDofIds );
            }
            for ( int i = 0; i < numDofIds; i++ ) {
                d_dofIds[j][i] = params->d_db->getScalar<int>( stringf( "dof_%d_%d", j, i ) );

                if ( d_valuesType == 1 ) {
                    auto key                 = stringf( "value_%d_%d", j, i );
                    d_dirichletValues1[j][i] = params->d_db->getWithDefault<double>( key, 0.0 );
                }
            }
        }
    }
}

// This is an in-place apply
void DirichletVectorCorrection::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                       AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rInternal = mySubsetVector( r, d_variable );

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of rInternal entering DirichletVectorCorrection::apply is : "
                  << rInternal->L2Norm() << std::endl;
    }

    if ( d_setResidual ) {
        this->applyResidual( u, rInternal );
    } else if ( d_isAttachedToVolumeOperator ) {
        this->applyZeroValues( rInternal );
    } else {
        this->applyNonZeroValues( rInternal );
    }

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of rInternal leaving DirichletVectorCorrection::apply is : "
                  << rInternal->L2Norm() << std::endl;
    }
}

void DirichletVectorCorrection::applyZeroValues( AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rInternal = mySubsetVector( r, d_variable );
    auto dof_map   = rInternal->getDOFManager();
    size_t numIds  = d_boundaryIds.size();
    for ( size_t j = 0; j < numIds; j++ ) {
        auto bnd =
            d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryIds[j], 0 );
        auto end_bnd = bnd.end();

        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
            const double val = 0.0;
            for ( auto &elem : d_dofIds[j] ) {
                rInternal->setValuesByGlobalID( 1, &bndGlobalIds[elem], &val );
            }
        }
    }
    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

void DirichletVectorCorrection::applyNonZeroValues( AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rInternal = mySubsetVector( r, d_variable );
    auto dof_map   = rInternal->getDOFManager();
    size_t numIds  = d_boundaryIds.size();
    for ( size_t j = 0; j < numIds; j++ ) {
        auto bnd =
            d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryIds[j], 0 );
        auto end_bnd = bnd.end();

        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );

            for ( size_t i = 0; i < d_dofIds[j].size(); ++i ) {
                double dVal =
                    ( d_valuesType == 1 ) ?
                        d_dirichletValues1[j][i] :
                        d_dirichletValues2->getLocalValueByGlobalID( bndGlobalIds[d_dofIds[j][i]] );
                dVal *= d_scalingFactor;
                rInternal->setValuesByGlobalID( 1, &bndGlobalIds[d_dofIds[j][i]], &dVal );
            }
        }
    }
    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

void DirichletVectorCorrection::applyResidual( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                               AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto uInternal = mySubsetVector( u, d_variable );
    auto dof_map   = uInternal->getDOFManager();
    size_t numIds  = d_boundaryIds.size();
    for ( size_t j = 0; j < numIds; j++ ) {
        auto bnd =
            d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryIds[j], 0 );
        auto end_bnd = bnd.end();

        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
            for ( size_t i = 0; i < d_dofIds[j].size(); i++ ) {
                double uVal = uInternal->getLocalValueByGlobalID( bndGlobalIds[d_dofIds[j][i]] );
                double dVal =
                    ( d_valuesType == 1 ) ?
                        d_dirichletValues1[j][i] :
                        d_dirichletValues2->getLocalValueByGlobalID( bndGlobalIds[d_dofIds[j][i]] );
                dVal = d_scalingFactor * ( uVal - dVal );
                r->setValuesByGlobalID( 1, &bndGlobalIds[d_dofIds[j][i]], &dVal );
            }
        }
    }
    r->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

std::shared_ptr<OperatorParameters>
DirichletVectorCorrection::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr )
{
    auto db = std::make_shared<AMP::Database>( "Dummy" );
    db->putScalar( "name", "DirichletMatrixCorrection" );
    db->putScalar( "skip_params", true );
    db->putScalar( "setResidual", true );
    db->putScalar( "isAttachedToVolumeOperator", true );
    db->putScalar( "number_of_ids", d_boundaryIds.size() );
    for ( int i = 0; i < (int) d_boundaryIds.size(); i++ )
        db->putScalar( stringf( "id_%i", i ), d_boundaryIds[i] );
    for ( int i = 0; i < (int) d_dofIds.size(); i++ ) {
        db->putScalar( stringf( "number_of_dofs_%i", i ), d_dofIds[i].size() );
        for ( int j = 0; j < (int) d_dofIds[i].size(); j++ ) {
            db->putScalar( stringf( "dof_%i_%i", i, j ), d_dofIds[i][j] );
            db->putScalar( stringf( "value_%i_%i", i, j ), 0 );
        }
    }
    for ( int i = 0; i < (int) d_dirichletValues1.size(); i++ ) {
        for ( int j = 0; j < (int) d_dirichletValues1[i].size(); j++ )
            db->putScalar( stringf( "value_%i_%i", i, j ),
                           d_dirichletValues1[i][j],
                           {},
                           AMP::Database::Check::Overwrite );
    }

    auto outParams = std::make_shared<DirichletMatrixCorrectionParameters>( db );
    // outParams->d_variable    = linearDiffusion->getOutputVariable();
    // outParams->d_inputMatrix = linearDiffusion->getMatrix();
    outParams->d_Mesh = d_Mesh;
    return outParams;
}

AMP::LinearAlgebra::Vector::shared_ptr
DirichletVectorCorrection::mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                           std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( d_Mesh ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        auto meshSubsetVec = vec->select( meshSelector );
        auto varSubsetVec  = meshSubsetVec->subsetVectorForVariable( var );
        return varSubsetVec;
    } else {
        return vec->subsetVectorForVariable( var );
    }
}

AMP::LinearAlgebra::Vector::const_shared_ptr
DirichletVectorCorrection::mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                                           std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( d_Mesh ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        auto meshSubsetVec = vec->select( meshSelector );
        auto varSubsetVec  = meshSubsetVec->subsetVectorForVariable( var );
        return varSubsetVec;
    } else {
        return vec->subsetVectorForVariable( var );
    }
}
} // namespace AMP::Operator
