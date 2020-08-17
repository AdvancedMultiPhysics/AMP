#include "AsyncMapOperator.h"

#include "AMP/ampmesh/MultiMesh.h"
#include "AsyncMapOperatorParameters.h"
#include "ProfilerApp.h"

namespace AMP {
namespace Operator {


AsyncMapOperator::AsyncMapOperator( const std::shared_ptr<OperatorParameters> &p )
    : AsynchronousOperator( p )
{
    // Fill some basic info
    auto params = std::dynamic_pointer_cast<AsyncMapOperatorParameters>( p );
    d_MapComm   = params->d_MapComm;
    d_mesh1     = params->d_Mesh1;
    d_mesh2     = params->d_Mesh2;
    AMP_INSIST( !d_MapComm.isNull(), "NULL communicator for map is invalid" );
    AMP_INSIST( d_MapComm.sumReduce<int>( d_mesh1.get() != nullptr ? 1 : 0 ) > 0,
                "Somebody must own mesh 1" );
    AMP_INSIST( d_MapComm.sumReduce<int>( d_mesh2.get() != nullptr ? 1 : 0 ) > 0,
                "Somebody must own mesh 2" );
    // Create a multimesh to use for the operator base class for subsetting
    std::vector<AMP::Mesh::Mesh::shared_ptr> meshes;
    if ( d_mesh1.get() != nullptr )
        meshes.push_back( d_mesh1 );
    if ( d_mesh2.get() != nullptr )
        meshes.push_back( d_mesh2 );
    d_Mesh = std::make_shared<AMP::Mesh::MultiMesh>( "mesh", d_MapComm, meshes );
    // Get the input variable
    bool var  = params->d_db->keyExists( "VariableName" );
    bool var1 = params->d_db->keyExists( "VariableName1" );
    bool var2 = params->d_db->keyExists( "VariableName2" );
    AMP_INSIST( var1 || var2 || var, "VariableName must exist in database" );
    if ( var ) {
        AMP_INSIST( !var1 && !var2,
                    "VariableName is used, VariableName1 and VariableName2cannot be used" );
        std::string variableName = params->d_db->getString( "VariableName" );
        d_inpVariable            = std::make_shared<AMP::LinearAlgebra::Variable>( variableName );
        d_outVariable            = std::make_shared<AMP::LinearAlgebra::Variable>( variableName );
    } else {
        AMP_INSIST( var1 && var2, "Both VariableName1 and VariableName2 must be used" );
        std::string variableName1 = params->d_db->getString( "VariableName1" );
        std::string variableName2 = params->d_db->getString( "VariableName2" );
        d_inpVariable             = std::make_shared<AMP::LinearAlgebra::Variable>( variableName1 );
        d_outVariable             = std::make_shared<AMP::LinearAlgebra::Variable>( variableName2 );
    }
}


AsyncMapOperator::~AsyncMapOperator() = default;


void AsyncMapOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                              AMP::LinearAlgebra::Vector::shared_ptr f )
{
    PROFILE_START( "apply" );
    applyStart( u, f );
    applyFinish( u, f );
    if ( requiresMakeConsistentSet() ) {
        AMP_ASSERT( d_OutputVector.get() != nullptr );
        d_OutputVector->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
    }
    PROFILE_STOP( "apply" );
}


bool AsyncMapOperator::requiresMakeConsistentSet() { return false; }

AMP::Mesh::Mesh::shared_ptr AsyncMapOperator::getMesh( int which )
{
    if ( which == 1 ) {
        return d_mesh1;
    } else if ( which == 2 ) {
        return d_mesh2;
    } else {
        AMP_ERROR( "Wrong option!" );
        return AMP::Mesh::Mesh::shared_ptr();
    }
}
} // namespace Operator
} // namespace AMP