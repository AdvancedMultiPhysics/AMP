#include "AMP/operators/map/StridedZAxisMap.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/VectorSelector.h"
#include "ProfilerApp.h"


namespace AMP::Operator {


/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
StridedZAxisMap::StridedZAxisMap( std::shared_ptr<const AMP::Operator::OperatorParameters> p )
    : ScalarZAxisMap( p )
{
    auto params = std::dynamic_pointer_cast<const Map3to1to3Parameters>( p );
    AMP_ASSERT( params );

    d_inpDofs   = params->d_db->getWithDefault<int>( "InputDOFsPerObject", 1 );
    d_inpStride = params->d_db->getWithDefault<int>( "InputStride", 0 );
    d_outDofs   = params->d_db->getWithDefault<int>( "OutputDOFsPerObject", 1 );
    d_outStride = params->d_db->getWithDefault<int>( "OutputStride", 0 );
}


/************************************************************************
 *  Destructor                                                           *
 ************************************************************************/
StridedZAxisMap::~StridedZAxisMap() = default;


/************************************************************************
 *  Check if the map type is "StridedZAxis"                              *
 ************************************************************************/
bool StridedZAxisMap::validMapType( const std::string &t ) { return t == "StridedZAxis"; }

void StridedZAxisMap::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{

    auto inpVar = getInputVariable();
    auto inpVec = u->subsetVectorForVariable( inpVar );
    if ( d_inpDofs != 1 ) {
        auto strided = inpVec->select( AMP::LinearAlgebra::VS_Stride( d_inpStride, d_inpDofs ) );
        AMP_ASSERT( strided );
        AMP::Operator::AsyncMapOperator::apply( strided, r );
    } else {
        AMP::Operator::AsyncMapOperator::apply( inpVec, r );
    }
}

void StridedZAxisMap::setVector( AMP::LinearAlgebra::Vector::shared_ptr result )
{
    auto outVar = getOutputVariable();
    auto outVec = result->subsetVectorForVariable( outVar );
    if ( d_outDofs != 1 ) {
        auto strided = outVec->select( AMP::LinearAlgebra::VS_Stride( d_outStride, d_outDofs ) );
        AMP_ASSERT( strided );
        AMP::Operator::Map3to1to3::setVector( strided );
    } else {
        AMP::Operator::Map3to1to3::setVector( outVec );
    }
}


} // namespace AMP::Operator
