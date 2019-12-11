#include "AMP/operators/map/StridedZAxisMap.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/PIO.h"
#include "ProfilerApp.h"


namespace AMP {
namespace Operator {


/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
StridedZAxisMap::StridedZAxisMap( const std::shared_ptr<AMP::Operator::OperatorParameters> &p )
    : ScalarZAxisMap( p )
{
    std::shared_ptr<Map3to1to3Parameters> params =
        std::dynamic_pointer_cast<Map3to1to3Parameters>( p );
    AMP_ASSERT( params );

    d_inpDofs   = params->d_db->getWithDefault( "InputDOFsPerObject", 1 );
    d_inpStride = params->d_db->getWithDefault( "InputStride", 0 );
    d_outDofs   = params->d_db->getWithDefault( "OutputDOFsPerObject", 1 );
    d_outStride = params->d_db->getWithDefault( "OutputStride", 0 );
}


/************************************************************************
 *  De-constructor                                                       *
 ************************************************************************/
StridedZAxisMap::~StridedZAxisMap() = default;

/************************************************************************
 *  Check if the map type is "StridedZAxis"                              *
 ************************************************************************/
bool StridedZAxisMap::validMapType( const std::string &t )
{
    if ( t == "StridedZAxis" )
        return true;
    return false;
}

void StridedZAxisMap::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{

    AMP::LinearAlgebra::Variable::shared_ptr inpVar = getInputVariable();
    AMP::LinearAlgebra::Vector::const_shared_ptr inpPhysics =
        u->constSubsetVectorForVariable( inpVar );
    AMP::pout << "after subset with d_inpStride : " << d_inpStride << " d_inpDofs " << d_inpDofs
              << " with inpVar " << inpVar->getName() << std::endl;
    if ( d_inpDofs != 1 ) {
        AMP::LinearAlgebra::Vector::const_shared_ptr inpStridedPhysics = inpPhysics->constSelect(
            AMP::LinearAlgebra::VS_Stride( d_inpStride, d_inpDofs ), inpVar->getName() );
        AMP_ASSERT( inpStridedPhysics );
        AMP::Operator::AsyncMapOperator::apply( inpStridedPhysics, r );
    } else {
        AMP::Operator::AsyncMapOperator::apply( inpPhysics, r );
    }
}

void StridedZAxisMap::setVector( AMP::LinearAlgebra::Vector::shared_ptr result )
{
    AMP::LinearAlgebra::Variable::shared_ptr outVar   = getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr outPhysics = result->subsetVectorForVariable( outVar );
    AMP::pout << "after subset with d_outStride : " << d_outStride << " d_outDofs " << d_outDofs
              << " with outVar " << outVar->getName() << std::endl;
    if ( d_outDofs != 1 ) {
        AMP::LinearAlgebra::Vector::shared_ptr outStridedPhysics = outPhysics->select(
            AMP::LinearAlgebra::VS_Stride( d_outStride, d_outDofs ), outVar->getName() );
        AMP_ASSERT( outStridedPhysics );
        AMP::Operator::Map3to1to3::setVector( outStridedPhysics );
    } else {
        AMP::Operator::Map3to1to3::setVector( outPhysics );
    }

    AMP::pout << "after stride " << std::endl;
}

} // namespace Operator
} // namespace AMP
