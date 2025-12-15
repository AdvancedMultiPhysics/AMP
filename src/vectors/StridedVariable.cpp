#include "AMP/vectors/StridedVariable.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/vectors/StridedVariable.h"
#include "AMP/vectors/VectorSelector.h"


namespace AMP::LinearAlgebra {


StridedVariable::StridedVariable( const std::string &name, size_t offset, size_t stride )
    : SubsetVariable( name )
{
    d_offset = offset;
    d_stride = stride;
}


std::shared_ptr<AMP::Discretization::DOFManager>
StridedVariable::getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    std::vector<size_t> dofs;
    dofs.reserve( parentDOF->numLocalDOF() );
    size_t offset2 = d_stride - d_offset % d_stride; // Change the sign of the offset
    for ( size_t DOF = parentDOF->beginDOF(); DOF < parentDOF->endDOF(); DOF++ ) {
        if ( ( DOF + offset2 ) % d_stride == 0 ) {
            dofs.push_back( DOF );
        }
    }
    auto iterator  = parentDOF->getIterator();
    auto subsetDOF = AMP::Discretization::subsetDOFManager::create(
        parentDOF, dofs, iterator, parentDOF->getComm() );
    return subsetDOF;
}

AMP::AMP_MPI StridedVariable::getComm( const AMP::AMP_MPI &comm ) const { return comm; }

std::shared_ptr<VectorSelector> StridedVariable::createVectorSelector() const
{
    return std::make_shared<VS_Stride>( d_offset, d_stride );
}


/****************************************************************
 * Restart                                                       *
 ****************************************************************/
StridedVariable::StridedVariable( int64_t fid ) : SubsetVariable( fid )
{
    AMP_ERROR( "Not finished" );
}
void StridedVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}


} // namespace AMP::LinearAlgebra
