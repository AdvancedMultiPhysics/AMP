#ifndef included_AMP_test_VectorTests_inline
#define included_AMP_test_VectorTests_inline

#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/data/VectorDataIterator.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


namespace AMP::LinearAlgebra {


template<typename VIEWER>
void VectorTests::DeepCloneOfView( AMP::UnitTest *utils )
{
    auto vector1 = d_factory->getVector();
    if ( !std::dynamic_pointer_cast<MultiVector>( vector1 ) )
        return;
    vector1      = VIEWER::view( vector1 )->getManagedVec();
    auto vector2 = vector1->clone();
    bool pass    = true;
    for ( size_t i = 0; i != vector1->numberOfDataBlocks(); i++ ) {
        pass &= ( vector1->getRawDataBlock<double>( i ) != vector2->getRawDataBlock<double>( i ) );
    }
    if ( pass )
        utils->passes( "Deep clone succeeded " + d_factory->name() );
    else
        utils->failure( "Deep clone failed " + d_factory->name() );
}


} // namespace AMP::LinearAlgebra


#endif
