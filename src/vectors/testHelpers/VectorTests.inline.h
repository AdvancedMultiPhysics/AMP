#ifndef included_test_VectorTests_inline
#define included_test_VectorTests_inline

#include "vectors/testHelpers/VectorTests.h"


namespace AMP {
namespace LinearAlgebra {


template <typename VIEWER>
void VectorTests::DeepCloneOfView( AMP::UnitTest *utils )
{
    AMP::LinearAlgebra::Vector::shared_ptr vector1( d_factory->getVector() );
    if ( !vector1->isA<AMP::LinearAlgebra::MultiVector>() )
        return;
    vector1                                        = VIEWER::view( vector1 );
    AMP::LinearAlgebra::Vector::shared_ptr vector2 = vector1->cloneVector();
    bool pass                                      = true;
    for ( size_t i = 0; i != vector1->numberOfDataBlocks(); i++ ) {
        pass &= ( vector1->getRawDataBlock<double>( i ) != vector2->getRawDataBlock<double>( i ) );
    }
    if ( pass )
        utils->passes( "Deep clone succeeded " + d_factory->name() );
    else
        utils->failure( "Deep clone failed " + d_factory->name() );
}


template <typename ITERATOR>
void VectorTests::both_VectorIteratorTests( AMP::LinearAlgebra::Vector::shared_ptr p, AMP::UnitTest *utils )
{
    typename ITERATOR::vector_type &ref = p->castTo<typename ITERATOR::vector_type>();

    int kk = p->getLocalSize();
    if ( ( p->end() - p->begin() ) == (int) p->getLocalSize() )
        utils->passes( "Subtracting begin from end " );
    else
        utils->failure( "Subtracting begin from end " );

    if ( (int) ( p->begin() - p->end() ) == -(int) p->getLocalSize() )
        utils->passes( "Subtracting end from beginning " );
    else
        utils->failure( "Subtracting end from beginning " );

    ITERATOR cur1, cur2;
    cur1 = cur2  = ref.begin();
    ITERATOR end = ref.end();
    ++cur1;
    ++cur2;
    int i = 0;
    while ( cur2 != end ) {
        if ( i == 10 )
            break;
        ++cur2;
        i++;
    }
    int tt = ( cur2 - cur1 );
    if ( i == tt )
        utils->passes( "Subtracting arbitrary iterators " );
    else
        utils->failure( "Subtracting arbitrary iterators " );

    p->setToScalar( 5.0 );
    i = 0;
    for ( cur1 = ref.begin(); cur1 != end; cur1++ ) {
        if ( ( *cur1 ) != 5.0 )
            break;
        i++;
    }
    if ( i == (int) p->getLocalSize() )
        utils->passes( "Iterating data access " );
    else
        utils->failure( "Iterating data access" );

    cur1 = end;
    i    = 0;
    do {
        --cur1;
        if ( ( *cur1 ) != 5.0 )
            break;
        i++;
    } while ( cur1 != ref.begin() );

    if ( i == kk )
        utils->passes( "Iterating backward data access" );
    else
        utils->failure( "Iterating backward data access" );

    if ( p->getLocalSize() > 7 ) {
        cur1 = ref.begin();
        cur2 = cur1 + 5;
        if ( ( cur2 - cur1 ) == 5 )
            utils->passes( "Adding and subtracting" );
        else
            utils->failure( "Adding and subtracting" );
        i = 0;
        while ( cur2 != end ) {
            i++;
            ++cur2;
        }
        if ( i == ( (int) p->getLocalSize() - 5 ) )
            utils->passes( "Adding and iterating" );
        else
            utils->failure( "Adding and iterating" );

        cur1 += 5;
        i = 0;
        while ( cur1 != end ) {
            i++;
            ++cur1;
        }
        if ( i == ( (int) p->getLocalSize() - 5 ) )
            utils->passes( "Add-equal and iterating" );
        else
            utils->failure( "Add-equal and iterating" );
    }
}


} // namespace LinearAlgebra
} // namespace AMP


#endif