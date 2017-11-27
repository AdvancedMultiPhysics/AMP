#include "vectors/trilinos/epetra/EpetraVectorData.h"


namespace AMP {
namespace LinearAlgebra {


EpetraVectorData::EpetraVectorData( Epetra_DataAccess method,
                                    const Epetra_BlockMap &map,
                                    double *data,
                                    int localStart,
                                    int localSize,
                                    int globalSize )
    : d_epetraVector( method, map, data ),
      d_iLocalStart( localStart ),
      d_iLocalSize( localSize ),
      d_iGlobalSize( globalSize )
{
}


void *EpetraVectorData::getRawDataBlockAsVoid( size_t i )
{
    if ( i > 1 )
        return nullptr;
    double *p;
    d_epetraVector.ExtractView( &p );
    return p;
}

const void *EpetraVectorData::getRawDataBlockAsVoid( size_t i ) const
{
    if ( i > 1 )
        return nullptr;
    double *p;
    d_epetraVector.ExtractView( &p );
    return p;
}

void EpetraVectorData::setValuesByLocalID( int num, size_t *indices, const double *vals )
{
    for ( int i = 0; i != num; i++ )
        d_epetraVector[indices[i]] = vals[i];
}

void EpetraVectorData::setLocalValuesByGlobalID( int num, size_t *indices, const double *vals )
{
    if ( num == 0 )
        return;
    AMP_ASSERT( getGlobalSize() < 0x80000000 );
    std::vector<int> indices2( num, 0 );
    for ( int i = 0; i < num; i++ )
        indices2[i] = (int) indices[i];
    d_epetraVector.ReplaceGlobalValues( num, const_cast<double *>( vals ), &indices2[0] );
}

void EpetraVectorData::addValuesByLocalID( int num, size_t *indices, const double *vals )
{
    if ( num == 0 )
        return;
    for ( int i = 0; i != num; i++ )
        d_epetraVector[indices[i]] += vals[i];
}

void EpetraVectorData::addLocalValuesByGlobalID( int num, size_t *indices, const double *vals )
{
    if ( num == 0 )
        return;
    AMP_ASSERT( getGlobalSize() < 0x80000000 );
    std::vector<int> indices2( num, 0 );
    for ( int i = 0; i < num; i++ )
        indices2[i] = (int) indices[i];
    d_epetraVector.SumIntoGlobalValues( num, const_cast<double *>( vals ), &indices2[0] );
}

void EpetraVectorData::getValuesByLocalID( int, size_t *, double * ) const
{
    AMP_ERROR( "This shouldn't be called" );
}

void EpetraVectorData::getLocalValuesByGlobalID( int, size_t *, double * ) const
{
    AMP_ERROR( "This shouldn't be called" );
}

void EpetraVectorData::putRawData( const double *in )
{
    double *p;
    d_epetraVector.ExtractView( &p );
    size_t N = getLocalSize();
    memcpy( p, in, N * sizeof( double ) );
}

void EpetraVectorData::copyOutRawData( double *out ) const { d_epetraVector.ExtractCopy( out ); }


} // namespace LinearAlgebra
} // namespace AMP