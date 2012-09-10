#include "vectors/NullVector.h"


namespace AMP {
namespace LinearAlgebra {


Vector::shared_ptr   NullVector::create ( const std::string &name )
{
    return Vector::shared_ptr  ( new NullVector ( Variable::shared_ptr ( new Variable ( name ) ) ) );
}


Vector::shared_ptr  NullVector::create ( const Variable::shared_ptr var )
{
    return Vector::shared_ptr ( new NullVector ( var ) );
}


NullVector::NullVector( Variable::shared_ptr var )
{
    setVariable ( var );
}


NullVector::~NullVector() 
{
}


boost::shared_ptr<ParameterBase> NullVector::getParameters () 
{ 
    return boost::shared_ptr<ParameterBase> (); 
}


Vector::shared_ptr NullVector::cloneVector(const Variable::shared_ptr name) const 
{ 
    return create ( name ); 
}


void NullVector::copyVector( const Vector::const_shared_ptr &rhs )
{
}


void NullVector::swapVectors(Vector &) 
{
}


void NullVector::aliasVector(Vector & ) 
{
}



void NullVector::setToScalar(double ) 
{
}


void NullVector::scale(double , const VectorOperations &) 
{
}


void NullVector::scale(double ) 
{
}


void NullVector::addScalar(const VectorOperations &, double ) 
{
}


void NullVector::add(const VectorOperations &, const VectorOperations &) 
{
}


void NullVector::subtract(const VectorOperations &, const VectorOperations &) 
{
}


void NullVector::multiply( const VectorOperations &, const VectorOperations &) 
{
}


void NullVector::divide( const VectorOperations &, const VectorOperations &) 
{
}


void NullVector::reciprocal(const VectorOperations &) 
{
}


void NullVector::linearSum(double , const VectorOperations &,
          double , const VectorOperations &) 
{
}


void NullVector::axpy(double , const VectorOperations &, const VectorOperations &) 
{
}


void NullVector::axpby(double , double, const VectorOperations &) 
{
}


void NullVector::abs(const VectorOperations &) 
{
}


double NullVector::min(void) const 
{ 
    return 0.0; 
}


double NullVector::max(void) const 
{ 
    return 0.0; 
}


void NullVector::setRandomValues(void) 
{
}


void NullVector::setValuesByLocalID ( int , size_t * , const double * ) 
{ 
    AMP_ERROR( "Can't set values for NullVector" ); 
}


void NullVector::setLocalValuesByGlobalID ( int , size_t * , const double * ) 
{ 
    AMP_ERROR( "Can't set values for NullVector" ); 
}


void NullVector::addValuesByLocalID ( int , size_t * , const double * ) 
{ 
    AMP_ERROR( "Can't set values for NullVector" ); 
}


void NullVector::addLocalValuesByGlobalID ( int , size_t * , const double * ) 
{ 
    AMP_ERROR( "Can't set values for NullVector" ); 
}


void NullVector::getLocalValuesByGlobalID ( int , size_t * , double * ) const 
{ 
    AMP_ERROR( "Can't set values for NullVector" ); 
}



void NullVector::makeConsistent ( ScatterType  ) 
{
}


void NullVector::assemble() 
{
}


double NullVector::L1Norm(void) const 
{ 
    return 0.0; 
}


double NullVector::L2Norm(void) const 
{ 
    return 0.0; 
}


double NullVector::maxNorm(void) const 
{ 
    return 0.0; 
}


double NullVector::dot(const VectorOperations &) const 
{ 
    return 0.0; 
}


void NullVector::putRawData ( double * ) 
{
}


size_t NullVector::getLocalSize() const 
{ 
    return 0; 
}


size_t NullVector::getGlobalSize() const 
{ 
    return 0; 
}


size_t NullVector::getGhostSize() const 
{ 
    return 0; 
}


void NullVector::setCommunicationList ( CommunicationList::shared_ptr  )
{ 
}


size_t NullVector::numberOfDataBlocks () const 
{ 
    return 0; 
}


size_t NullVector::sizeOfDataBlock ( size_t ) const 
{ 
    return 0; 
}


void*NullVector::getRawDataBlockAsVoid ( size_t ) 
{ 
    return 0; 
}


const void *NullVector::getRawDataBlockAsVoid ( size_t ) const 
{ 
    return 0; 
}


void NullVector::addCommunicationListToParameters ( CommunicationList::shared_ptr ) 
{
}

      /// \endcond

}
}
