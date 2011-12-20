#include "utils/Utilities.h"
#include "ManagedVector.h"
#include <stdexcept>



namespace AMP {
namespace LinearAlgebra {


  Vector::shared_ptr  ManagedVector::subsetVectorForVariable ( const Variable::shared_ptr &name )
  {
    Vector::shared_ptr  retVal;
    if ( !d_vBuffer )
    {
      retVal = d_Engine->castTo<Vector>().subsetVectorForVariable ( name );
    }
    if ( !retVal )
    {
      retVal = Vector::subsetVectorForVariable ( name );
    }
    return retVal;
  }

  bool ManagedVector::isAnAliasOf ( Vector &rhs )
  {
    bool retVal = false;
    if ( rhs.isA<ManagedVector>() )
    {
      ManagedVector &other = rhs.castTo<ManagedVector> ();
      if ( d_vBuffer && ( other.d_vBuffer == d_vBuffer ) )
      {
        retVal = true;
      }
    }
    return retVal;
  }

  ManagedVector::ManagedVector ( shared_ptr  alias )
    : Vector ( boost::dynamic_pointer_cast<VectorParameters> ( alias->castTo<ManagedVector>().getParameters() ) ) ,
      d_vBuffer ( alias->castTo<ManagedVector>().d_vBuffer ) ,
      d_Engine ( alias->castTo<ManagedVector>().d_Engine->cloneEngine ( d_vBuffer ) )
  {
    d_Engine = alias->castTo<ManagedVector>().d_Engine;
    setVariable ( alias->getVariable() );
    d_pParameters = alias->castTo<ManagedVector>().d_pParameters;
    aliasGhostBuffer ( alias );
  }


  void ManagedVector::copyVector ( const Vector::const_shared_ptr &other )
  {
    if ( other->getLocalSize() != getLocalSize() )
    {  // Another error condition
      AMP_ERROR( "Destination vector and source vector not the same size" );
    }
    fireDataChange();
    VectorDataIterator  cur1 = begin();
    VectorDataIterator  end1 = end();
    ConstVectorDataIterator  cur2 = other->begin();
    while ( cur1 != end1 )
    {
      *cur1 = *cur2;
      ++cur1;
      ++cur2;
    }
    copyGhostValues ( other );
  }


  void ManagedVector::swapVectors ( Vector &other )
  {
    ManagedVector &in = other.castTo<ManagedVector> ();
    d_vBuffer.swap ( in.d_vBuffer );
    std::swap ( d_pParameters , in.d_pParameters );
    d_Engine->swapEngines ( in.d_Engine );
  }


  void ManagedVector::aliasVector ( Vector &other )
  {
    ManagedVector &in = other.castTo<ManagedVector> ();
    d_pParameters = in.d_pParameters;
    d_vBuffer = in.d_vBuffer;
  }

  void ManagedVector::getValuesByGlobalID ( int numVals , size_t *ndx , double *vals ) const
  {
    INCREMENT_COUNT("Virtual");
    Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec.get() == NULL ) {
        Vector::getValuesByGlobalID ( numVals , ndx , vals );
    } else {
        vec->getValuesByGlobalID ( numVals , ndx , vals );
    }
  }

  void ManagedVector::getLocalValuesByGlobalID ( int numVals , size_t *ndx , double *vals ) const
  {
    INCREMENT_COUNT("Virtual");
    if ( d_vBuffer ) {
        for ( int i = 0 ; i != numVals ; i++ )
           vals[i] = (*d_vBuffer)[ndx[i] - d_CommList->getStartGID() ];
    } else {
        Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
        vec->getLocalValuesByGlobalID ( numVals , ndx , vals );
    }
  }

  void ManagedVector::getGhostValuesByGlobalID ( int numVals , size_t *ndx , double *vals ) const
  {
    INCREMENT_COUNT("Virtual");
    Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec.get() == NULL ) {
        Vector::getGhostValuesByGlobalID ( numVals , ndx , vals );
    } else {
        vec->getGhostValuesByGlobalID ( numVals , ndx , vals );
    }
  }

  void ManagedVector::setValuesByLocalID(int i, size_t *id , const double *val)
  {
    INCREMENT_COUNT("Virtual");
    AMP_ASSERT ( *d_UpdateState != ADDING );
    *d_UpdateState = SETTING;
    d_Engine->setValuesByLocalID ( i , id , val );
    fireDataChange();
  }

  void ManagedVector::setLocalValuesByGlobalID(int numVals , size_t *ndx , const double *vals )
  {
    INCREMENT_COUNT("Virtual");
    AMP_ASSERT ( *d_UpdateState != ADDING );
    *d_UpdateState = SETTING;
    d_Engine->setLocalValuesByGlobalID ( numVals, ndx, vals );
    fireDataChange();
  }

  void ManagedVector::setGhostValuesByGlobalID ( int numVals , size_t *ndx , const double *vals )
  {
    INCREMENT_COUNT("Virtual");
    Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec.get() == NULL ) {
        Vector::setGhostValuesByGlobalID ( numVals , ndx , vals );
    } else {
        vec->setGhostValuesByGlobalID ( numVals , ndx , vals );
    }
  }

  void ManagedVector::setValuesByGlobalID ( int numVals , size_t *ndx , const double *vals )
  {
    Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec.get() != NULL ) {
        INCREMENT_COUNT("Virtual");
        AMP_ASSERT ( *d_UpdateState != ADDING );
        *d_UpdateState = SETTING;
        Vector::shared_ptr vec = boost::dynamic_pointer_cast<Vector>( d_Engine );
        vec->setValuesByGlobalID ( numVals, ndx, vals );
        fireDataChange();
    } else {
        std::vector<size_t> local_ndx;  local_ndx.reserve(numVals);
        std::vector<double> local_val;  local_val.reserve(numVals);
        std::vector<size_t> ghost_ndx;  ghost_ndx.reserve(numVals);
        std::vector<double> ghost_val;  ghost_val.reserve(numVals);
        for (int i=0; i<numVals; i++) {
            if ( ( ndx[i] < getLocalStartID() ) || ( ndx[i] >= (getLocalStartID() + getLocalMaxID()) ) ) {
                ghost_ndx.push_back(ndx[i]);
                ghost_val.push_back(vals[i]);
            } else {
                local_ndx.push_back(ndx[i]);
                local_val.push_back(vals[i]);
            }
        }
        if ( ghost_ndx.size() > 0 )
            setGhostValuesByGlobalID( ghost_ndx.size(), &ghost_ndx[0], &ghost_val[0] );
        if ( local_ndx.size() > 0 )
            setLocalValuesByGlobalID( local_ndx.size(), &local_ndx[0], &local_val[0] );
    }
  }

  void ManagedVector::addValuesByLocalID(int i, size_t *id , const double *val)
  {
    INCREMENT_COUNT("Virtual");
    AMP_ASSERT ( *d_UpdateState != SETTING );
    *d_UpdateState = ADDING;
    d_Engine->addValuesByLocalID ( i , id , val );
    fireDataChange();
  }

  void ManagedVector::addLocalValuesByGlobalID(int i, size_t *id , const double *val)
  {
    INCREMENT_COUNT("Virtual");
    AMP_ASSERT ( *d_UpdateState != SETTING );
    *d_UpdateState = ADDING;
    d_Engine->addLocalValuesByGlobalID ( i , id , val );
    fireDataChange();
  }


}
}
