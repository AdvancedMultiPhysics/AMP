#include <algorithm>

#include "SubsetVector.h"
#include "SubsetVariable.h"

namespace AMP {
namespace LinearAlgebra {


/****************************************************************
* Contructors                                                   *
****************************************************************/
Vector::shared_ptr  SubsetVector::view ( Vector::shared_ptr v , Variable::shared_ptr var_in )
{
    boost::shared_ptr<SubsetVariable> var = boost::dynamic_pointer_cast<SubsetVariable>( var_in );
    AMP_ASSERT( var.get() != NULL );
    // Subset the DOFManager and create a new communication list
    boost::shared_ptr<AMP::Discretization::subsetDOFManager> subsetDOF = 
        var->castTo<SubsetVariable>().getSubsetDOF( v->getDOFManager() );
    std::vector<size_t> remote_DOFs = subsetDOF->getRemoteDOFs();
    bool ghosts = v->getComm().maxReduce<char>(remote_DOFs.size()>0)==1;
    AMP::LinearAlgebra::CommunicationList::shared_ptr commList;
    if ( !ghosts ) {
        commList = AMP::LinearAlgebra::CommunicationList::createEmpty( subsetDOF->numLocalDOF(), subsetDOF->getComm() );
    } else {
        // Construct the communication list
        AMP::LinearAlgebra::CommunicationListParameters::shared_ptr params( new AMP::LinearAlgebra::CommunicationListParameters );
        params->d_comm = subsetDOF->getComm();
        params->d_localsize = subsetDOF->numLocalDOF();
        params->d_remote_DOFs = remote_DOFs;
        commList = AMP::LinearAlgebra::CommunicationList::shared_ptr( new AMP::LinearAlgebra::CommunicationList(params) );
    }
    // Create the new subset vector
    boost::shared_ptr<SubsetVector> retVal( new SubsetVector() );
    retVal->setVariable ( var );
    retVal->d_ViewVector = v;
    retVal->d_DOFManager = subsetDOF;
    retVal->setCommunicationList( commList );
    retVal->d_SubsetLocalIDToViewGlobalID = subsetDOF->getLocalParentDOFs();
    // Get the data blocks
    AMP_ERROR("Need to compute the data blocks");
    return retVal;
}


/****************************************************************
* Functions to access the raw data blocks                       *
****************************************************************/
size_t SubsetVector::numberOfDataBlocks () const
{
    return d_dataBlockSize.size();
}
size_t SubsetVector::sizeOfDataBlock ( size_t i ) const
{
    return d_dataBlockSize[i];
}
void *SubsetVector::getRawDataBlockAsVoid ( size_t i )
{
    double *ptr = d_dataBlockPtr[i];
    return (void *) ptr;
  }

const void *SubsetVector::getRawDataBlockAsVoid ( size_t i ) const
{
    double *ptr = d_dataBlockPtr[i];
    return (const void *) ptr;
}



void  SubsetVector::swapVectors ( Vector &rhs )
{
    SubsetVector &s = rhs.castTo<SubsetVector> ();
    d_Space.swap ( s.d_Space );
    std::swap ( d_ViewVector , s.d_ViewVector );
}


void  SubsetVector::addLocalValuesByGlobalID ( int cnt , size_t *ndx ,  const double *vals )
{
    INCREMENT_COUNT("Virtual");
    if ( d_ViewVector )
    {
      size_t  *t = new size_t[ cnt ];
      for ( int i = 0 ; i != cnt ; i++ )
      {
        t[i] = d_SubsetLocalIDToViewGlobalID [ ndx[i] - getCommunicationList()->getStartGID() ];
      }
      d_ViewVector->addLocalValuesByGlobalID ( cnt , t , vals );
      delete [] t;
    }
    else
    {
      for ( int i = 0 ; i != cnt ; i++ )
      {
        d_Space[ndx[i] - getCommunicationList()->getStartGID()] += vals[i];
      }
    }
}

  void  SubsetVector::addValuesByLocalID ( int cnt , size_t *ndx ,  const double *vals )
  {
    INCREMENT_COUNT("Virtual");
    if ( d_ViewVector )
    {
      size_t  *t = new size_t[ cnt ];
      for ( int i = 0 ; i != cnt ; i++ )
      {
        t[i] = d_SubsetLocalIDToViewGlobalID [ ndx[i] ];
      }
      d_ViewVector->addValuesByLocalID ( cnt , t , vals );
      delete [] t;
    }
    else
    {
      for ( int i = 0 ; i != cnt ; i++ )
      {
        d_Space[ndx[i]] += vals[i];
      }
    }
  }

  void  SubsetVector::getLocalValuesByGlobalID ( int cnt , size_t *ndx , double *vals ) const
  {
    INCREMENT_COUNT("Virtual");
    if ( d_ViewVector )
    {
      size_t  *t = new size_t[ cnt ];
      for ( int i = 0 ; i != cnt ; i++ )
      {
        t[i] = d_SubsetLocalIDToViewGlobalID [ ndx[i] - getCommunicationList()->getStartGID() ];
      }
      d_ViewVector->getLocalValuesByGlobalID ( cnt , t , vals );
      delete [] t;
    }
    else
    {
      for ( int i = 0 ; i != cnt ; i++ )
      {
        vals[i] = d_Space[ndx[i] - getCommunicationList()->getStartGID()];
      }
    }
  }

  void  SubsetVector::setLocalValuesByGlobalID ( int cnt , size_t *ndx ,  const double *vals )
  {
    INCREMENT_COUNT("Virtual");
    if ( d_ViewVector )
    {
      size_t  *t = new size_t[ cnt ];
      for ( int i = 0 ; i != cnt ; i++ )
      {
        t[i] = d_SubsetLocalIDToViewGlobalID [ ndx[i] - getCommunicationList()->getStartGID() ];
      }
      d_ViewVector->setLocalValuesByGlobalID ( cnt , t , vals );
      delete [] t;
    }
    else
    {
      for ( int i = 0 ; i != cnt ; i++ )
      {
        d_Space[ndx[i] - getCommunicationList()->getStartGID()] = vals[i];
      }
    }
  }

  void  SubsetVector::setValuesByLocalID ( int cnt , size_t *ndx ,  const double *vals )
  {
    INCREMENT_COUNT("Virtual");
    if ( d_ViewVector )
    {
      size_t  *t = new size_t[ cnt ];
      for ( int i = 0 ; i != cnt ; i++ )
      {
        t[i] = d_SubsetLocalIDToViewGlobalID [ ndx[i] ];
      }
      d_ViewVector->setValuesByLocalID ( cnt , t , vals );
      delete [] t;
    }
    else
    {
      for ( int i = 0 ; i != cnt ; i++ )
      {
        d_Space[ndx[i]] = vals[i];
      }
    }
  }

  void  SubsetVector::putRawData ( double *t )
  {
    if ( d_ViewVector )
    {
      d_ViewVector->setLocalValuesByGlobalID ( getLocalSize() , &(d_SubsetLocalIDToViewGlobalID[0]) , t );
    }
    else
    {
      std::copy ( t , t+d_Space.size() , d_Space.begin() );
    }
  }

  size_t SubsetVector::getLocalSize () const
  {
    return getCommunicationList ()->numLocalRows ();
  }

  size_t SubsetVector::getGlobalSize () const
  {
    return getCommunicationList ()->getTotalSize();
  }


  void  SubsetVector::aliasVector ( Vector & )
  {
    AMP_ERROR( "cannot alias a subset vector.....yet" );
  }

  Vector::shared_ptr  SubsetVector::cloneVector ( Variable::shared_ptr p ) const
  {
    SubsetVector *retVal = new SubsetVector ();
    retVal->d_Space.resize ( getCommunicationList()->numLocalRows() );
    retVal->setVariable ( p );
    if ( getCommunicationList () )
      retVal->setCommunicationList ( getCommunicationList () );
    return Vector::shared_ptr ( retVal );
  }

  std::string  SubsetVector::type () const
  {
    std::string retVal = "Subset Vector";
    if ( d_ViewVector )
    {
      retVal += " ( view of ";
      retVal += d_ViewVector->type();
      retVal += " )";
    }
    return retVal;
  }


}
}


