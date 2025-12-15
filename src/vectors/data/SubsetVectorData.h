#ifndef included_AMP_SubsetVectorData
#define included_AMP_SubsetVectorData

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"

#include <vector>

namespace AMP::LinearAlgebra {


//! Parameters used to instantiate a Vector
class SubsetVectorParameters
{
public:
    //! The CommunicationList for a vector
    std::shared_ptr<CommunicationList> d_CommList = nullptr;

    //! The DOF_Manager for a vector
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManager = nullptr;

    Vector::shared_ptr d_ViewVector; // Vector we subsetted for the view
};

/** \class SubsetVector
  * \brief This vector is a subset of an AMP Vector
  * \details
    Given an AMP Vector, this class will create a view of a subset of the
    vector.  For instance, if \f$\mathbf{a} = \{ a_0 a_1 a_2 \ldots a_n\}\f$,
    and \f$S\f$ is a set of non-negative integers \f$( s_0 s_1 \ldots s_m )\f$
    such that \f$0 \le s_i \le n\f$, then the SubsetVector will be the
    vector \f$\mathbf{a}_S = \{ a_{s_0} a_{s_1} a_{s_2} \ldots a_{s_m}\} \f$.

    This class provides a factory method called view:
    \code
      AMP::LinearAlgebra::Vector::shared_ptr  vec1;
      auto vec2 = AMP::SubsetVector( vec1 , subsetVar );
      auto vec3 = vec2->clone();
      vec3->setName( "subset2" );
    \code

    Since this is a view, any change to vec2 will be reflected on vec1 and
    vice versa.  vec2 is a sparse vector, with a mapping of new index to old.
    vec3, on the other hand, is a dense vector without an index.  If a lot
    of computation is necessary on the sparse vector, the data can be copied
    in and out:

    \code
      // Subset the vector to make a sparse vector.
      vec2 = AMP::SubsetVector( vec1 , subsetVar )

      // Copy the sparse vector data to a dense vector
      vec3.copyVector( vec2 );

      // Perform whatever math
      performComputation( vec3 );

      // Copy data back to vec2, and, consequently, vec1
      vec2.copyVector( vec3 );
    \endcode
  */
class SubsetVectorData : public GhostDataHelper<double>
{

public:
    std::string VectorDataName() const override
    {
        return "SubsetVectorData of " + d_ViewVector->type();
    }
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t i ) const override;

    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;
    void putRawData( const void *in, const typeID &id ) override;
    void getRawData( void *out, const typeID &id ) const override;
    typeID getType( size_t ) const override { return d_typeID; }
    uint64_t getDataID() const override { return d_ViewVector->getDataID(); }
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }
    void swapData( VectorData & ) override;
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;
    bool hasContiguousData() const override { return numberOfDataBlocks() > 1 ? false : true; }
    SubsetVectorData() {}
    const AMP_MPI &getComm() const override;
    explicit SubsetVectorData( std::shared_ptr<SubsetVectorParameters> params );

private:
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;

    // Internal data
    Vector::shared_ptr d_ViewVector;                   // Vector we subsetted for the view
    size_t d_parentLocalStartID;                       // Offset for the parent
    std::vector<size_t> d_SubsetLocalIDToViewGlobalID; // The list of global ID in the parent vector
    AMP::typeID d_typeID;
    std::vector<size_t> d_dataBlockSize; // The size of the data blocks
    std::vector<void *> d_dataBlockPtr;  // The pointers to the data blocks
    std::shared_ptr<AMP::Discretization::DOFManager>
        d_DOFManager; // this will contain the subset DOF
};


} // namespace AMP::LinearAlgebra


#endif
