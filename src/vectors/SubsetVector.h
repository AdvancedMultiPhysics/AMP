#ifndef included_AMP_SubsetVector
#define included_AMP_SubsetVector

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/operations/VectorOperationsDefault.h"
#include <vector>

namespace AMP {
namespace LinearAlgebra {

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
      AMP::LinearAlgebra::Vector::shared_pt   vec2 = AMP::SubsetVector( vec1 , subsetVar );
      AMP::LinearAlgebra::Vector::shared_ptr  vec3 = vec2->clone( "subset2" );
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
class SubsetVector : public Vector
{

public:
    static Vector::shared_ptr view( Vector::shared_ptr, Variable::shared_ptr );
    static Vector::const_shared_ptr view( Vector::const_shared_ptr, Variable::shared_ptr );

    std::string type() const override;
    std::string VectorDataName() const override { return "SubsetVector"; }
    using Vector::cloneVector;
    Vector::shared_ptr cloneVector( Variable::shared_ptr ) const override;
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t i ) const override;
    void swapVectors( Vector &rhs ) override;
    void aliasVector( Vector &rhs ) override;
    size_t getLocalSize() const override;
    size_t getGlobalSize() const override;
    void assemble() override {}

    void addValuesByLocalID( int, size_t *, const double * ) override;
    void setValuesByLocalID( int, size_t *, const double * ) override;
    void getValuesByLocalID( int, size_t *, double *vals ) const override;
    void addLocalValuesByGlobalID( int, size_t *, const double * ) override;
    void setLocalValuesByGlobalID( int, size_t *, const double * ) override;
    void getLocalValuesByGlobalID( int, size_t *, double * ) const override;
    void putRawData( const double *in ) override;
    void copyOutRawData( double *out ) const override;

    uint64_t getDataID() const override { return d_ViewVector->getDataID(); }
    bool isTypeId( size_t hash, size_t ) const override
    {
        return hash == typeid( double ).hash_code();
    }
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }
    void swapData( VectorData & ) override { AMP_ERROR( "Not finished" ); }

private:
    SubsetVector() { d_VectorOps = new VectorOperationsDefault<double>(); }
    void computeIDMap();

    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;

    // Internal data
    Vector::shared_ptr d_ViewVector;                   // Vector we subsetted for the view
    std::vector<size_t> d_SubsetLocalIDToViewGlobalID; // The list of global ID in the parent vector
    std::vector<size_t> d_dataBlockSize;               // The size of the data blocks
    std::vector<double *> d_dataBlockPtr;              // The pointers to the data blocks
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
