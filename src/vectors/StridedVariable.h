#ifndef included_AMP_StridedVariable_H
#define included_AMP_StridedVariable_H

#include "AMP/vectors/SubsetVariable.h"

namespace AMP::LinearAlgebra {

/** \class StridedVariable
 * \brief An AMP Variable that describes how to stride a vector to create
 * a SubsetVector
 * \see SubsetVector
 * \see StridedIndexer
 */
class StridedVariable : public SubsetVariable
{
public:
    /** \brief Constructor
     * \param[in] name  The name of the new variable
     * \param[in] offset  The offset to start striding a vector
     * \param[in] stride  The stride of the vector
     */
    StridedVariable( const std::string &name, size_t offset, size_t stride );

    std::shared_ptr<AMP::Discretization::DOFManager>
        getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> ) const override;

    AMP::AMP_MPI getComm( const AMP::AMP_MPI &comm ) const override;

    std::shared_ptr<VectorSelector> createVectorSelector() const override;

    std::string className() const override { return "StridedVariable"; }

    void writeRestart( int64_t ) const override;
    StridedVariable( int64_t );

private:
    StridedVariable();
    size_t d_offset = 0;
    size_t d_stride = 1;
};
} // namespace AMP::LinearAlgebra

#endif
