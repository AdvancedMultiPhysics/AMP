#ifndef included_AMP_CommSelfVariable_H
#define included_AMP_CommSelfVariable_H

#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/SubsetVariable.h"

namespace AMP::LinearAlgebra {


/** \class MeshVariable
 * \brief An AMP Variable that describes how to subset a DOF for a mesh
 * \see SubsetVector
 */
class CommSelfVariable final : public SubsetVariable
{
public:
    /** \brief Constructor
     * \param[in] name  The name of the new variable
     */
    CommSelfVariable( const std::string &name );

    std::shared_ptr<AMP::Discretization::DOFManager>
        getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> ) const override;

    AMP::AMP_MPI getComm( const AMP::AMP_MPI &comm ) const override;


public: // Functions inherited from Variable
    std::string className() const override { return "CommSelfVariable"; }
    uint64_t getID() const override;
    std::shared_ptr<VectorSelector> createVectorSelector() const override;
    void writeRestart( int64_t ) const override;
    CommSelfVariable( int64_t );
    Vector::shared_ptr view( Vector::shared_ptr ) const override;
    using SubsetVariable::view;

private:
    CommSelfVariable();
};
} // namespace AMP::LinearAlgebra

#endif
