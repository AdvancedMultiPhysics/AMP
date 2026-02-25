#ifndef included_AMP_NativePetscVectorOperations
#define included_AMP_NativePetscVectorOperations

#include "petsc.h"
#include "petsc/private/vecimpl.h"
#include "petscvec.h"

#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"

namespace AMP::LinearAlgebra {


/*! \struct Vec
    \brief PETSc vector
*/

/** \class NativePetscVector
 * \brief An AMP Vector that uses PETSc for parallel data management, linear algebra,
 * etc.
 * \details  This is an AMP wrapper to PETSc.
 * This class is used when PETSc is chosen as the default linear algebra engine.
 */
class NativePetscVectorOperations : public VectorOperationsDefault<PetscScalar>
{
private:
    void resetArray();
    void resetArray() const;

    //**********************************************************************
    // functions that operate on VectorData
    static Vec getPetscVec( VectorData &x );
    static Vec getPetscVec( const VectorData &x );
    static Vec getConstPetscVec( const VectorData &x );

    static NativePetscVectorData *getNativeVec( VectorData &vx );
    static const NativePetscVectorData *getNativeVec( const VectorData &vx );

public:
    void copy( const VectorData &x, VectorData &z ) override;
    void zero( VectorData &x ) override;
    void setToScalar( const Scalar &alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
    void scale( const Scalar &alpha, const VectorData &x, VectorData &y ) override;
    void scale( const Scalar &alpha, VectorData &x ) override;
    void add( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void subtract( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void multiply( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void divide( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void reciprocal( const VectorData &x, VectorData &y ) override;
    void linearSum( const Scalar &alpha,
                    const VectorData &x,
                    const Scalar &beta,
                    const VectorData &y,
                    VectorData &z ) override;
    void
    axpy( const Scalar &alpha, const VectorData &x, const VectorData &y, VectorData &z ) override;
    void
    axpby( const Scalar &alpha, const Scalar &beta, const VectorData &x, VectorData &y ) override;
    void abs( const VectorData &x, VectorData &z ) override;
    void addScalar( const VectorData &x, const Scalar &alpha_in, VectorData &y ) override;
    Scalar min( const VectorData &x ) const override;
    Scalar max( const VectorData &x ) const override;
    Scalar L1Norm( const VectorData &x ) const override;
    Scalar L2Norm( const VectorData &x ) const override;
    Scalar maxNorm( const VectorData &x ) const override;
    Scalar dot( const VectorData &x, const VectorData &y ) const override;
    Scalar localL1Norm( const VectorData &x ) const override;
    Scalar localL2Norm( const VectorData &x ) const override;
    Scalar localMaxNorm( const VectorData &x ) const override;
    Scalar localDot( const VectorData &x, const VectorData &y ) const override;
    void axpbypcz( const Scalar &alpha,
                   const VectorData &x,
                   const Scalar &beta,
                   const VectorData &y,
                   const Scalar &gamma,
                   VectorData &z );

private:
public: // Pull VectorOperations into the current scope
    using VectorOperationsDefault::abs;
    using VectorOperationsDefault::add;
    using VectorOperationsDefault::axpby;
    using VectorOperationsDefault::axpy;
    using VectorOperationsDefault::divide;
    using VectorOperationsDefault::dot;
    using VectorOperationsDefault::L1Norm;
    using VectorOperationsDefault::L2Norm;
    using VectorOperationsDefault::linearSum;
    using VectorOperationsDefault::max;
    using VectorOperationsDefault::maxNorm;
    using VectorOperationsDefault::min;
    using VectorOperationsDefault::minQuotient;
    using VectorOperationsDefault::multiply;
    using VectorOperationsDefault::reciprocal;
    using VectorOperationsDefault::scale;
    using VectorOperationsDefault::setRandomValues;
    using VectorOperationsDefault::subtract;
    using VectorOperationsDefault::wrmsNorm;
    using VectorOperationsDefault::wrmsNormMask;

private:
    std::shared_ptr<PetscRandom> d_PetscRandom; // PETSc random context
};


} // namespace AMP::LinearAlgebra

#endif
