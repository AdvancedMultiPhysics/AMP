#ifndef included_AMP_ManagedThyraVector
#define included_AMP_ManagedThyraVector

// AMP includes
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"


namespace AMP::LinearAlgebra {


/** \class ManagedThyraVector
 * \brief  Vector capable of returning an Epetra_Vector from a ManagedVector
 * \details  One popular package of numerical algorithms is Trilinos.  Many
 * of the pieces of Trilinos rely on a basic vector called an Epetra vector.
 * This class will provide an Epetra_Vector view on a vector.  This should
 * not be used explicitly.  Rather, the EpetraVector interface provides
 * the getEpetra_Vector interface.  In this case, the class will populate
 * an Epetra_Vector with a view of an array that the ManagedVector has.
 *
 * This class is the class returned by EpetraVector::view() and
 * EpetraVector::constView().
 *
 * \see EpetraVector
 */
class ManagedThyraVector : public Vector, public ThyraVector
{
public:
    /** \brief Create a view of a vector
     * \param[in] alias  Vector to view
     */
    explicit ManagedThyraVector( Vector::shared_ptr alias );

    //! Destructor
    virtual ~ManagedThyraVector();

    // These methods are adequately documented in a base class
    std::string type() const override;

    std::unique_ptr<Vector> rawClone() const override;
    void swapVectors( Vector &other ) override;
    void copyVector( Vector::const_shared_ptr vec ) override;

    std::shared_ptr<Vector> getManagedVec() override { return shared_from_this(); }
    std::shared_ptr<const Vector> getManagedVec() const override { return shared_from_this(); }
};

} // namespace AMP::LinearAlgebra

#endif
