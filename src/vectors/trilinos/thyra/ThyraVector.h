#ifndef included_AMP_ThyraVector
#define included_AMP_ThyraVector

// AMP includes
#include "AMP/vectors/Vector.h"

// Thyra includes
DISABLE_WARNINGS
#include "Thyra_VectorDefaultBase_decl.hpp"
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/**
 *  \class  ThyraVector
 *  \brief  ThyraVector is a bridge between AMP::LinearAlgebra::Vector and
 *  the Thyra::VectorDefaultBase data structure.
 *
 *  A ThyraVector has a Thyra::VectorDefaultBase data structure.  Given an
 *  AMP::LinearAlgebra::Vector, this class can create a Thyra view without
 *  copying the data.  As such, this class serves three purposes:
 *  -# Provides a Thyra Vector for derived classes to use, fill, manage, etc.
 *  -# Provides an interface for accessing this Thyra Vector independent of derived classes
 *  -# Provides a static method for creating a Thyra view of an AMP Vector.
 */
class ThyraVector
{
public:
    //!  Destructor
    virtual ~ThyraVector();

    /**
     *  \brief  Obtain Thyra Vector for use in Thyra routines
     *
     *  This function is used to get a Thyra vector.  The following idiom
     *  should be used since it fails gracefully.  In this function,
     *  a view may be created before the Vec is extracted
     */
    virtual Teuchos::RCP<Thyra::VectorBase<double>> getVec();

    /**
     *  \brief  Obtain Thyra Vector for use in Thyra routines
     *
     *  This function is used to get a Thyra vector.  The following idiom
     *  should be used since it fails gracefully.  In this function,
     *  a view may be created before the Vec is extracted
     */
    virtual Teuchos::RCP<const Thyra::VectorBase<double>> getVec() const;

    /**
     *  \brief  If needed, create a Thyra wrapper for AmpVector.  Otherwise, return AmpVector.
     *  \details The function attempts to return a view with the least amount of work.
     *     It will never copy data.  If the vector cannot be wrapped it wll return an error.
     *  \param  AmpVector  a shared pointer to a Vector
     */
    static std::shared_ptr<ThyraVector> view( Vector::shared_ptr AmpVector );

    /**
     *  \brief  If needed, create a Thyra wrapper for AmpVector.  Otherwise, return AmpVector.
     *  \details The function attempts to return a view with the least amount of work.
     *     It will never copy data.  If the vector cannot be wrapped it wll return an error.
     *  \param  AmpVector  a shared pointer to a Vector
     */
    static std::shared_ptr<const ThyraVector> constView( Vector::const_shared_ptr AmpVector );


    //! Return an AMP Vector from the Thyra::VectorBase
    static AMP::LinearAlgebra::Vector::shared_ptr view( Thyra::VectorBase<double> *vec );

    //! Return an AMP Vector from the Thyra::VectorBase
    static AMP::LinearAlgebra::Vector::const_shared_ptr
    constView( const Thyra::VectorBase<double> *vec );

public:
    inline auto getNativeVec() { return getVec(); }
    inline auto getNativeVec() const { return getVec(); }
    virtual std::shared_ptr<Vector> getManagedVec()             = 0;
    virtual std::shared_ptr<const Vector> getManagedVec() const = 0;

protected:
    /**
     *  \brief  Thyra Vector holding data in the vector
     *
     *  Whether created with VecCreate (called Native) or
     *  a view of an AMP:Vector (called Managed), this pointer
     *  is what is used when calling the Thyra Vector interface
     */
    Teuchos::RCP<Thyra::VectorBase<double>> d_thyraVec;

    /**
     *  \brief  Swap the underlying PETSc Vec with another
     *  AMP::LinearAlgebra::Vector.
     */
    void swapThyraVec( ThyraVector &rhs ) { std::swap( d_thyraVec, rhs.d_thyraVec ); }

    /**
     *  \brief  Construct a ThyraVector
     *
     *  This can only be called by a derived class or the static function below.  There is
     *  no need to create this vector directly since it is virtual.
     */
    ThyraVector();
};
} // namespace AMP::LinearAlgebra


#endif
