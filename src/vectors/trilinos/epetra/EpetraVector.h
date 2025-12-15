#ifndef included_AMP_EpetraVector
#define included_AMP_EpetraVector

#include "AMP/vectors/Vector.h"

DISABLE_WARNINGS
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/**  \class EpetraVector
  *  \brief A class that manages an Epetra_Vector
  *
  *  \see PetscVector
  *  \see SundialsVector
  *
  *  \details An EpetraVector presents an Epetra_Vector class.  Given an
  *  AMP::LinearAlgebra::Vector, this class can create an Epetra view without
  *  copying the data.  As such, this class serves three purposes:
  *  -# Provides an Epetra_Vector for derived classes to use, fill, manage, etc.
  *  -# Provides an interface for accessing this Epetra_Vector independent of base or derived
  classes
  *  -# Provides a static method for creating an Epetra_Vector view of an AMP Vector.
  */
class EpetraVector final
{
public:
    /**  \brief Destructor
     */
    ~EpetraVector();

    /**
      *  \brief  Obtain Epetra_Vector for use in Trilinos routines
      *
      *  \details This function is used to get a Epetra vector.  The
      *  following idiom should be used since it fails gracefully.  In
      *  this function, a view may be created before the Vec is extracted
      *  \see view()
      *  \returns Epetra_Vector wrapper for this vector
      *\code
      double DoEpetraMax( Vector::shared_ptr  &in )
      {
        // Create an Epetra_Vector, if necessary
        auto view = EpetraVector::view( in );
        // Extract the Epetra_Vector
        Epetra_Vector &in_vec = view->getEpetra_Vector();
        // Perform an Epetra_Vector operation
        retrun in_vec.MaxValue ( &abs );
      }
      \endcode
      */
    inline Epetra_Vector &getEpetra_Vector() { return *d_epetra; }

    /**
      *  \brief  Obtain Epetra_Vector for use in Trilinos routines
      *
      *  \see view()
      *  \returns Epetra_Vector wrapper for this vector
      *  \details This function is used to get a Epetra vector.  The
      *  following idiom should be used since it fails gracefully.  In
      *  this function, a view may be created before the Epetra_Vector is extracted
      *\code
      double DoEpetraMax( Vector::shared_ptr  &in )
      {
        // Create an Epetra_Vector, if necessary
        auto view = EpetraVector::view( in );
        // Extract the Epetra_Vector
        Epetra_Vector &in_vec = view->getEpetra_Vector();
        // Perform an Epetra_Vector operation
        retrun in_vec.MaxValue ( &abs );
      }
      \endcode
      */
    inline const Epetra_Vector &getEpetra_Vector() const { return *d_epetra; }

    /**
     *  \brief  Obtain a view of a vector with an Epetra_Vector wrapper
     *  \param[in] vec  The vector to get an Epetra_Vector view of.
     *  \return A Vector::shared_ptr guaranteed to have an Epetra_Vector
     *   wrapper available through the getEpetra_Vector() interface.
     *  \see getEpetra_Vector()
     *  \details  If the vector has an Epetra_Vector wrapper already
     *  created, it is returned.  Otherwise, it will try to create an
     *  Epetra_Vector wrapper around the Vector.  If it fails, an
     *  exception is thrown.
     */
    static std::shared_ptr<EpetraVector> view( Vector::shared_ptr vec );

    /**
     *  \brief  Obtain a view of a vector with an Epetra_Vector wrapper
     *  \param[in] vec The vector to get an Epetra_Vector view of.
     *  \return A Vector::shared_ptr guaranteed to have an Epetra_Vector
     *   wrapper available through the getEpetra_Vector() interface.
     *  \see getEpetra_Vector()
     *  \details  If the vector has an Epetra_Vector wrapper already
     *  created, it is returned.  Otherwise, it will try to create an
     *  Epetra_Vector wrapper around the Vector.  If it fails, an
     *  exception is thrown.
     */
    static std::shared_ptr<const EpetraVector> constView( Vector::const_shared_ptr vec );

public:
    inline Epetra_Vector &getNativeVec() { return *d_epetra; }
    inline const Epetra_Vector &getNativeVec() const { return *d_epetra; }
    inline std::shared_ptr<Vector> getManagedVec() { return d_AMP; }
    inline std::shared_ptr<const Vector> getManagedVec() const { return d_AMP; }

private:
    EpetraVector() = delete;
    explicit EpetraVector( std::shared_ptr<Vector> );

private:
    std::shared_ptr<Epetra_Vector> d_epetra;
    std::shared_ptr<Vector> d_AMP;
};


} // namespace AMP::LinearAlgebra


#endif
