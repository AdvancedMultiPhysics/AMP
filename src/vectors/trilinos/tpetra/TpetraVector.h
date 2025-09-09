#ifndef included_AMP_TpetraVector
#define included_AMP_TpetraVector

#include "AMP/vectors/Vector.h"

DISABLE_WARNINGS
#include "Tpetra_Map_decl.hpp"
#include "Tpetra_Vector_decl.hpp"
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/**  \class TpetraVector
  *  \brief A class that manages an Tpetra::Vector
  *
  *  \see EpetraVector
  *  \see PetscVector
  *  \see SundialsVector
  *
  *  \details An TpetraVector presents an Tpetra::Vector class.  Given an
  *  AMP::LinearAlgebra::Vector, this class can create an Tpetra view without
  *  copying the data.  As such, this class serves three purposes:
  *  -# Provides an Tpetra::Vector for derived classes to use, fill, manage, etc.
  *  -# Provides an interface for accessing this Tpetra::Vector independent of base or derived
  classes
  *  -# Provides a static method for creating an Tpetra::Vector view of an AMP Vector.
  */
class TpetraVector final
{
public:
    /**  \brief Destructor
     */
    ~TpetraVector();

    /**
      *  \brief  Obtain Tpetra::Vector for use in Trilinos routines
      *
      *  \details This function is used to get a Tpetra vector.  The
      *  following idiom should be used since it fails gracefully.  In
      *  this function, a view may be created before the Vec is extracted
      *  \see view()
      *  \returns Tpetra::Vector wrapper for this vector
      *\code
      double DoTpetraMax( Vector::shared_ptr  &in )
      {
        // Create an Tpetra::Vector, if necessary
        auto view = TpetraVector::view( in );
        // Extract the Tpetra::Vector
        Tpetra::Vector &in_vec = view->getTpetra_Vector();
        // Perform an Tpetra::Vector operation
        retrun in_vec.MaxValue ( &abs );
      }
      \endcode
      */
    inline Tpetra::Vector<> &getTpetra_Vector() { return *d_tpetra; }

    /**
      *  \brief  Obtain Tpetra::Vector for use in Trilinos routines
      *
      *  \see view()
      *  \returns Tpetra::Vector wrapper for this vector
      *  \details This function is used to get a Tpetra vector.  The
      *  following idiom should be used since it fails gracefully.  In
      *  this function, a view may be created before the Tpetra::Vector is extracted
      *\code
      double DoTpetraMax( Vector::shared_ptr  &in )
      {
        // Create an Tpetra::Vector, if necessary
        auto view = TpetraVector::view( in );
        // Extract the Tpetra::Vector
        Tpetra::Vector &in_vec = view->getTpetra_Vector();
        // Perform an Tpetra::Vector operation
        retrun in_vec.MaxValue ( &abs );
      }
      \endcode
      */
    inline const Tpetra::Vector<> &getTpetra_Vector() const { return *d_tpetra; }

    /**
     *  \brief  Obtain a view of a vector with an Tpetra::Vector wrapper
     *  \param[in] vec  The vector to get an Tpetra::Vector view of.
     *  \return A Vector::shared_ptr guaranteed to have an Tpetra::Vector
     *   wrapper available through the getTpetra::Vector() interface.
     *  \see getTpetra_Vector()
     *  \details  If the vector has an Tpetra::Vector wrapper already
     *  created, it is returned.  Otherwise, it will try to create an
     *  Tpetra::Vector wrapper around the Vector.  If it fails, an
     *  exception is thrown.
     */
    static std::shared_ptr<TpetraVector> view( Vector::shared_ptr vec );

    /**
     *  \brief  Obtain a view of a vector with an Tpetra::Vector wrapper
     *  \param[in] vec The vector to get an Tpetra::Vector view of.
     *  \return A Vector::shared_ptr guaranteed to have an Tpetra::Vector
     *   wrapper available through the getTpetra_Vector() interface.
     *  \see getTpetra_Vector()
     *  \details  If the vector has an Tpetra::Vector wrapper already
     *  created, it is returned.  Otherwise, it will try to create an
     *  Tpetra::Vector wrapper around the Vector.  If it fails, an
     *  exception is thrown.
     */
    static std::shared_ptr<const TpetraVector> constView( Vector::const_shared_ptr vec );

public:
    inline Tpetra::Vector<> &getNativeVec() { return *d_tpetra; }
    inline const Tpetra::Vector<> &getNativeVec() const { return *d_tpetra; }
    inline std::shared_ptr<Vector> getManagedVec() { return d_AMP; }
    inline std::shared_ptr<const Vector> getManagedVec() const { return d_AMP; }

private:
    TpetraVector() = delete;
    explicit TpetraVector( std::shared_ptr<Vector> );

private:
    std::shared_ptr<Tpetra::Vector<>> d_tpetra;
    std::shared_ptr<Vector> d_AMP;
};


} // namespace AMP::LinearAlgebra


#endif
