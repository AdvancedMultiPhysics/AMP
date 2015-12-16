// This file wraps enable_shared_from_this so that it is more flexible
#ifndef included_AMP_enable_shared_from_this
#define included_AMP_enable_shared_from_this


#include "Utilities.h"
#include "utils/shared_ptr.h"
#include <iostream>

#ifdef USE_BOOST_PTR
#define base_enable_shared_from_this boost::enable_shared_from_this
#else
#define base_enable_shared_from_this std::enable_shared_from_this
#endif


namespace AMP {


/** \brief Enhancement of std::enable_shared_from_this
  * \details  This class enhances/replaces std::enable_shared_from_this.
  *    AMP has use cases that are not properly handled by enable_shared_from_this.
  *    Specifically some of the vectors in AMP are managed by external packages
  *    that do not use shared_ptr.  AMP then requires a shared_ptr for some of its
  *    interfaces.  This causes a problem because if we try to use shared_from_this()
  *    on a pointer that is not managed by the smart pointer, then the function crashes.
  *    An example of this occurs in PetscManagedVector where PETSc manages the vectors,
  *    but we want to enable the underlying data to be an AMP vector.  This class provides
  *    the additional functionallity
  */
template <class T>
class enable_shared_from_this : public base_enable_shared_from_this<T> {
public:
    AMP::shared_ptr<T> shared_from_this()
    {
        AMP::shared_ptr<T> ptr;
        if ( weak_ptr_.use_count() == 0 ) {
            T *tmp = dynamic_cast<T *>( this );
            AMP_ASSERT( tmp != NULL );
            try {
                base_enable_shared_from_this<T> *tmp2 = this;
                ptr                                   = tmp2->shared_from_this();
            }
            catch ( ... ) {
                ptr = AMP::shared_ptr<T>( tmp, []( void * ) {} );
            }
            weak_ptr_ = ptr;
        }
        else {
            ptr = AMP::shared_ptr<T>( weak_ptr_ );
        }
        return ptr;
    }
    AMP::shared_ptr<const T> shared_from_this() const
    {
        AMP::shared_ptr<const T> ptr;
        if ( weak_ptr_.use_count() == 0 ) {
            const T *tmp = dynamic_cast<const T *>( this );
            AMP_ASSERT( tmp != NULL );
            try {
                const base_enable_shared_from_this<T> *tmp2 = this;
                ptr                                         = tmp2->shared_from_this();
            }
            catch ( ... ) {
                // Note: Clang on MAC has issues with the const version of this line, hence the
                // const_cast
                ptr = AMP::shared_ptr<T>( const_cast<T *>( tmp ), []( void * ) {} );
            }
            weak_ptr_ = const_pointer_cast<T>( ptr );
        }
        else {
            ptr = AMP::shared_ptr<const T>( weak_ptr_ );
        }
        return ptr;
    }
    virtual ~enable_shared_from_this<T>() {}
protected:
    mutable AMP::weak_ptr<T> weak_ptr_;
};


} // AMP namespace

#endif
