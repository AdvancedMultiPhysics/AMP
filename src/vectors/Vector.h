#ifndef included_AMP_Vector
#define included_AMP_Vector

#include "AMP/utils/Units.h"
#include "AMP/utils/enable_shared_from_this.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/VectorOperations.h"

#include <iosfwd>
#include <memory>
#include <string>


namespace AMP::Discretization {
class DOFManager;
}
namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {


class VectorSelector;


/** \brief Abstraction of a discrete Vector in a linear simulation
 * \details  This class encapsulates many BLAS level 1 operations
 * for use within AMP.  There are a variety of subclasses that
 * implement wrappers around various libraries so that these vectors
 * can be used by methods in those libraries.  Also, there are
 * several subclasses that allow combination of Vectors.  In general,
 * a user will only need to use Vector as the others are different
 * implementation details the developer need not manage.
 *
 * Vectors are created by factories.  For instance, SimpleVector
 * implements a create method that returns a shared pointer to a
 * Vector (Vector::shared_ptr).  The MeshAdpater interface implements
 * a createVector interface that also returns a Vector::shared_ptr.
 * Unless some specialized use of a Vector is needed (see SimpleVector),
 * the Vector::shared_ptr is the only type that should be used when
 * implementing math in AMP.
 *
 * Each Vector is associated with a Variable.  This Variable describes
 * the field this vector represents.
 * The problem may be discretized on a domain and linearized to
 * \f$\mathbf{L}\mathbf{\tilde{u}}=\mathbf{f}\f$.  In this case
 * \f$\mathbf{\tilde{u}}\f$ and \f$\mathbf{f}\f$ are Vectors.
 */
class Vector : public AMP::enable_shared_from_this<Vector>
{

public: // typedefs
    /** \typedef shared_ptr
     * \brief Shorthand for shared pointer to Vector
     */
    typedef std::shared_ptr<Vector> shared_ptr;

    /** \typedef shared_ptr
     * \brief Shorthand for shared pointer to Vector
     */
    typedef std::shared_ptr<const Vector> const_shared_ptr;

    // Deprecated
    typedef VectorDataIterator<double> iterator;
    typedef VectorDataIterator<const double> const_iterator;


public: // Constructor/destructors
    //! Empty Constructor
    Vector();

    /**
     * \brief  Create an empty vector with the given name
     * \param[in] name          Name of the vector
     */
    explicit Vector( const std::string &name );

    /**
     * \brief  Create an vector
     * \param[in] data          Vector data
     * \param[in] ops           Vector operations
     * \param[in] var           Variable
     * \param[in] DOFManager    DOF manager (may be null)
     */
    Vector( std::shared_ptr<VectorData> data,
            std::shared_ptr<VectorOperations> ops,
            std::shared_ptr<Variable> var,
            std::shared_ptr<AMP::Discretization::DOFManager> DOFManager );

    //! No direct copying
    Vector( const Vector & ) = delete;

    //! No direct copying
    void operator=( const Vector & ) = delete;

    //! Destructor
    virtual ~Vector();


public: // Virtual functions
    /** \brief Return the name of the vector
     */
    virtual std::string type() const;

    //@{
    /** \brief Allocate space in the same fashion as <i>this</i>
     * \param[in] name  The variable to associate with the new vector
     * \details  This will allocate new space with identical layout as <i>this</i>.
     * \return  A Vector shared pointer
     * It will have the same number of blocks, each with the same engines and same number of
     * entries.
     */
    virtual std::unique_ptr<Vector> rawClone( const std::shared_ptr<Variable> name ) const;

    /** \brief  Swap the data in this Vector for another
      * \param[in]  other  Vector to swap data with
      * \details Effectively, this is
      * \code
      Vector *a;
      Vector *b;
      std::swap ( a, b );
        \endcode
      * without a and b exchanging pointers.
     */
    virtual void swapVectors( Vector &other );

    /** \brief  Selects a portion of this vector and puts a view into a vector
     * \param[in]  criterion  The method for deciding inclusion in the view
     */
    virtual Vector::shared_ptr selectInto( const VectorSelector &criterion );

    // This is the const version of selectInto.
    virtual Vector::const_shared_ptr selectInto( const VectorSelector &criterion ) const;

    virtual void copyVector( std::shared_ptr<const Vector> x )
    {
        AMP_ASSERT( x );
        d_VectorOps->copy( *( x->getVectorData() ), *d_VectorData );
    }


public: // the next set of functions defines the public math. interface for vectors
    /**
     * \brief  Set vector equal to x
     *      For Vectors, \f$\mathit{this}_i = x_i\f$.
     * \param[in] x         a vector
     */
    void copy( const Vector &x );

    /**
     *\brief Set vector entries (including ghosts) to zero
     *\details This is equivalent (but more efficient) to calling setToScalar ( 0.0 ) followed by a
     *     makeConsistent(SET)
     */
    void zero();

    /**
     * \brief  Set all compenents of a vector to a scalar.
     *      For Vectors, the components of <em>this</em> are set to \f$\alpha\f$.
     * \param[in] alpha     scalar value
     */
    void setToScalar( const Scalar &alpha );

    /**
     * \brief Set data in this vector to random values on [0,1).
     */
    void setRandomValues();

    /**
     * \brief  Set vector equal to scaled input.
     *      For Vectors, \f$\mathit{this}_i = \alpha x_i\f$.
     * \param[in] alpha     a scalar
     * \param[in] x         a vector
     */
    void scale( const Scalar &alpha, const Vector &x );

    /**
     * \brief  Scale a vector.
     *     For Vectors, \f$\mathit{this}_i = \alpha\mathit{this}_i\f$.
     * \param[in] alpha     a scalar
     */
    void scale( const Scalar &alpha );

    /**
     * \brief  Adds two vectors.
     *      For Vectors, \f$\mathit{this}_i = x_i + y_i\f$.
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     */
    void add( const Vector &x, const Vector &y );

    /**
     * \brief Subtracts one vector from another.
     *     For Vectors, \f$\mathit{this}_i = x_i - y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     */
    void subtract( const Vector &x, const Vector &y );

    /**
     * \brief Component-wise multiply one vector with another.
     *    For Vectors, \f$\mathit{this}_i = x_i  y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     */
    void multiply( const Vector &x, const Vector &y );

    /**
     * \brief Component-wise divide one vector by another.
     *    For Vectors, \f$\mathit{this}_i = x_i / y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     */
    void divide( const Vector &x, const Vector &y );

    /**
     * \param x  a vector
     * \brief Set this to the component-wise reciprocal of a vector.  \f$\mathit{this}_i =
     * 1/x_i\f$.
     */
    void reciprocal( const Vector &x );

    /**
     * \brief Set a vector to be a linear combination of two vectors.
     *      \f$\mathit{this}_i = \alpha x_i + \beta y_i\f$.
     * \param[in] alpha     a scalar
     * \param[in] x         a vector
     * \param[in] beta      a scalar
     * \param[in] y         a vector
     */
    void linearSum( const Scalar &alpha, const Vector &x, const Scalar &beta, const Vector &y );

    /**
     * \brief Set this vector to alpha * x + y.  \f$\mathit{this}_i = \alpha x_i + y_i\f$.
     * \param[in] alpha    a scalar
     * \param[in] x        a vector
     * \param[in] y        a vector
     */
    void axpy( const Scalar &alpha, const Vector &x, const Vector &y );

    /**
     * \brief Set this vector alpha * x + this.
     *     \f$\mathit{this}_i = \alpha x_i + \beta \mathit{this}_i \f$
     * \param[in] alpha    a scalar
     * \param[in] beta     a scalar
     * \param[in] x        a vector
     */
    void axpby( const Scalar &alpha, const Scalar &beta, const Vector &x );

    /**
     * \brief Set this to the component-wise absolute value of a vector.
     *     \f$\mathit{this}_i = |x_i|\f$.
     * \param[in] x        a vector
     */
    void abs( const Vector &x );

    /**
     * \brief set vector to \f$x + \alpha \bar{1}\f$.
     * \param[in] x a vector
     * \param[in] alpha a scalar
     * \details  for vectors, \f$\mathit{this}_i = x_i + \alpha\f$.
     */
    void addScalar( const Vector &x, const Scalar &alpha );

    /**
     * \brief modify vector to have max value 'val' specified
     */
    void setMax( const Scalar &val );

    /**
     * \brief modify vector to have min value 'val' specified
     */
    void setMin( const Scalar &val );

    /**
     * \brief Return the minimum value of the vector.  \f$\min_i \mathit{this}_i\f$.
     */
    Scalar min() const;

    /**
     * \brief Return the maximum value of the vector.  \f$\max_i \mathit{this}_i\f$.
     */
    Scalar max() const;

    /**
     * \brief Return the maximum value of the vector.  \f$\max_i \mathit{this}_i\f$.
     */
    Scalar sum() const;

    /**
     * \brief Return the maximum value of the vector.  \f$\max_i \mathit{this}_i\f$.
     */
    Scalar mean() const;

    /**
     * \brief Return discrete @f$ L_1 @f$ -norm of this vector.
     * \details Returns \f[\sum_i |\mathit{this}_i|\f]
     */
    Scalar L1Norm() const;

    /**
     * \brief Return discrete @f$ L_2 @f$ -norm of this vector.
     * \details Returns \f[\sqrt{\sum_i \mathit{this}_i^2}\f]
     */
    Scalar L2Norm() const;

    /**
     * \brief Return the @f$ L_\infty @f$ -norm of this vector.
     * \details Returns \f[\max_i |\mathit{this}_i|\f]
     */
    Scalar maxNorm() const;
    /**
     * \brief Returns the minimum of the quotient of two vectors:
     *    \f[\min_{i,y_i\neq0} x_i/\mathit{this}_i\f]
     * \param[in] x a vector
     * \return \f[\min_{i,y_i\neq0} x_i/\mathit{this}_i\f]
     */
    Scalar minQuotient( const Vector &x ) const;

    /**
     * \brief Return a weighted norm of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_i x^2_iy^2_i}{n}}\f]
     */
    Scalar wrmsNorm( const Vector &x, const Vector &y ) const;

    /**
     * \brief Return a weighted norm of a subset of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \param[in] mask a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_{i,\mathit{mask}_i>0} x^2_iy^2_i}{n}}\f]
     */
    Scalar wrmsNormMask( const Vector &x, const Vector &mask, const Vector &y ) const;

    /**
     * \brief Return the dot product of this vector with the argument vector.
     * \details Returns \f[\sum_i x_i\mathit{this}_i\f]
     * \param[in] x        a vector
     */
    Scalar dot( const Vector &x ) const;

    /**
     * \brief Return the L2 norm of this vector and the dot product with the argument vector
     * \details Returns \f[\sum_i this_i^2 \f] and  \f[\sum_i x_i\mathit{this}_i\f].
     *    Note that this is an unoptimized version purely meant to provide functionality
     * \param[in] x        a vector
     * \param[in] y a vector
     * \return std::pair of L2Norm of vector and dot in that order
     */
    std::pair<Scalar, Scalar> L2NormAndDot( const Vector &x ) const;

    /**
     * \brief Check if two vectors are equal
     * \details Returns true if each vaue is within tol of the corresponding
     *    vaue in the other vector
     * \param[in] x        a vector
     * \param[in] tol      tolerance to use
     */
    bool equals( const Vector &x, const Scalar &tol = 1e-6 ) const;


public: // Clone vectors
    /** \brief Allocate space in the same fashion as <i>this</i>
     * \details  This will allocate new space with identical layout as <i>this</i>.
     *    It will have the same number of blocks, each with the same engines and same number
     *    of entries.  The vector will be associated with the same Variable.
     * \return  A Vector shared pointer
     */
    std::shared_ptr<Vector> clone() const;

    /** \brief Allocate space in the same fashion as <i>this</i>
     * \details  This will allocate new space with identical layout as <i>this</i>.
     *    It will have the same number of blocks, each with the same engines and same
     *    number of entries.  The vector will be associated with a clone of the same Variable
     *    with the given name.
     * \param[in] name  Name to give the variable associated with this vector
     * \return  A Vector shared pointer
     */
    std::shared_ptr<Vector> clone( const std::string &name ) const;

    /** \brief Allocate space in the same fashion as <i>this</i>
     * \details  This will allocate new space with identical layout as <i>this</i>.
     *    It will have the same number of blocks, each with the same engines and same number
     *    of entries.
     * \param[in] name  The variable to associate with the new vector
     * \return  A Vector shared pointer
     */
    std::shared_ptr<Vector> clone( const std::shared_ptr<Variable> name ) const;


public: // Up/down-cast vectors
    /** \brief Copy up/down-casting  <i>this</i>
     * \details  The vector will be associated with the same Variable, and will contain
     *    a copy of the data in x after up/down-casting it.
     * \return  A Vector shared pointer
     */
    void copyCast( std::shared_ptr<const Vector> x );

public: // Get/Set data/variables/operations
    //! Get the units for this Vector
    inline auto getUnits() const { return d_units; }

    //! Get the units for this Vector
    virtual void setUnits( AMP::Units );

    //! Get the DOFManager for this Vector
    inline std::shared_ptr<AMP::Discretization::DOFManager> getDOFManager() const;

    //! Return the pointer to the VectorData
    std::shared_ptr<VectorData> getVectorData() { return d_VectorData; }

    //! Return the pointer to the VectorData
    std::shared_ptr<const VectorData> getVectorData() const { return d_VectorData; }

    //! Return the pointer to the VectorOperation
    std::shared_ptr<VectorOperations> getVectorOperations() { return d_VectorOps; }

    //! Return the pointer to the VectorOperation
    std::shared_ptr<const VectorOperations> getVectorOperations() const { return d_VectorOps; }

    /** \brief Change the variable associated with this vector
     * \param[in] name  The new variable
     */
    void setVariable( const std::shared_ptr<Variable> name );

    /** \brief  Get the variable associated with this vector
     * \return  A shared point to the Variable associated with this Vector
     */
    const std::shared_ptr<Variable> getVariable() const;

    /** \brief  Get the variable associated with this vector
     * \return  A shared point to the Variable associated with this Vector
     */
    std::shared_ptr<Variable> getVariable();

    //! Return integer number of patch data components in vector
    size_t getNumberOfComponents() const;

    //! Return the vector name
    std::string getName() const;

    //! Set the vector name
    void setName( const std::string &name );

    /** \brief Renames
     * \details  Renames the vector and any components matching src to dst.
     * \param[in] src  The original variable name
     * \param[in] dst  The new variable name
     */
    virtual void rename( const std::string &src, const std::string &dst );

public: // Subset/Select
    /** \brief  Selects a portion of this vector and creates a view.
      * \param[in]  criterion  The method for deciding inclusion in the view
      * \param[in]  variable_name  The name of the vector to be created
      * \details To use, we recommend the following pattern
      \code
      // Vector to be "view"ed
      Vector::shared_ptr data;

      // .. set up all the data storage in data

      // Get a view on the data tagged displacement
      auto displacement = data->select( VS_ByVariableName( "displacement" ), "displacement view" );
      \endcode
      */
    shared_ptr select( const VectorSelector &criterion, const std::string &variable_name );

    /** \brief  Selects a portion of this vector and creates a view.
      * \param[in]  criterion  The method for deciding inclusion in the view
      * \param[in]  variable_name  The name of the vector to be created
      * \details To use, we recommend the following pattern
      \code
      // Vector to be "view"ed
      Vector::shared_ptr   data;

      // .. set up all the data storage in data

      // Get a view on the data tagged displacement
      auto displacement = data->select( VS_ByVariableName( "displacement" ), "displacement view" );
      \endcode
      */
    const_shared_ptr select( const VectorSelector &criterion,
                             const std::string &variable_name ) const;

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \see MultiVector
     */
    Vector::shared_ptr subsetVectorForVariable( const std::string &name );

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \details A null pointer will be returned if the vector does not contain
     * a sub-vector associated with the Variable
     * \see MultiVector
     */
    Vector::const_shared_ptr subsetVectorForVariable( const std::string &name ) const;

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] var  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \details A null pointer will be returned if the vector does not contain
     * a sub-vector associated with the Variable
     * \see MultiVector
     */
    Vector::shared_ptr subsetVectorForVariable( std::shared_ptr<const Variable> var );

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] var  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \details A null pointer will be returned if the vector does not contain
     * a sub-vector associated with the Variable
     * \see MultiVector
     */
    Vector::const_shared_ptr subsetVectorForVariable( std::shared_ptr<const Variable> var ) const;

    /** \brief Retrieve ith subvector
     * \details Returns the ith subvector based on teh number of components.
     * \param[in] index  Index to retrieve
     * \return  A Vector shared pointer
     */
    Vector::shared_ptr subsetVectorForComponent( size_t index );

    /** \brief Retrieve ith subvector
     * \details Returns the ith subvector based on teh number of components.
     * \param[in] index  Index to retrieve
     * \return  A Vector shared pointer
     */
    Vector::const_shared_ptr subsetVectorForComponent( size_t index ) const;

    /** \brief  Swap the data in this Vector for another
      * \param[in]  other Vector to swap data with
      * \details A null pointer will be returned if the vector does not contain
      * a sub-vector associated with the Variable. Effectively, this is
      * \code
      Vector *a;
      Vector *b;
      std::swap ( a, b );
        \endcode
      * without a and b exchanging pointers.
     */
    void swapVectors( Vector::shared_ptr other );

    /** \brief  If a particular type of view of this Vector has been created,
     * return it.
     * \tparam VIEW_TYPE The type of view to look for
     * \return A view of this vector
     */
    template<typename VIEW_TYPE>
    std::shared_ptr<VIEW_TYPE> getView() const;

    /** \brief  If a particular type of view of this Vector has been created,
     * return true.
     * \tparam VIEW_TYPE The type of view to look for
     * \return True if a view of this type has been created.  False, otherwise.
     */
    template<typename VIEW_TYPE>
    bool hasView() const;

    /** \brief Add a view of this vector to an internal queue.
     * \param[in] v The view to add
     */
    template<typename VIEW_TYPE>
    void registerView( std::shared_ptr<VIEW_TYPE> v ) const;

    /** \brief Associate the ghost buffer of a Vector with this Vector
     * \param in  The Vector to share a ghost buffer with
     */
    void aliasGhostBuffer( Vector::shared_ptr in );


public: // Iterators/Data
    /**
     * \brief Return an iterator to the beginning of the data
     * \returns A VectorDataIterator
     * \details Since the Vector presents an interface to a contiguous
     *     block of data, it is natural for it to provide a random
     *     access iterator.
     * \warning The non-const version of the iterators will automatically
     *     leave the vector in a non-consistent state.  The user may
     *     be required to call makeConsistent.
     */
    template<class TYPE = double>
    inline VectorDataIterator<TYPE> begin()
    {
        return d_VectorData->begin<TYPE>();
    }

    /// @copydoc VectorData::begin()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> begin() const
    {
        return d_VectorData->begin<const TYPE>();
    }

    /// @copydoc VectorData::begin()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> constBegin() const
    {
        return d_VectorData->constBegin<const TYPE>();
    }

    /**
     * \brief Return an iterator to the end of the data
     * \returns A VectorDataIterator
     * \details Since the Vector presents an interface to a contiguous
     *     block of data, it is natural for it to provide a random
     *     access iterator.
     * \warning The non-const version of the iterators will automatically
     *     leave the vector in a non-consistent state.  The user may
     *     be required to call makeConsistent.
     */
    template<class TYPE = double>
    inline VectorDataIterator<TYPE> end()
    {
        return d_VectorData->end<TYPE>();
    }

    /// @copydoc VectorData::end()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> end() const
    {
        return d_VectorData->end<const TYPE>();
    }

    /// @copydoc VectorData::end()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> constEnd() const
    {
        return d_VectorData->constEnd<const TYPE>();
    }

    /** \brief Obtain a particular contiguous block of data cast to RETURN_TYPE
     * \tparam RETURN_TYPE  The pointer type of the return
     * \param[in] i  Which block
     * \return A contiguous array of type RETURN_TYPE
     */
    template<typename RETURN_TYPE>
    inline RETURN_TYPE *getRawDataBlock( size_t i = 0 )
    {
        return d_VectorData->getRawDataBlock<RETURN_TYPE>( i );
    }

    /** \brief Obtain a particular contiguous block of data cast to RETURN_TYPE
     * \tparam RETURN_TYPE  The pointer type of the return
     * \param[in] i  Which block
     * \return A const contiguous array of type RETURN_TYPE
     */
    template<typename RETURN_TYPE>
    inline const RETURN_TYPE *getRawDataBlock( size_t i = 0 ) const
    {
        return d_VectorData->getRawDataBlock<RETURN_TYPE>( i );
    }


public: // VectorData operations
    inline bool hasComm() const { return d_VectorData->hasComm(); }
    inline AMP_MPI getComm() const { return d_VectorData->getComm(); }
    inline std::string VectorDataName() const { return d_VectorData->VectorDataName(); }
    inline size_t numberOfDataBlocks() const { return d_VectorData->numberOfDataBlocks(); }
    inline size_t sizeOfDataBlock( size_t i = 0 ) const
    {
        return d_VectorData->sizeOfDataBlock( i );
    }
    template<class TYPE>
    inline void putRawData( const TYPE *buf )
    {
        d_VectorData->putRawData( buf );
    }
    template<class TYPE>
    inline void getRawData( TYPE *buf ) const
    {
        d_VectorData->getRawData( buf );
    }
    inline size_t getLocalSize() const { return d_VectorData->getLocalSize(); }
    inline size_t getGlobalSize() const { return d_VectorData->getGlobalSize(); }
    inline size_t getLocalStartID() const { return d_VectorData->getLocalStartID(); }
    template<class TYPE>
    inline void setValuesByLocalID( int num, size_t *indices, const TYPE *vals )
    {
        d_VectorData->setValuesByLocalID( num, indices, vals );
    }
    template<class TYPE>
    inline void setLocalValuesByGlobalID( int num, size_t *indices, const TYPE *vals )
    {
        d_VectorData->setLocalValuesByGlobalID( num, indices, vals );
    }
    template<class TYPE>
    inline void addValuesByLocalID( int num, size_t *indices, const TYPE *vals )
    {
        d_VectorData->addValuesByLocalID( num, indices, vals );
    }
    template<class TYPE>
    inline void addLocalValuesByGlobalID( int num, size_t *indices, const TYPE *vals )
    {
        d_VectorData->addLocalValuesByGlobalID( num, indices, vals );
    }
    template<class TYPE>
    inline void getLocalValuesByGlobalID( int num, size_t *indices, TYPE *vals ) const
    {
        d_VectorData->getLocalValuesByGlobalID( num, indices, vals );
    }
    inline uint64_t getDataID() const { return d_VectorData->getDataID(); }
    inline void *getRawDataBlockAsVoid( size_t i )
    {
        return d_VectorData->getRawDataBlockAsVoid( i );
    }
    inline const void *getRawDataBlockAsVoid( size_t i ) const
    {
        return d_VectorData->getRawDataBlockAsVoid( i );
    }
    inline size_t sizeofDataBlockType( size_t i ) const
    {
        return d_VectorData->sizeofDataBlockType( i );
    }
    template<typename TYPE>
    inline bool isType( size_t block ) const
    {
        constexpr auto type = getTypeID<TYPE>();
        return d_VectorData->isType( type, block );
    }
    inline void swapData( VectorData &rhs ) { d_VectorData->swapData( rhs ); }
    inline std::shared_ptr<VectorData> cloneData() const { return d_VectorData->cloneData(); }
    inline AMP::LinearAlgebra::UpdateState getUpdateStatus() const
    {
        return d_VectorData->getLocalUpdateStatus();
    }
    inline void setUpdateStatus( AMP::LinearAlgebra::UpdateState state )
    {
        d_VectorData->setUpdateStatus( state );
    }
    inline void makeConsistent() { d_VectorData->makeConsistent(); }
    inline void makeConsistent( AMP::LinearAlgebra::ScatterType t )
    {
        d_VectorData->makeConsistent( t );
    }
    //! Get the CommunicationList for this Vector
    inline std::shared_ptr<CommunicationList> getCommunicationList() const
    {
        return d_VectorData->getCommunicationList();
    }
    inline void setCommunicationList( std::shared_ptr<CommunicationList> comm )
    {
        d_VectorData->setCommunicationList( comm );
    }
    inline void dataChanged() { return d_VectorData->dataChanged(); }

    inline size_t getGhostSize() const { return d_VectorData->getGhostSize(); }
    template<typename TYPE>
    inline void setGhostValuesByGlobalID( int num, const size_t *indices, const TYPE *vals )
    {
        d_VectorData->setGhostValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void addGhostValuesByGlobalID( int num, const size_t *indices, const TYPE *vals )
    {
        d_VectorData->addGhostValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void setValuesByGlobalID( int num, const size_t *indices, const TYPE *vals )
    {
        d_VectorData->setValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void addValuesByGlobalID( int num, const size_t *indices, const TYPE *vals )
    {
        d_VectorData->addValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void getValuesByGlobalID( int num, const size_t *indices, TYPE *vals ) const
    {
        d_VectorData->getValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void getGhostValuesByGlobalID( int num, const size_t *indices, TYPE *vals ) const
    {
        d_VectorData->getGhostValuesByGlobalID( num, indices, vals );
    }
    template<typename TYPE>
    inline void getValuesByLocalID( int num, const size_t *indices, TYPE *vals ) const
    {
        d_VectorData->getValuesByLocalID( num, indices, vals );
    }

    inline bool containsGlobalElement( size_t id )
    {
        return d_VectorData->containsGlobalElement( id );
    }

    inline void dumpOwnedData( std::ostream &out, size_t GIDoffset = 0, size_t LIDoffset = 0 ) const
    {
        d_VectorData->dumpOwnedData( out, GIDoffset, LIDoffset );
    }
    inline void dumpGhostedData( std::ostream &out, size_t offset = 0 ) const
    {
        d_VectorData->dumpGhostedData( out, offset );
    }

public: // Get values
    /**
     * \brief Return a value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getValuesByGlobalID to get the value
     */
    template<typename TYPE = double>
    TYPE getValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a local value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getLocalValuesByGlobalID to get the value
     */
    template<typename TYPE = double>
    TYPE getLocalValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a ghost value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getGhostValuesByGlobalID to get the value
     */
    template<typename TYPE = double>
    TYPE getGhostValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a local value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getValuesByGlobalID to get the value
     */
    template<typename TYPE = double>
    TYPE getValueByLocalID( size_t i ) const;


public: // Set values
    /**
     * \brief Set a value in the vector.
     * \param[in] i The global index into the vector
     * \param[in] v The value to set
     * \details This uses setValuesByGlobalID to set the value
     */
    template<typename TYPE>
    void setValueByGlobalID( size_t i, TYPE v );


public:
    //! Get a unique id hash for the vector
    uint64_t getID() const;


public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     * \param manager   Restart manager
     */
    Vector( int64_t fid, AMP::IO::RestartManager *manager );


protected:                                                         // Internal data
    AMP::Units d_units;                                            // Optional units for the data
    std::shared_ptr<Variable> d_Variable;                          // Variable
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManager; // The DOF_Manager
    std::shared_ptr<VectorData> d_VectorData;                      // Pointer to data
    std::shared_ptr<VectorOperations> d_VectorOps;                 // Pointer to a VectorOperations
    std::shared_ptr<std::vector<std::any>> d_Views;                // Views of the vector
};


//! Stream operator
std::ostream &operator<<( std::ostream &out, const Vector::shared_ptr );
//! Stream operator
std::ostream &operator<<( std::ostream &out, const Vector & );


} // namespace AMP::LinearAlgebra

#include "AMP/vectors/Vector.inline.h"

#endif
