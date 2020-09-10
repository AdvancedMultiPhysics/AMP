#ifndef included_AMP_Vector
#define included_AMP_Vector


#include <iosfwd>
#include <string>

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/ParameterBase.h"
#include "AMP/utils/RNG.h"
#include "AMP/utils/enable_shared_from_this.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/VectorOperations.h"
#include <memory>


namespace AMP {
namespace LinearAlgebra {


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
    /** \brief Constructor
     * \param[in] parameters  A pointer to a parameters class
     * \see VectorParameters
     */
    explicit Vector( VectorParameters::shared_ptr parameters );

    /** \brief Destructor
     */
    virtual ~Vector();


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
    void zero( void );

    /**
     * \brief  Set all compenents of a vector to a scalar.
     *      For Vectors, the components of <em>this</em> are set to \f$\alpha\f$.
     * \param[in] alpha     scalar double value
     */
    void setToScalar( double alpha );

    /**
     * \brief Set data in this vector to random values on [0,1).
     */
    void setRandomValues( void );

    /**
     * \brief Set data in this vector to random values using
     *      a particular generator
     * \param[in] rng       The generator to use.
     */
    void setRandomValues( RNG::shared_ptr rng );

    /**
     * \brief  Set vector equal to scaled input.
     *      For Vectors, \f$\mathit{this}_i = \alpha x_i\f$.
     * \param[in] alpha     a scalar double
     * \param[in] x         a vector
     */
    void scale( double alpha, const Vector &x );

    /**
     * \brief  Scale a vector.
     *     For Vectors, \f$\mathit{this}_i = \alpha\mathit{this}_i\f$.
     * \param[in] alpha     a scalar double
     */
    void scale( double alpha );

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
    void linearSum( double alpha, const Vector &x, double beta, const Vector &y );

    /**
     * \brief Set this vector to alpha * x + y.  \f$\mathit{this}_i = \alpha x_i + y_i\f$.
     * \param[in] alpha    a scalar
     * \param[in] x        a vector
     * \param[in] y        a vector
     */
    void axpy( double alpha, const Vector &x, const Vector &y );

    /**
     * \brief Set this vector alpha * x + this.
     *     \f$\mathit{this}_i = \alpha x_i + \beta \mathit{this}_i \f$
     * \param[in] alpha    a scalar
     * \param[in] beta     a scalar
     * \param[in] x        a vector
     */
    void axpby( double alpha, double beta, const Vector &x );

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
    void addScalar( const Vector &x, double alpha_in );

    /**
     * \brief Return the minimum value of the vector.  \f$\min_i \mathit{this}_i\f$.
     */
    double min( void ) const;

    /**
     * \brief Return the maximum value of the vector.  \f$\max_i \mathit{this}_i\f$.
     */
    double max( void ) const;

    /**
     * \brief Return discrete @f$ L_1 @f$ -norm of this vector.
     * \details Returns \f[\sum_i |\mathit{this}_i|\f]
     */
    double L1Norm( void ) const;

    /**
     * \brief Return discrete @f$ L_2 @f$ -norm of this vector.
     * \details Returns \f[\sqrt{\sum_i \mathit{this}_i^2}\f]
     */
    double L2Norm( void ) const;

    /**
     * \brief Return the @f$ L_\infty @f$ -norm of this vector.
     * \details Returns \f[\max_i |\mathit{this}_i|\f]
     */
    double maxNorm( void ) const;
    /**
     * \brief Returns the minimum of the quotient of two vectors:
     *    \f[\min_{i,y_i\neq0} x_i/\mathit{this}_i\f]
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\min_{i,y_i\neq0} x_i/y_i\f]
     */
    double minQuotient( const Vector &x ) const;

    /**
     * \brief Return a weighted norm of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_i x^2_iy^2_i}{n}}\f]
     */
    double wrmsNorm( const Vector &x, const Vector &y ) const;

    /**
     * \brief Return a weighted norm of a subset of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \param[in] mask a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_{i,\mathit{mask}_i>0} x^2_iy^2_i}{n}}\f]
     */
    double wrmsNormMask( const Vector &x, const Vector &mask, const Vector &y ) const;


    /**
     * \brief Return the dot product of this vector with the argument vector.
     * \details Returns \f[\sum_i x_i\mathit{this}_i\f]
     * \param[in] x        a vector
     */
    double dot( const Vector &x ) const;

    bool equals( const Vector &a, double tol = 0.000001 ) const;


public: // Virtual functions
    /** \brief Return the name of the vector
     */
    virtual std::string type() const = 0;

    /** \brief Allocate space in the same fashion as <i>this</i>
     * \details  This will allocate new space with identical layout as <i>this</i>.
     * \return  A Vector shared pointer
     * It will have the same number of blocks, each with the same engines and same number of
     * entries.  The vector will
     * be associated with the same Variable.
     */
    Vector::shared_ptr cloneVector() const;

    /** \brief Allocate space in the same fashion as <i>this</i>
     * \param[in] name  Name to give the variable associated with this vector
     * \details  This will allocate new space with identical layout as <i>this</i>.
     * \return  A Vector shared pointer
     * It will have the same number of blocks, each with the same engines and same
     * number of entries.  The vector will be associated with a clone of the same Variable with the
     * given name
     */
    Vector::shared_ptr cloneVector( const std::string &name ) const;

    //! \name Vector memory manipulation
    //! \brief These methods control memory allocation, copying data, aliasing data, and swapping
    //! pointers among Vector
    //! instantiations
    //@{
    /** \brief Allocate space in the same fashion as <i>this</i>
     * \param[in] name  The variable to associate with the new vector
     * \details  This will allocate new space with identical layout as <i>this</i>.
     * \return  A Vector shared pointer
     * It will have the same number of blocks, each with the same engines and same number of
     * entries.
     */
    virtual Vector::shared_ptr cloneVector( const Variable::shared_ptr name ) const = 0;

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
    virtual void swapVectors( Vector &other ) = 0;

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \see MultiVector
     */
    virtual Vector::shared_ptr subsetVectorForVariable( Variable::const_shared_ptr name );

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \see MultiVector
     */
    virtual Vector::const_shared_ptr
    constSubsetVectorForVariable( Variable::const_shared_ptr name ) const;

    /** \brief Return a parameters description of this vector
     * \return Parameters
     */
    virtual std::shared_ptr<ParameterBase> getParameters();

    /** \brief  Selects a portion of this vector and creates a view.
      * \param[in]  criterion  The method for deciding inclusion in the view
      * \param[in]  variable_name  The name of the vector to be created
      * \details To use, we recommend the following pattern
      \code
      // Vector to be "view"ed
      Vector::shared_ptr   data;

      // .. set up all the data storage in data

      // Get a view on the data tagged displacement
      Vector::shared_ptr  displacement = data->select ( VS_ByVariableName ( "displacement" ),
      "displacement view" );
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
      Vector::shared_ptr  displacement = data->select ( VS_ByVariableName ( "displacement" ),
      "displacement view" );
      \endcode
      */
    const_shared_ptr constSelect( const VectorSelector &criterion,
                                  const std::string &variable_name ) const;

    //! Get the DOFManager for this Vector
    AMP::Discretization::DOFManager::shared_ptr getDOFManager() const;


    /** \brief  Selects a portion of this vector and puts a view into a vector
     * \param[in]  criterion  The method for deciding inclusion in the view
     */
    virtual Vector::shared_ptr selectInto( const VectorSelector &criterion );

    // This is the const version of selectInto.
    virtual Vector::const_shared_ptr selectInto( const VectorSelector &criterion ) const;

    virtual void copyVector( std::shared_ptr<const Vector> x )
    {
        d_VectorOps->copy( *( x->getVectorData() ), *d_VectorData );
    }


public: // Non-virtual functions
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
    void setVariable( const Variable::shared_ptr name );

    /** \brief  Get the variable associated with this vector
     * \return  A shared point to the Variable associated with this Vector
     */
    const Variable::shared_ptr getVariable() const;

    /** \brief  Get the variable associated with this vector
     * \return  A shared point to the Variable associated with this Vector
     */
    Variable::shared_ptr getVariable();

    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \see MultiVector
     */
    Vector::shared_ptr subsetVectorForVariable( const std::string &name );


    /** \brief Retrieve a sub-vector associated with a particular Variable
     * \param[in] name  Variable by which to retrieve a subvector
     * \return  A Vector shared pointer
     * \see MultiVector
     */
    Vector::const_shared_ptr constSubsetVectorForVariable( const std::string &name ) const;

    /** \brief  Swap the data in this Vector for another
      * \param[in]  other Vector to swap data with
      * \details Effectively, this is
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
    Vector::shared_ptr getView() const;

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
    void registerView( Vector::shared_ptr v ) const;

    /** \brief Set the default RNG of this vector
     * \param[in] rng  The generator to set
     */
    static void setDefaultRNG( RNG::shared_ptr rng );

    /** \brief Get the current default RNG of this vector
     * \return  The current default RNG.
     * \details  If setDefaultRNG has not been called, this returns
     * an AMP::RNG base class.
     */
    static RNG::shared_ptr getDefaultRNG();

    /** \brief Associate the ghost buffer of a Vector with this Vector
     * \param in  The Vector to share a ghost buffer with
     */
    void aliasGhostBuffer( Vector::shared_ptr in );

public: // Non-virtual functions
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

public:
    /****************************************************************
     *  VectorData operations                                       */
    inline bool hasComm() const { return d_VectorData->hasComm(); }
    inline AMP_MPI getComm() const { return d_VectorData->getComm(); }
    inline std::string VectorDataName() const { return d_VectorData->VectorDataName(); }
    inline size_t numberOfDataBlocks() const { return d_VectorData->numberOfDataBlocks(); }
    inline size_t sizeOfDataBlock( size_t i = 0 ) const
    {
        return d_VectorData->sizeOfDataBlock( i );
    }
    inline void putRawData( const double *buf ) { d_VectorData->putRawData( buf ); }
    inline void copyOutRawData( double *buf ) const { d_VectorData->copyOutRawData( buf ); }
    inline size_t getLocalSize() const { return d_VectorData->getLocalSize(); }
    inline size_t getGlobalSize() const { return d_VectorData->getGlobalSize(); }
    inline size_t getLocalStartID() const { return d_VectorData->getLocalStartID(); }
    inline void setValuesByLocalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->setValuesByLocalID( num, indices, vals );
    }
    inline void setLocalValuesByGlobalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->setLocalValuesByGlobalID( num, indices, vals );
    }
    inline void addValuesByLocalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->addValuesByLocalID( num, indices, vals );
    }
    inline void addLocalValuesByGlobalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->addLocalValuesByGlobalID( num, indices, vals );
    }
    inline void getLocalValuesByGlobalID( int num, size_t *indices, double *vals ) const
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
    inline bool isTypeId( size_t hash, size_t block ) const
    {
        return d_VectorData->isTypeId( hash, block );
    }
    inline void swapData( VectorData &rhs ) { d_VectorData->swapData( rhs ); }
    inline std::shared_ptr<VectorData> cloneData() const { return d_VectorData->cloneData(); }
    inline AMP::LinearAlgebra::VectorData::UpdateState getUpdateStatus() const
    {
        return d_VectorData->getUpdateStatus();
    }
    inline void setUpdateStatus( AMP::LinearAlgebra::VectorData::UpdateState state )
    {
        d_VectorData->setUpdateStatus( state );
    }
    inline void makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType t )
    {
        d_VectorData->makeConsistent( t );
    }
    //! Get the CommunicationList for this Vector
    inline CommunicationList::shared_ptr getCommunicationList() const
    {
        return d_VectorData->getCommunicationList();
    }
    inline void setCommunicationList( CommunicationList::shared_ptr comm )
    {
        d_VectorData->setCommunicationList( comm );
    }
    inline void dataChanged() { return d_VectorData->dataChanged(); }

    // missed on first round
    inline size_t getGlobalMaxID() const { return d_VectorData->getGlobalMaxID(); }
    inline size_t getLocalMaxID() const { return d_VectorData->getLocalMaxID(); }
    inline size_t getGhostSize() const { return d_VectorData->getGhostSize(); }
    inline void setGhostValuesByGlobalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->setGhostValuesByGlobalID( num, indices, vals );
    }
    inline void setValuesByGlobalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->setValuesByGlobalID( num, indices, vals );
    }
    inline void addValuesByGlobalID( int num, size_t *indices, const double *vals )
    {
        d_VectorData->addValuesByGlobalID( num, indices, vals );
    }
    inline void getGhostAddValuesByGlobalID( int num, size_t *indices, double *vals ) const
    {
        d_VectorData->getGhostAddValuesByGlobalID( num, indices, vals );
    }
    inline void getValuesByGlobalID( int num, size_t *indices, double *vals ) const
    {
        d_VectorData->getValuesByGlobalID( num, indices, vals );
    }
    inline void getGhostValuesByGlobalID( int num, size_t *indices, double *vals ) const
    {
        d_VectorData->getGhostValuesByGlobalID( num, indices, vals );
    }
    inline void getValuesByLocalID( int num, size_t *indices, double *vals ) const
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

    /*******************************************************************/

    // These should probably be removed as they add to the interface bloat
public: // Non virtual functions
    /**
     * \brief Return a value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getValuesByGlobalID to get the value
     */
    double getValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a local value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getLocalValuesByGlobalID to get the value
     */
    double getLocalValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a ghost value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getGhostValuesByGlobalID to get the value
     */
    double getGhostValueByGlobalID( size_t i ) const;

    /**
     * \brief Return a local value from the vector.
     * \param[in] i The global index into the vector
     * \return The value stored at the index
     * \details This uses getValuesByGlobalID to get the value
     */
    double getValueByLocalID( size_t i ) const;


protected:
    // Constructor
    Vector();

    void setVectorData( std::shared_ptr<VectorData> data ) { d_VectorData = data; }


private:
    // The following are not implemented
    explicit Vector( const Vector & );
    void operator=( const Vector & );


protected:                                                       // Internal data
    static RNG::shared_ptr d_DefaultRNG;                         // default RNG
    Variable::shared_ptr d_pVariable;                            // Variable
    AMP::Discretization::DOFManager::shared_ptr d_DOFManager;    // The DOF_Manager
    std::shared_ptr<VectorData> d_VectorData;                    // Pointer to data
    std::shared_ptr<VectorOperations> d_VectorOps;               // Pointer to a VectorOperations
    std::shared_ptr<std::vector<std::weak_ptr<Vector>>> d_Views; // Views of the vector
    std::ostream *d_output_stream;                               // output stream for vector data
};


//! Stream operator
std::ostream &operator<<( std::ostream &out, const Vector::shared_ptr );
//! Stream operator
std::ostream &operator<<( std::ostream &out, const Vector & );


} // namespace LinearAlgebra
} // namespace AMP

#include "Vector.inline.h"

#endif
