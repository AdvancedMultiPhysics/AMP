#ifndef included_AMP_MultiVector
#define included_AMP_MultiVector


#include "AMP/vectors/Vector.h"


namespace AMP {
namespace LinearAlgebra {


/** \brief A collection of AMP Vectors that appear as one vector
 * \details
 *    Given a set of vectors, they can be collected into a singlevector.  This class
 *    accomplishes this task.
 */
class MultiVector final : public Vector
{
public:
    //! Return the first vector in the MultiVector
    inline auto begin() { return d_vVectors.begin(); }

    //! Return the first vector in the MultiVector
    inline auto begin() const { return d_vVectors.begin(); }

    //! Return one past the last vector in the MultiVector
    inline auto end() { return d_vVectors.end(); }

    //! Return one past the last vector in the MultiVector
    inline auto end() const { return d_vVectors.end(); }


    /** \brief Determine if a Vector is a constituent
     * \param[in]  p  The vector to look for
     * \return True if the pointer p is in the multivector
     */
    bool containsPointer( const Vector::shared_ptr p ) const;

    /** \brief Create a new multivector in parallel
     * \param[in] name  Variable describing the new vector
     * \param[in] comm  Communicator to build the MultiVector on
     * \param[in] vecs  Optional list of vectors in the MultiVector
     */
    static std::shared_ptr<MultiVector>
    create( std::shared_ptr<Variable> name,
            const AMP_MPI &comm,
            const std::vector<Vector::shared_ptr> &vecs = std::vector<Vector::shared_ptr>() );

    /** \brief Create a new multivector in parallel
     * \param[in] name  Name of the new vector
     * \param[in] comm  Communicator to build the MultiVector on
     * \param[in] vecs  Optional list of vectors in the MultiVector
     */
    static std::shared_ptr<MultiVector>
    create( const std::string &name,
            const AMP_MPI &comm,
            const std::vector<Vector::shared_ptr> &vecs = std::vector<Vector::shared_ptr>() );

    /** \brief Create a new multivector in parallel
     * \param[in] name  Variable describing the new vector
     * \param[in] comm  Communicator to build the MultiVector on
     * \param[in] vecs  Optional list of vectors in the MultiVector
     */
    static std::shared_ptr<const MultiVector>
    const_create( std::shared_ptr<Variable> name,
                  const AMP_MPI &comm,
                  const std::vector<Vector::const_shared_ptr> &vecs =
                      std::vector<Vector::const_shared_ptr>() );

    /** \brief Create a new multivector in parallel
     * \param[in] name  Name of the new vector
     * \param[in] comm  Communicator to build the MultiVector on
     * \param[in] vecs  Optional list of vectors in the MultiVector
     */
    static std::shared_ptr<const MultiVector>
    const_create( const std::string &name,
                  const AMP_MPI &comm,
                  const std::vector<Vector::const_shared_ptr> &vecs =
                      std::vector<Vector::const_shared_ptr>() );

    /** \brief Create a multivector view of a vector
     * \param[in] vec  The vector to view
     * \param[in] comm  Communicator to create the MultiVector on
     * \details  If vec is a MultiVector, it is returned.  Otherwise, a MultiVector is created
     * and vec is added to it.  If vec is not a parallel vector(such as a SimpleVector), comm
     * must be specified.
     */
    static std::shared_ptr<MultiVector> view( Vector::shared_ptr vec,
                                              const AMP_MPI &comm = AMP_MPI( AMP_COMM_NULL ) );

    /** \brief Create a multivector view of a vector
     * \param[in] vec  The vector to view
     * \param[in] comm  Communicator to create the MultiVector on
     * \details  If vec is a MultiVector, it is returned.  Otherwise, a MultiVector is created
     * and vec is added to it.  If vec is not a parallel vector(such as a SimpleVector), comm
     * must be specified.
     */
    static std::shared_ptr<const MultiVector>
    constView( Vector::const_shared_ptr vec, const AMP_MPI &comm = AMP_MPI( AMP_COMM_NULL ) );

    /** \brief Encapsulate a vector in a MultiVector
     * \param[in] vec  The vector to view
     * \param[in] comm  Communicator to create the MultiVector on
     * \details  If vec is a MultiVector, it is returned.  Otherwise, a MultiVector is created
     * and vec is added to it.  If vec is not a parallel vector(such as a SimpleVector), comm
     * must be specified.
     */
    static std::shared_ptr<MultiVector>
    encapsulate( Vector::shared_ptr vec, const AMP_MPI &comm = AMP_MPI( AMP_COMM_NULL ) );

    /** \brief Replace a vector in a MultiVector
     * \details  This function will replace a given vector in the multivector with a different
     * vector.  The original and new vectors must share the same DOFManager.
     * This is a purly local operation and does not require communication, but should be called
     * by all processors that own the vectors.
     * \param[in] oldVec  The original vector to replace
     * \param[in] newVec  The new vector to use
     */
    virtual void replaceSubVector( Vector::shared_ptr oldVec, Vector::shared_ptr newVec );

    /** \brief  Add a vector to a MultiVector.
     *   Note: this is a collective operation, vec may be NULL on some processors
     * \param[in]  vec  The Vector to add to the MultiVector
     */
    virtual void addVector( Vector::shared_ptr vec );

    /** \brief  Add vector(s) to a MultiVector.
     *   Note: This is a collective operation, vec may be different sizes on different processors
     * \param[in]  vec  The Vector to add to the MultiVector
     */
    virtual void addVector( std::vector<Vector::shared_ptr> vec );

    /** \brief  Remove a vector from a MultiVector
     * \param[in]  vec  The Vector to remove from the MultiVector
     */
    virtual void eraseVector( Vector::shared_ptr vec );

    /** \brief  Return the i-th Vector in this MultiVector
     * \param[in]  i  Which vector to return
     * \return  The i-th Vector
     */
    virtual Vector::shared_ptr getVector( size_t i );

    /** \brief  Return the i-th Vector in this MultiVector
     * \param[in]  i  Which vector to return
     * \return  The i-th Vector
     */
    virtual Vector::const_shared_ptr getVector( size_t i ) const;

    /** \brief  Obtain the number of Vector objects in this MultiVector
     * \return  The number of Vector objects in this MultiVector
     */
    size_t getNumberOfSubvectors() const;

    //!  Destructor
    virtual ~MultiVector();

    std::string type() const override;

    std::unique_ptr<Vector> rawClone( const std::shared_ptr<Variable> name ) const override;

    void swapVectors( Vector &other ) override;

protected:
    Vector::shared_ptr selectInto( const VectorSelector & ) override;
    Vector::const_shared_ptr selectInto( const VectorSelector &criterion ) const override;

    //! The list of AMP Vectors that comprise this Vector
    std::vector<Vector::shared_ptr> d_vVectors;

    /** \brief A convenience method for extracting vectors from a base class
     * \param[in]  vec  The vector to extract a vector from
     * \param[in]  which  Which vector to get
     * \return     The extracted vector
     */
    const Vector::shared_ptr &getVector( const Vector &vec, size_t which ) const;

    /** \brief A convenience method for extracting vectors from a base class
     * \param[in]  vec  The vector to extract a vector from
     * \param[in]  which  Which vector to get
     * \return     The extracted vector
     */
    Vector::shared_ptr &getVector( Vector &vec, size_t which ) const;

    /** Constructor:  create a MultiVector with a particular variable
     * \param[in]  name  The vector to create the MultiVector from
     */
    explicit MultiVector( const std::string &name, const AMP_MPI &comm );


private:
    // Helper function to add a vector without updating the DOF manager
    void addVectorHelper( Vector::shared_ptr vec );

    // Helper function to reset the vector data
    inline void resetVectorData();

    // Helper function to reset the vector operations
    inline void resetVectorOperations();
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
