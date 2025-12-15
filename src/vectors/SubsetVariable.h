#ifndef included_AMP_SubsetVariable_H
#define included_AMP_SubsetVariable_H

#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"


namespace AMP::LinearAlgebra {


/** \class SubsetVariable
 * \brief A variable used to create a SubsetVector
 * \see SubsetVector
 * \see VectorIndexer
 */
class SubsetVariable : public Variable
{
public:
    /** \brief Constructor
     * \param[in]  name  The name of the variable
     */
    explicit inline SubsetVariable( const std::string &name ) : Variable( name ) {}

    /** \brief Return a DOFManager that describes the subset
     * \return The DOFManager
     * \param[in]  manager  The DOF manager we want to subset
     */
    virtual std::shared_ptr<AMP::Discretization::DOFManager>
    getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> manager ) const = 0;

    /** \brief Return the comm for the subset
     * \return The comm
     * \param[in]  comm  The comm of the original vector / DOFManager
     */
    virtual AMP::AMP_MPI getComm( const AMP::AMP_MPI &comm ) const = 0;

    /** \brief  This vector is a subset of an AMP Vector
     *  \details  Given an AMP Vector, this will create a view of a subset of the vector.
     *     For instance, if \f$\mathbf{a} = \{ a_0 a_1 a_2 \ldots a_n\}\f$,
     *     and \f$S\f$ is a set of non-negative integers \f$( s_0 s_1 \ldots s_m )\f$
     *     such that \f$0 \le s_i \le n\f$, then the SubsetVector will be the
     *     vector \f$\mathbf{a}_S = \{ a_{s_0} a_{s_1} a_{s_2} \ldots a_{s_m}\} \f$.
     *
     *     Since this is a view, any change to vec2 will be reflected on vec1 and
     *     vice versa.  vec2 is a sparse vector, with a mapping of new index to old.
     *     vec3, on the other hand, is a dense vector without an index.  If a lot
     *     of computation is necessary on the sparse vector, the data can be copied
     *     in and out:
     *
     *     \code
     *        // Subset the vector to make a sparse vector.
     *        vec2 = AMP::SubsetVector( vec1 , subsetVar )
     *
     *        // Copy the sparse vector data to a dense vector
     *        vec3.copyVector( vec2 );
     *
     *        // Perform whatever math
     *        performComputation( vec3 );
     *
     *        // Copy data back to vec2, and, consequently, vec1
     *        vec2.copyVector( vec3 );
     *    \endcode
     */
    virtual Vector::shared_ptr view( Vector::shared_ptr ) const;

    /** \brief  This vector is a subset of an AMP Vector
     *  \details  Given an AMP Vector, this will create a view of a subset of the vector.
     *     For instance, if \f$\mathbf{a} = \{ a_0 a_1 a_2 \ldots a_n\}\f$,
     *     and \f$S\f$ is a set of non-negative integers \f$( s_0 s_1 \ldots s_m )\f$
     *     such that \f$0 \le s_i \le n\f$, then the SubsetVector will be the
     *     vector \f$\mathbf{a}_S = \{ a_{s_0} a_{s_1} a_{s_2} \ldots a_{s_m}\} \f$.
     *
     *     Since this is a view, any change to vec2 will be reflected on vec1 and
     *     vice versa.  vec2 is a sparse vector, with a mapping of new index to old.
     *     vec3, on the other hand, is a dense vector without an index.  If a lot
     *     of computation is necessary on the sparse vector, the data can be copied
     *     in and out:
     *
     *     \code
     *        // Subset the vector to make a sparse vector.
     *        vec2 = AMP::SubsetVector( vec1 , subsetVar )
     *
     *        // Copy the sparse vector data to a dense vector
     *        vec3.copyVector( vec2 );
     *
     *        // Perform whatever math
     *        performComputation( vec3 );
     *
     *        // Copy data back to vec2, and, consequently, vec1
     *        vec2.copyVector( vec3 );
     *    \endcode
     */
    virtual Vector::const_shared_ptr view( Vector::const_shared_ptr ) const;

public: // Functions inherited from Variable
    std::string className() const override { return "SubsetVariable"; }
    std::shared_ptr<VectorSelector> createVectorSelector() const override;
    void writeRestart( int64_t ) const override;
    SubsetVariable( int64_t );
};


} // namespace AMP::LinearAlgebra


#endif
