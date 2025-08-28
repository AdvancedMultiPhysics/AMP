#ifndef included_AMP_ManagedSundialsVector
#define included_AMP_ManagedSundialsVector

#include "AMP/vectors/sundials/SundialsVector.h"


extern "C" {
#include "sundials/sundials_nvector.h"
}

namespace AMP::LinearAlgebra {


/**
 * \class ManagedSundialsVector
 * \brief A class that can provide a Sundials N_Vector view of an AMP Vector.
 * \details
 *  This class should not be used explicitly.  It is the return type of
 *  SundialsVector::view() and SundialsVector::constView().
 *
 * \see SundialsVector
 */

class ManagedSundialsVector : public Vector, public SundialsVector
{

public:
    /** \brief Create a view to an AMP vector
     * \param[in] alias  Vector to view
     */
    explicit ManagedSundialsVector( shared_ptr alias );

    /** \brief Destructor
     */
    virtual ~ManagedSundialsVector();

    // These are adequately documented in a base class or there is little need for the documentation
    std::string type() const override;
    std::unique_ptr<Vector> rawClone() const override;
    void swapVectors( Vector &other ) override;

    std::shared_ptr<Vector> getManagedVec() override { return shared_from_this(); }
    std::shared_ptr<const Vector> getManagedVec() const override { return shared_from_this(); }

private:
    explicit ManagedSundialsVector( const ManagedSundialsVector & );
    void operator=( const ManagedSundialsVector & );

    static N_Vector_Ops createNVectorOps();
    static N_Vector clone_AMP( N_Vector w );
    static N_Vector cloneempty_no_impl( N_Vector w );
    static void freeVectorComponents_AMP( N_Vector v );
    // static void space_no_impl(N_Vector v, long int *lrw, long int *liw);
    static realtype *getarraypointer_no_impl( N_Vector v );
    static void setarraypointer_no_impl( realtype *v_data, N_Vector v );
    static void linearSum_AMP( realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z );
    static void setToScalar_AMP( realtype c, N_Vector z );
    static void multiply_AMP( N_Vector x, N_Vector y, N_Vector z );
    static void divide_AMP( N_Vector x, N_Vector y, N_Vector z );
    static void scale_AMP( realtype c, N_Vector x, N_Vector z );
    static void abs_AMP( N_Vector x, N_Vector z );
    static void reciprocal_AMP( N_Vector x, N_Vector z );
    static void addScalar_AMP( N_Vector x, realtype b, N_Vector z );
    static realtype dot_AMP( N_Vector x, N_Vector y );
    static realtype maxNorm_AMP( N_Vector x );
    static realtype WRMSNorm_AMP( N_Vector x, N_Vector w );
    static realtype WRMSNormMask_AMP( N_Vector x, N_Vector w, N_Vector mask );
    static realtype min_AMP( N_Vector x );
    static realtype wl2norm_no_impl( N_Vector x, N_Vector w );
    static realtype L1Norm_AMP( N_Vector x );
    static void compare_no_impl( realtype c, N_Vector x, N_Vector z );
    static booleantype invtest_no_impl( N_Vector x, N_Vector z );
    static booleantype constrmask_no_impl( N_Vector c, N_Vector x, N_Vector m );
    static realtype minquotient_AMP( N_Vector num, N_Vector denom );
};
} // namespace AMP::LinearAlgebra

#endif
