#ifndef included_AMP_TrilinosLinearOP
#define included_AMP_TrilinosLinearOP

// Trilinos includes
#include "Thyra_LinearOpBase_def.hpp"


namespace AMP {
namespace Solver {


/**
  * The TrilinosLinearOP is a wrapper for a Thyra LinearOpBase to 
  * wrap AMP LinearOperators for use with Trilinos NOX solvers.
  */
class TrilinosLinearOP: public Thyra::LinearOpBase<double>
{
public:
    
    //! Empty constructor
    TrilinosLinearOP();

    //! Destructor
    virtual ~TrilinosLinearOP();

    // Functions inherited from Thyra::LinearOpBase
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > range() const;
    virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > domain () const;
	virtual bool opSupportedImpl(Thyra::EOpTransp) const;
    virtual void applyImpl(const Thyra::EOpTransp M_trans, const Thyra::MultiVectorBase<double> &X, 
        const Teuchos::Ptr< Thyra::MultiVectorBase<double> > &Y, const double alpha, const double beta) const;

private:



};


}
}

#endif

