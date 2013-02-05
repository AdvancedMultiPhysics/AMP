#include "solvers/TrilinosThyraModelEvaluator.h"
#include "solvers/TrilinosLinearOP.h"
#include "vectors/trilinos/ThyraVectorSpaceWrapper.h"
#include "utils/Utilities.h"

#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_LinearOpBase.hpp"

namespace AMP {
namespace Solver {


/****************************************************************
*  Constructors                                                 *
****************************************************************/
TrilinosThyraModelEvaluator::TrilinosThyraModelEvaluator( boost::shared_ptr<TrilinosThyraModelEvaluatorParameters> params )
{
    d_dofs = params->d_dofs;
    d_linearOP = Teuchos::RCP<TrilinosLinearOP>( new TrilinosLinearOP() );
}


/****************************************************************
*  Destructor                                                   *
****************************************************************/
TrilinosThyraModelEvaluator::~TrilinosThyraModelEvaluator()
{
}



/****************************************************************
*  Evaluate the model                                           *
****************************************************************/
void TrilinosThyraModelEvaluator::evalModelImpl( const ::Thyra::ModelEvaluatorBase::InArgs<double> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<double> &outArgs ) const
{
    AMP_ASSERT(nonnull(inArgs.get_x()));
		 
    const Thyra::ConstDetachedVectorView<double> x(inArgs.get_x());

    const Teuchos::RCP< Thyra::VectorBase<double> > f_out = outArgs.get_f();
    const Teuchos::RCP< Thyra::LinearOpBase<double> > W_out = outArgs.get_W_op();

    if (nonnull(f_out)) {
        // Evaluate the residual
        const Thyra::DetachedVectorView<double> f(f_out);
        AMP_ERROR("Not finished");
    }

    if (nonnull(W_out)) {
        // Get the jacobian
        AMP_ERROR("Not finished");
        /*Teuchos::RCP<Epetra_Operator> W_epetra = Thyra::get_Epetra_Operator(*W_out);
        Teuchos::RCP<Epetra_CrsMatrix> W_epetracrs = rcp_dynamic_cast<Epetra_CrsMatrix>(W_epetra);
        TEUCHOS_ASSERT(nonnull(W_epetracrs));
        Epetra_CrsMatrix& DfDx = *W_epetracrs;
        DfDx.PutScalar(0.0);
        //
        // Fill W = DfDx
        //
        // W = DfDx = [      1.0,  2*x[1] ]
        //            [ 2*d*x[0],     -d  ]
        //
        double values[2];
        int indexes[2];
        // Row [0]
        values[0] = 1.0;           indexes[0] = 0;
        values[1] = 2.0*x[1];      indexes[1] = 1;
        DfDx.SumIntoGlobalValues( 0, 2, values, indexes );
        // Row [1]
        values[0] = 2.0*d_*x[0];   indexes[0] = 0;
        values[1] = -d_;           indexes[1] = 1;
        DfDx.SumIntoGlobalValues( 1, 2, values, indexes );*/
    }
}


/****************************************************************
* Functions derived from Thyra::StateFuncModelEvaluatorBase     *
* that are not implimented yet                                  *
****************************************************************/
Teuchos::RCP<const ::Thyra::VectorSpaceBase<double> > TrilinosThyraModelEvaluator::get_x_space() const
{
    Teuchos::RCP<LinearAlgebra::ThyraVectorSpaceWrapper> vector_space(new LinearAlgebra::ThyraVectorSpaceWrapper(d_dofs));
    return vector_space;
}
Teuchos::RCP<const ::Thyra::VectorSpaceBase<double> > TrilinosThyraModelEvaluator::get_f_space() const
{
    Teuchos::RCP<LinearAlgebra::ThyraVectorSpaceWrapper> vector_space(new LinearAlgebra::ThyraVectorSpaceWrapper(d_dofs));
    return vector_space;
}
::Thyra::ModelEvaluatorBase::InArgs<double> TrilinosThyraModelEvaluator::getNominalValues() const
{
    AMP_ERROR("Not implimented yet");
    return ::Thyra::ModelEvaluatorBase::InArgs<double>();
}
Teuchos::RCP< ::Thyra::LinearOpBase<double> > TrilinosThyraModelEvaluator::create_W_op() const
{
    return d_linearOP;
}
void TrilinosThyraModelEvaluator::set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<double> >& W_factory)
{
    d_W_factory = W_factory;
}
Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<double> > TrilinosThyraModelEvaluator::get_W_factory() const
{
    return d_W_factory;
}
::Thyra::ModelEvaluatorBase::InArgs<double> TrilinosThyraModelEvaluator::createInArgs() const
{
    ::Thyra::ModelEvaluatorBase::InArgsSetup<double> inArgs;
    inArgs.setModelEvalDescription(this->description());
    inArgs.setSupports(::Thyra::ModelEvaluatorBase::IN_ARG_x);
    return inArgs;
}
::Thyra::ModelEvaluatorBase::OutArgs<double> TrilinosThyraModelEvaluator::createOutArgsImpl() const
{
    ::Thyra::ModelEvaluatorBase::OutArgsSetup<double> outArgs;
    outArgs.setModelEvalDescription(this->description());
    outArgs.setSupports(::Thyra::ModelEvaluatorBase::OUT_ARG_f);
    outArgs.setSupports(::Thyra::ModelEvaluatorBase::OUT_ARG_W_op);
    return outArgs;
}



}
}

