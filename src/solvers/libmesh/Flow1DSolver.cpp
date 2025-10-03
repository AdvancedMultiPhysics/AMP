#include "AMP/solvers/libmesh/Flow1DSolver.h"
#include "AMP/operators/subchannel/FlowFrapconJacobian.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"

namespace AMP::Solver {

Flow1DSolver::Flow1DSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{

    AMP_ASSERT( parameters );

    initialize( parameters );
}

Flow1DSolver::~Flow1DSolver() = default;

void Flow1DSolver::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> ) {}

void Flow1DSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    getFromInput( parameters->d_db );

    if ( d_pOperator ) {
        auto Operator =
            std::dynamic_pointer_cast<AMP::Operator::FlowFrapconJacobian>( d_pOperator );

        AMP_ASSERT( Operator );

        d_inpVariable = Operator->d_inpVariable;
        d_outVariable = Operator->d_outVariable;
        d_K           = Operator->d_K;
        d_De          = Operator->d_De;
        d_G           = Operator->d_G;
        d_Re          = Operator->d_Re;
        d_Pr          = Operator->d_Pr;
        d_Cp          = Operator->getCp();
        d_dCp         = Operator->getdCp();
        zPoints       = Operator->getZLocations();
        d_cladVec     = Operator->d_cladVec;
    }
}

void Flow1DSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{

    if ( d_pOperator ) {}
}

void Flow1DSolver::resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator ) {
        d_pOperator->reset( params );
    }
}

void Flow1DSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                          std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{

    auto flowInputVec = u->subsetVectorForVariable( d_inpVariable );
    auto flowRhsVec   = f->subsetVectorForVariable( d_inpVariable );

    d_numpoints = zPoints.size();

    for ( size_t i = 1; i < (size_t) d_numpoints; i++ ) {

        double cur_node, next_node;

        cur_node  = zPoints[i - 1];
        next_node = zPoints[i];

        // double Heff, he_z, T_b_i, T_b_im1, T_c_i;
        // double T_b =0, Tin, frhs;
        double Heff, he_z, T_b_i, T_b_im1;
        double frhs;

        Heff = ( 0.023 * d_K / d_De ) * std::pow( d_Re, 0.8 ) * std::pow( d_Pr, 0.4 );
        he_z = next_node - cur_node;

        // T_c_i   = d_cladVec->getValueByLocalID(i);
        T_b_im1 = flowInputVec->getValueByLocalID( i - 1 );
        T_b_i   = flowInputVec->getValueByLocalID( i );
        frhs    = flowRhsVec->getValueByLocalID( i );

        T_b_i =
            ( T_b_im1 + frhs ) * ( 1. / ( 1.0 + ( ( 4 * Heff * he_z ) / ( d_Cp * d_G * d_De ) ) ) );

        flowInputVec->setValuesByLocalID( 1, &i, &T_b_i );

    } // end for i
}


std::shared_ptr<AMP::LinearAlgebra::Variable> Flow1DSolver::getInputVariable( int )
{
    return d_inpVariable;
}
} // namespace AMP::Solver
