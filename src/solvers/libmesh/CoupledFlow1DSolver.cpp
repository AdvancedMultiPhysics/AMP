#include "AMP/solvers/libmesh/CoupledFlow1DSolver.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/operators/subchannel/FlowFrapconJacobian.h"
#include "AMP/solvers/libmesh/Flow1DSolver.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Solver {

CoupledFlow1DSolver::CoupledFlow1DSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters ), d_numpoints( 0 )
{
    AMP_ASSERT( parameters );

    auto params = std::dynamic_pointer_cast<CoupledFlow1DSolverParameters>( parameters );

    std::string flowOutVar = params->d_pOperator->getOutputVariable()->getName();

    d_pOperator =
        std::dynamic_pointer_cast<AMP::Operator::CoupledFlowFrapconOperator>( params->d_pOperator );

    d_flow1DSolver         = std::dynamic_pointer_cast<Flow1DSolver>( params->d_flow1DSolver );
    std::string flowInpVar = ( d_flow1DSolver->getInputVariable() )->getName();

    auto tmp_db1 = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db1->putScalar( "BoundaryId", 4 );
    tmp_db1->putScalar( "InputVariable", flowOutVar );
    tmp_db1->putScalar( "OutputVariable", flowInpVar );
    auto mapflowInternal3to1Params =
        std::make_shared<AMP::Operator::MapOperatorParameters>( tmp_db1 );
    mapflowInternal3to1Params->d_MapMesh = ( d_flow1DSolver->getOperator() )->getMesh();
    mapflowInternal3to1Params->d_MapComm = mapflowInternal3to1Params->d_MapMesh->getComm();
    d_flowInternal3to1 = std::make_shared<AMP::Operator::Map3Dto1D>( mapflowInternal3to1Params );

    std::shared_ptr<AMP::Database> tmp_db2 = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db2->putScalar( "BoundaryId", 4 );
    tmp_db2->putScalar( "InputVariable", flowInpVar );
    tmp_db2->putScalar( "OutputVariable", flowOutVar );
    auto mapflowInternal1to3Params =
        std::make_shared<AMP::Operator::MapOperatorParameters>( tmp_db2 );
    mapflowInternal1to3Params->d_MapMesh = ( d_flow1DSolver->getOperator() )->getMesh();
    mapflowInternal1to3Params->d_MapComm = mapflowInternal1to3Params->d_MapMesh->getComm();
    d_flowInternal1to3 = std::make_shared<AMP::Operator::Map1Dto3D>( mapflowInternal1to3Params );

    std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( d_flowInternal3to1 )
        ->setZLocations( std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( d_flowInternal1to3 )
                             ->getZLocations() );

    int d_numpoints = ( std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( d_flowInternal1to3 ) )
                          ->getNumZlocations();
    d_SimpleVariable = std::make_shared<AMP::LinearAlgebra::Variable>( flowInpVar );

    d_flowInput  = AMP::LinearAlgebra::createSimpleVector<double>( d_numpoints, d_SimpleVariable );
    d_flowOutput = AMP::LinearAlgebra::createSimpleVector<double>( d_numpoints, d_SimpleVariable );
}

CoupledFlow1DSolver::~CoupledFlow1DSolver() = default;

void CoupledFlow1DSolver::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> ) {}

void CoupledFlow1DSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{

    if ( d_pOperator ) {}
}

void CoupledFlow1DSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator ) {
        d_pOperator->reset( params );
    }
}

void CoupledFlow1DSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    d_inpVariable = d_flow1DSolver->getOperator()->getInputVariable();
    d_outVariable = d_flowInternal1to3->getOutputVariable();

    d_Sol = u->subsetVectorForVariable( d_outVariable );
    d_Rhs = f->subsetVectorForVariable( d_outVariable );
    AMP_ASSERT( d_Rhs->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    ( std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( d_flowInternal3to1 ) )
        ->setVector( d_flowInput );
    ( std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( d_flowInternal1to3 ) )
        ->setVector( d_Sol );

    d_flowInternal3to1->apply( d_Rhs, nullVec );
    d_flow1DSolver->apply( d_flowInput, d_flowOutput );
    d_flowInternal1to3->apply( d_flowOutput, nullVec );
}
} // namespace AMP::Solver
