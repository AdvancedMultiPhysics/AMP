#include "AMP/operators/subchannel/CoupledFlowFrapconOperator.h"
#include "AMP/operators/subchannel/CoupledFlowFrapconOperatorParameters.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Operator {

CoupledFlowFrapconOperator::CoupledFlowFrapconOperator(
    std::shared_ptr<const OperatorParameters> params )
    : ColumnOperator( params )
{
    d_Mesh        = params->d_Mesh;
    auto myparams = std::dynamic_pointer_cast<const CoupledFlowFrapconOperatorParameters>( params );
    d_operators.push_back( myparams->d_Map3to1 );

    std::string flowOutVar = ( ( myparams->d_Map1to3 )->getOutputVariable() )->getName();

    std::string flowInpVar = ( ( myparams->d_FlowOperator )->getOutputVariable() )->getName();
    d_SimpleVariable.reset( new AMP::LinearAlgebra::Variable( flowInpVar ) );

    d_numpoints = std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( myparams->d_Map1to3 )
                      ->getNumZlocations();
    d_zPoints.resize( d_numpoints );

    d_flowInput  = AMP::LinearAlgebra::createSimpleVector<double>( d_numpoints, d_SimpleVariable );
    d_flowOutput = AMP::LinearAlgebra::createSimpleVector<double>( d_numpoints, d_SimpleVariable );

    auto tmp_db1 = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db1->putScalar( "BoundaryId", 4 );
    tmp_db1->putScalar( "InputVariable", flowOutVar );
    tmp_db1->putScalar( "OutputVariable", "FlowInternal" );
    auto mapflowInternal3to1Params =
        std::make_shared<AMP::Operator::MapOperatorParameters>( tmp_db1 );
    mapflowInternal3to1Params->d_Mesh =
        std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( myparams->d_Map3to1 )->getMesh();
    mapflowInternal3to1Params->d_MapMesh =
        std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( myparams->d_Map3to1 )->getMesh();
    mapflowInternal3to1Params->d_MapComm = mapflowInternal3to1Params->d_MapMesh->getComm();
    d_flowInternal3to1.reset( new AMP::Operator::Map3Dto1D( mapflowInternal3to1Params ) );

    std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( d_flowInternal3to1 )
        ->setVector( d_flowInput );

    d_operators.push_back( d_flowInternal3to1 );
    d_operators.push_back( myparams->d_FlowOperator );

    auto tmp_db2 = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db2->putScalar( "BoundaryId", 4 );
    tmp_db2->putScalar( "InputVariable", "FlowInternal" );
    tmp_db2->putScalar( "OutputVariable", flowOutVar );
    auto mapflowInternal1to3Params =
        std::make_shared<AMP::Operator::MapOperatorParameters>( tmp_db2 );
    mapflowInternal1to3Params->d_Mesh =
        ( std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( myparams->d_Map3to1 ) )->getMesh();
    mapflowInternal1to3Params->d_MapMesh =
        ( std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( myparams->d_Map3to1 ) )->getMesh();
    mapflowInternal1to3Params->d_MapComm = mapflowInternal1to3Params->d_MapMesh->getComm();
    d_flowInternal1to3.reset( new AMP::Operator::Map1Dto3D( mapflowInternal1to3Params ) );

    ( std::dynamic_pointer_cast<AMP::Operator::Map3Dto1D>( d_flowInternal3to1 ) )
        ->setZLocations(
            ( std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( d_flowInternal1to3 ) )
                ->getZLocations() );

    d_operators.push_back( d_flowInternal1to3 );
    d_operators.push_back( myparams->d_Map1to3 );
}

void CoupledFlowFrapconOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                        AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rInternal = subsetInputVector( r );
    std::dynamic_pointer_cast<AMP::Operator::Map1Dto3D>( d_operators[4] )->setVector( rInternal );

    d_operators[0]->apply( u, nullptr );
    d_operators[1]->apply( u, nullptr );
    d_operators[2]->apply( d_flowInput, d_flowOutput );
    // d_operators[3]->apply(nullptr, d_flowInput, nullptr, a, b);  // Is this necessary
    d_operators[4]->apply( d_flowOutput, nullptr );
}
} // namespace AMP::Operator
