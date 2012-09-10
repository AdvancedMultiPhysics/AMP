#include "CoupledFlowFrapconOperator.h"
#include "CoupledFlowFrapconOperatorParameters.h"
#include "utils/Utilities.h"
#include "vectors/SimpleVector.h"


namespace AMP {
namespace Operator {

    CoupledFlowFrapconOperator::CoupledFlowFrapconOperator(const boost::shared_ptr<OperatorParameters>& params)
      : ColumnOperator(params)
    {
      d_Mesh = params->d_Mesh;
      boost::shared_ptr<CoupledFlowFrapconOperatorParameters> myparams = boost::dynamic_pointer_cast<CoupledFlowFrapconOperatorParameters>(params);
      d_Operators.push_back(myparams->d_Map3to1);

      std::string flowOutVar =( (myparams->d_Map1to3)->getOutputVariable())->getName() ;

      std::string flowInpVar =( (myparams->d_FlowOperator)->getOutputVariable())->getName() ;
      d_SimpleVariable.reset(new AMP::LinearAlgebra::Variable(flowInpVar ));

      d_numpoints = (boost::dynamic_pointer_cast<AMP::Operator::Map1Dto3D> (myparams->d_Map1to3) )->getNumZlocations();
      d_zPoints.resize(d_numpoints);

      d_flowInput = AMP::LinearAlgebra::SimpleVector::create( d_numpoints, d_SimpleVariable ); 
      d_flowOutput = AMP::LinearAlgebra::SimpleVector::create( d_numpoints, d_SimpleVariable ); 

      boost::shared_ptr<AMP::InputDatabase> tmp_db1 (new AMP::InputDatabase("Dummy"));
      tmp_db1->putInteger("BoundaryId",4);
      tmp_db1->putString("InputVariable",flowOutVar );
      tmp_db1->putString("OutputVariable","FlowInternal" );
      boost::shared_ptr<AMP::Operator::MapOperatorParameters> mapflowInternal3to1Params (new AMP::Operator::MapOperatorParameters( tmp_db1 ));
      mapflowInternal3to1Params->d_Mesh    = (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (myparams->d_Map3to1) )->getMesh();
      mapflowInternal3to1Params->d_MapMesh = (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (myparams->d_Map3to1) )->getMesh();
      mapflowInternal3to1Params->d_MapComm = mapflowInternal3to1Params->d_MapMesh->getComm();
      d_flowInternal3to1.reset(new AMP::Operator::Map3Dto1D(mapflowInternal3to1Params) );

      (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (d_flowInternal3to1))->setVector(d_flowInput);

      d_Operators.push_back(d_flowInternal3to1);
      d_Operators.push_back(myparams->d_FlowOperator);

      boost::shared_ptr<AMP::InputDatabase> tmp_db2 (new AMP::InputDatabase("Dummy"));
      tmp_db2->putInteger("BoundaryId",4);
      tmp_db2->putString("InputVariable","FlowInternal" );
      tmp_db2->putString("OutputVariable", flowOutVar );
      boost::shared_ptr<AMP::Operator::MapOperatorParameters> mapflowInternal1to3Params (new AMP::Operator::MapOperatorParameters( tmp_db2 ));
      mapflowInternal1to3Params->d_Mesh    = (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (myparams->d_Map3to1) )->getMesh();
      mapflowInternal1to3Params->d_MapMesh = (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (myparams->d_Map3to1) )->getMesh();
      mapflowInternal1to3Params->d_MapComm = mapflowInternal1to3Params->d_MapMesh->getComm();
      d_flowInternal1to3.reset(new AMP::Operator::Map1Dto3D(mapflowInternal1to3Params) );

      (boost::dynamic_pointer_cast<AMP::Operator::Map3Dto1D> (d_flowInternal3to1) )->setZLocations( (boost::dynamic_pointer_cast<AMP::Operator::Map1Dto3D> (d_flowInternal1to3) )->getZLocations());

      d_Operators.push_back(d_flowInternal1to3);
      d_Operators.push_back(myparams->d_Map1to3);
    }

    void
      CoupledFlowFrapconOperator :: apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
          AMP::LinearAlgebra::Vector::const_shared_ptr u, AMP::LinearAlgebra::Vector::shared_ptr r,
          const double a, const double b)
      {
        AMP::LinearAlgebra::Vector::shared_ptr   nullVec;

        //AMP::LinearAlgebra::Variable::shared_ptr inpVar = (boost::dynamic_pointer_cast<AMP::Operator::Map1Dto3D> (d_Operators[3]) )->getOutputVariable();
        //AMP::LinearAlgebra::Vector::const_shared_ptr uInternal = subsetInputVector( u );
        AMP::LinearAlgebra::Vector::shared_ptr rInternal = subsetInputVector( r );
        //(boost::dynamic_pointer_cast<AMP::Operator::Map1Dto3D> (d_Operators[3]))->setVector(uInternal);  // Is this necessary
        (boost::dynamic_pointer_cast<AMP::Operator::Map1Dto3D> (d_Operators[4]))->setVector(rInternal);

        d_Operators[0]->apply(nullVec, u, nullVec, a, b);
        d_Operators[1]->apply(nullVec, u, nullVec, a, b);
        d_Operators[2]->apply(nullVec, d_flowInput, d_flowOutput, a, b);
        //d_Operators[3]->apply(nullVec, d_flowInput, nullVec, a, b);  // Is this necessary
        d_Operators[4]->apply(nullVec, d_flowOutput, nullVec, a, b);
      }


}
}


