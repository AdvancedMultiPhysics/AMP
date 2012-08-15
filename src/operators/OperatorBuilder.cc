#include "OperatorBuilder.h"
#include "utils/Utilities.h"
#include "DirichletMatrixCorrection.h"
#include "DirichletVectorCorrection.h"
#include "MassMatrixCorrection.h"
#include "NeumannVectorCorrection.h"
#include "PressureBoundaryVectorCorrection.h"
#include "RobinMatrixCorrection.h"
#include "RobinVectorCorrection.h"
#include "ColumnBoundaryOperator.h"
#include "FlowFrapconOperator.h"
#include "FlowFrapconJacobian.h"
#include "SubchannelTwoEqLinearOperator.h"
#include "SubchannelTwoEqNonlinearOperator.h"
#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "FickSoretNonlinearFEOperator.h"
#include "VolumeIntegralOperator.h"
#include "MassLinearFEOperator.h"
#include "NeutronicsRhs.h"
#include "LinearBVPOperator.h"
#include "NonlinearBVPOperator.h"
#include "ParameterFactory.h"
#include "ElementOperationFactory.h"
#include "map/MapSurface.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/VectorBuilder.h"

#include "operators/mechanics/MechanicsConstants.h"
#include "ampmesh/StructuredMeshHelper.h"

#include <string>

namespace AMP {
namespace Operator {

boost::shared_ptr<Operator>
OperatorBuilder::createOperator(boost::shared_ptr<OperatorParameters>  in_params)
{
  boost::shared_ptr<Operator> retOperator;
  
  AMP_INSIST(in_params.get()!=NULL, "ERROR: OperatorBuilder::createOperator has NULL input");
  AMP_INSIST(in_params->d_db.get()!=NULL, "ERROR: OperatorBuilder::createOperator has NULL database pointer in in_params");
  
  std::string name = in_params->d_db->getString("name");
  
  if(name=="DirichletMatrixCorrection")
    {
      retOperator.reset(new DirichletMatrixCorrection(boost::dynamic_pointer_cast<DirichletMatrixCorrectionParameters>(in_params)));
    }
  else if (name=="DirichletVectorCorrection")
    {
      retOperator.reset(new DirichletVectorCorrection(boost::dynamic_pointer_cast<DirichletVectorCorrectionParameters>(in_params)));
    }
  else if (name=="NeumannVectorCorrection")
    {
      retOperator.reset(new NeumannVectorCorrection(boost::dynamic_pointer_cast<NeumannVectorCorrectionParameters>(in_params)));
    }
  else if (name=="PressureBoundaryVectorCorrection")
    {
      retOperator.reset(new PressureBoundaryVectorCorrection(boost::dynamic_pointer_cast<PressureBoundaryVectorCorrectionParameters>(in_params)));
    }
  else if (name=="RobinMatrixCorrection")
    {
      retOperator.reset(new RobinMatrixCorrection(boost::dynamic_pointer_cast<RobinMatrixCorrectionParameters>(in_params)));
    }
  else if (name=="RobinVectorCorrection")
    {
      retOperator.reset(new RobinVectorCorrection(boost::dynamic_pointer_cast<NeumannVectorCorrectionParameters>(in_params)));
    }
  else if(name=="MechanicsLinearFEOperator")
    {
      retOperator.reset(new MechanicsLinearFEOperator(boost::dynamic_pointer_cast<MechanicsLinearFEOperatorParameters>(in_params)));
    }
  else if(name=="MechanicsNonlinearFEOperator")
    {
      retOperator.reset(new MechanicsNonlinearFEOperator(boost::dynamic_pointer_cast<MechanicsNonlinearFEOperatorParameters>(in_params)));
    }
  else if(name=="DiffusionLinearFEOperator")
    {
      retOperator.reset(new DiffusionLinearFEOperator(boost::dynamic_pointer_cast<DiffusionLinearFEOperatorParameters>(in_params)));
    }
  else if(name=="DiffusionNonlinearFEOperator")
    {
      retOperator.reset(new DiffusionNonlinearFEOperator(boost::dynamic_pointer_cast<DiffusionNonlinearFEOperatorParameters>(in_params)));
    }
  else if(name=="FickSoretNonlinearFEOperator")
    {
      retOperator.reset(new FickSoretNonlinearFEOperator(boost::dynamic_pointer_cast<FickSoretNonlinearFEOperatorParameters>(in_params)));
    }
  else if(name=="VolumeIntegralOperator")
    {
      retOperator.reset(new VolumeIntegralOperator(boost::dynamic_pointer_cast<VolumeIntegralOperatorParameters>(in_params)));
    }
  else if((name=="LinearBVPOperator")||(name=="NonlinearBVPOperator"))
    {
      boost::shared_ptr<BVPOperatorParameters> bvpOperatorParameters = boost::dynamic_pointer_cast<BVPOperatorParameters>(in_params);
      
      AMP_INSIST(bvpOperatorParameters.get()!=NULL, "ERROR: NULL BVPOperatorParameters passed");
      AMP_INSIST(bvpOperatorParameters->d_volumeOperatorParams.get()!=NULL,"ERROR: BVPOperatorParameters has NULL volumeOperatorParams pointer");
      AMP_INSIST(bvpOperatorParameters->d_boundaryOperatorParams.get()!=NULL,"ERROR: BVPOperatorParameters has NULL boundaryOperatorParams pointer");
      
      bvpOperatorParameters->d_volumeOperator = OperatorBuilder::createOperator(bvpOperatorParameters->d_volumeOperatorParams);
      bvpOperatorParameters->d_boundaryOperator = boost::dynamic_pointer_cast<BoundaryOperator>(OperatorBuilder::createOperator(bvpOperatorParameters->d_boundaryOperatorParams));
      
      if(name=="LinearBVPOperator")
        {
          retOperator.reset(new LinearBVPOperator(boost::dynamic_pointer_cast<BVPOperatorParameters>(in_params)));
        }
      else
        {
          retOperator.reset(new NonlinearBVPOperator(boost::dynamic_pointer_cast<BVPOperatorParameters>(in_params)));
        }
    }
  else if (name=="FlowFrapconOperator")
    {
      retOperator.reset(new FlowFrapconOperator(boost::dynamic_pointer_cast<FlowFrapconOperatorParameters>(in_params)));
    }
  else if (name=="FlowFrapconJacobian")
    {
      retOperator.reset(new FlowFrapconJacobian(boost::dynamic_pointer_cast<FlowFrapconJacobianParameters>(in_params)));
    }
  else if(name=="NeutronicsRhs")
    {
      retOperator.reset(new NeutronicsRhs(boost::dynamic_pointer_cast<NeutronicsRhsParameters>(in_params)));
    }
  else if(name=="Mesh3Dto1D")
    {
      //    retOperator.reset(new Mesh3Dto1D(boost::dynamic_pointer_cast<OperatorParameters>(in_params)));
    }
  
  return retOperator;
}

boost::shared_ptr<Operator>
OperatorBuilder::createOperator(AMP::Mesh::Mesh::shared_ptr meshAdapter,
				std::string operatorName,
				boost::shared_ptr<AMP::Database> tmp_input_db,
                                boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
                                boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory	)
{

  boost::shared_ptr<Operator> retOperator;

  boost::shared_ptr<AMP::InputDatabase> input_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(tmp_input_db);

  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(operatorName));

  AMP_INSIST(operator_db.get()!=NULL, "Error:: OperatorBuilder::createOperator(): No operator database entry with given name exists in input database");

  // we create the element physics model if a database entry exists
  // and the incoming element physics model pointer is NULL
  if( (elementPhysicsModel.get()==NULL) && (operator_db->keyExists("LocalModel" ) ) )
  {
    // extract the name of the local model from the operator database
    std::string localModelName = operator_db->getString("LocalModel");
    // check whether a database exists in the global database
    // (NOTE: not the operator database) with the given name
      AMP_INSIST(input_db->keyExists(localModelName), "Error:: OperatorBuilder::createOperator(): No local model database entry with given name exists in input database");

      boost::shared_ptr<AMP::Database> localModel_db = input_db->getDatabase(localModelName);
      AMP_INSIST(localModel_db.get()!=NULL, "Error:: OperatorBuilder::createOperator(): No local model database entry with given name exists in input databaseot");

      //If a non-NULL factory is being supplied through the argument list
      // use it, else call the AMP ElementPhysicsModelFactory interface
      if(localModelFactory.get()!=NULL)
	{
	  elementPhysicsModel = localModelFactory->createElementPhysicsModel(localModel_db);
	}
      else
	{
	  elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(localModel_db);
	}

      AMP_INSIST(elementPhysicsModel.get()!=NULL, "Error:: OperatorBuilder::createOperator(): local model creation failed");
      
    }

  std::string operatorType =operator_db->getString("name");
  
  if(operatorType=="MechanicsLinearFEOperator")
    {
      retOperator = OperatorBuilder::createLinearMechanicsOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="MechanicsNonlinearFEOperator")
    {
      retOperator = OperatorBuilder::createNonlinearMechanicsOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="DiffusionLinearFEOperator")
    {
      retOperator = OperatorBuilder::createLinearDiffusionOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="DiffusionNonlinearFEOperator")
    {
      retOperator = OperatorBuilder::createNonlinearDiffusionOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="FickSoretNonlinearFEOperator")
    {
      retOperator = OperatorBuilder::createNonlinearFickSoretOperator(meshAdapter,
								      operatorName,
								      input_db,
								      elementPhysicsModel,
								      localModelFactory);
    }
  else if(operatorType=="FlowFrapconOperator")
    {
        retOperator = OperatorBuilder::createFlowFrapconOperator(meshAdapter, operator_db);
    }
  else if(operatorType=="FlowFrapconJacobian")
    {
      retOperator = OperatorBuilder::createFlowFrapconJacobian(meshAdapter, operator_db);
    }
  else if(operatorType=="SubchannelTwoEqLinearOperator")
    {
        retOperator = OperatorBuilder::createSubchannelTwoEqLinearOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="SubchannelTwoEqNonlinearOperator")
    {
        retOperator = OperatorBuilder::createSubchannelTwoEqNonlinearOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="NeutronicsRhsOperator")
    {
      retOperator = OperatorBuilder::createNeutronicsRhsOperator(meshAdapter, operator_db);
    }
  else if(operatorType=="MassLinearFEOperator")
    {
      retOperator = OperatorBuilder::createMassLinearFEOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="VolumeIntegralOperator")
    {
      retOperator = OperatorBuilder::createVolumeIntegralOperator(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if(operatorType=="LinearBVPOperator")
    {
      // note that we pass in the full database here and not the operator db
      retOperator = OperatorBuilder::createLinearBVPOperator(meshAdapter, operatorName, input_db, elementPhysicsModel, localModelFactory);
    }
  else if(operatorType=="NonlinearBVPOperator")
    {
      // note that we pass in the full database here and not the operator db
      retOperator = OperatorBuilder::createNonlinearBVPOperator(meshAdapter, operatorName, input_db, elementPhysicsModel, localModelFactory);
    }
  else if(operatorType=="DirichletMatrixCorrection")
    {
    }
  else if (operatorType=="DirichletVectorCorrection")
    {
      retOperator = OperatorBuilder::createDirichletVectorCorrection(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if (operatorType=="NeumannVectorCorrection")
    {
    }
  else if (operatorType=="PressureBoundaryVectorCorrection")
    {
      retOperator = OperatorBuilder::createPressureBoundaryVectorCorrection(meshAdapter, operator_db, elementPhysicsModel);
    }
  else if (operatorType=="RobinMatrixCorrection")
    {
    }
  
  return retOperator;
}
  
AMP::Operator::Operator::shared_ptr
OperatorBuilder::createFlowFrapconOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					    boost::shared_ptr<AMP::InputDatabase> input_db)
{
  
  // now create the flow frapcon operator
  boost::shared_ptr<AMP::Database> flowOp_db;
  if(input_db->getString("name")=="FlowFrapconOperator")
    {
      flowOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  
  AMP_INSIST(flowOp_db.get()!=NULL, "Error: The database object for FlowFrapconOperator is NULL");
  
  boost::shared_ptr<AMP::Operator::FlowFrapconOperatorParameters> flowOpParams(new AMP::Operator::FlowFrapconOperatorParameters( flowOp_db ));
  flowOpParams->d_Mesh = meshAdapter;
  boost::shared_ptr<AMP::Operator::FlowFrapconOperator> flowOp (new AMP::Operator::FlowFrapconOperator( flowOpParams ));
  
  return flowOp;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createSubchannelTwoEqLinearOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					    boost::shared_ptr<AMP::InputDatabase> input_db,
						boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{
    // first create a SubchannelPhysicsModel
    boost::shared_ptr<AMP::Operator::SubchannelPhysicsModel> transportModel;

    if(elementPhysicsModel.get()!=NULL)
    {
      transportModel = boost::dynamic_pointer_cast<AMP::Operator::SubchannelPhysicsModel>(elementPhysicsModel);
    }
    else
    {
      boost::shared_ptr<AMP::Database> transportModel_db;
      if (input_db->keyExists("SubchannelPhysicsModel")) {
        transportModel_db = input_db->getDatabase("SubchannelPhysicsModel");
      } else {
        AMP_INSIST(false, "Key ''SubchannelPhysicsModel'' is missing!");
      }
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
      transportModel = boost::dynamic_pointer_cast<SubchannelPhysicsModel>(elementPhysicsModel);
    }

    AMP_INSIST(transportModel.get()!=NULL,"NULL transport model");
    // create the operator
    boost::shared_ptr<AMP::Database> subchannel_db;
    if(input_db->getString("name")=="SubchannelTwoEqLinearOperator")
    {
      subchannel_db = input_db;
    }
    else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }

    AMP_INSIST(subchannel_db.get()!=NULL, "Error: The database object for SubchannelTwoEqLinearOperator is NULL");

    boost::shared_ptr<AMP::Operator::SubchannelOperatorParameters> subchannelParams(new AMP::Operator::SubchannelOperatorParameters( subchannel_db ));
    subchannelParams->d_Mesh = meshAdapter;
    subchannelParams->d_subchannelPhysicsModel = transportModel ;

    subchannelParams->d_dofMap = AMP::Discretization::simpleDOFManager::create( meshAdapter, 
        AMP::Mesh::StructuredMeshHelper::getXYFaceIterator(meshAdapter,1), AMP::Mesh::StructuredMeshHelper::getXYFaceIterator(meshAdapter,0), 2);
    boost::shared_ptr<AMP::Operator::SubchannelTwoEqLinearOperator> subchannelOp (new AMP::Operator::SubchannelTwoEqLinearOperator( subchannelParams ));

    return subchannelOp;
}

  AMP::Operator::Operator::shared_ptr
OperatorBuilder::createSubchannelTwoEqNonlinearOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
    boost::shared_ptr<AMP::InputDatabase> input_db,
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{

  // first create a SubchannelPhysicsModel
  boost::shared_ptr<AMP::Operator::SubchannelPhysicsModel> transportModel;

  if(elementPhysicsModel.get()!=NULL)
  {
    transportModel = boost::dynamic_pointer_cast<AMP::Operator::SubchannelPhysicsModel>(elementPhysicsModel);
  }
  else
  {
    boost::shared_ptr<AMP::Database> transportModel_db;
    if (input_db->keyExists("SubchannelPhysicsModel")) {
      transportModel_db = input_db->getDatabase("SubchannelPhysicsModel");
    } else {
      AMP_INSIST(false, "Key ''SubchannelPhysicsModel'' is missing!");
    }
    elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
    transportModel = boost::dynamic_pointer_cast<SubchannelPhysicsModel>(elementPhysicsModel);
  }

  AMP_INSIST(transportModel.get()!=NULL,"NULL transport model");

  // create the operator
  boost::shared_ptr<AMP::Database> subchannel_db;
  if(input_db->getString("name")=="SubchannelTwoEqNonlinearOperator")
  {
    subchannel_db = input_db;
  }
  else
  {
    AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
  }

  AMP_INSIST(subchannel_db.get()!=NULL, "Error: The database object for SubchannelTwoEqNonlinearOperator is NULL");

  boost::shared_ptr<AMP::Operator::SubchannelOperatorParameters> subchannelParams(new AMP::Operator::SubchannelOperatorParameters( subchannel_db ));
  subchannelParams->d_Mesh = meshAdapter;
  subchannelParams->d_subchannelPhysicsModel = transportModel ;
  boost::shared_ptr<AMP::Operator::SubchannelTwoEqNonlinearOperator> subchannelOp (new AMP::Operator::SubchannelTwoEqNonlinearOperator( subchannelParams ));

  return subchannelOp;
}

  AMP::Operator::Operator::shared_ptr
OperatorBuilder::createNeutronicsRhsOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
    boost::shared_ptr<AMP::InputDatabase> input_db)
{

  // now create the Neutronics operator
  boost::shared_ptr<AMP::Database> NeutronicsOp_db;
  if(input_db->getString("name")=="NeutronicsRhsOperator")
  {
    NeutronicsOp_db = input_db;
  }
  else
  {
    AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
  }

  AMP_INSIST(NeutronicsOp_db.get()!=NULL, "Error: The database object for Neutronics Source Operator is NULL");

  boost::shared_ptr<AMP::Operator::NeutronicsRhsParameters> neutronicsOpParams(new AMP::Operator::NeutronicsRhsParameters( NeutronicsOp_db ));
  neutronicsOpParams->d_Mesh = meshAdapter;
  boost::shared_ptr<AMP::Operator::NeutronicsRhs> neutronicsOp (new AMP::Operator::NeutronicsRhs( neutronicsOpParams ));

  return neutronicsOp;
}

  AMP::Operator::Operator::shared_ptr
OperatorBuilder::createLinearDiffusionOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
    boost::shared_ptr<AMP::InputDatabase> input_db,
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{

  // first create a DiffusionTransportModel
  boost::shared_ptr<AMP::Operator::DiffusionTransportModel> transportModel;

  if(elementPhysicsModel.get()!=NULL)
  {
    transportModel = boost::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>(elementPhysicsModel);
  }
  else
  {
    boost::shared_ptr<AMP::Database> transportModel_db;
    if (input_db->keyExists("DiffusionTransportModel")) {
      transportModel_db = input_db->getDatabase("DiffusionTransportModel");
    } else {
      AMP_INSIST(false, "Key ''DiffusionTransportModel'' is missing!");
    }
    elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
    transportModel = boost::dynamic_pointer_cast<DiffusionTransportModel>(elementPhysicsModel);
  }

  AMP_INSIST(transportModel.get()!=NULL,"NULL transport model");

  // next create a ElementOperation object
  AMP_INSIST(input_db->keyExists("DiffusionElement"), "Key ''DiffusionElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> diffusionLinElem = ElementOperationFactory::createElementOperation(input_db->getDatabase("DiffusionElement"));

  // now create the linear diffusion operator
  boost::shared_ptr<AMP::Database> diffusionLinFEOp_db;
  if(input_db->getString("name")=="DiffusionLinearFEOperator")
  {
    diffusionLinFEOp_db = input_db;
  }
  else
  {
    AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
  }

  AMP_INSIST(diffusionLinFEOp_db.get()!=NULL, "Error: The database object for DiffusionLinearFEOperator is NULL");

  boost::shared_ptr<AMP::Operator::DiffusionLinearFEOperatorParameters> diffusionOpParams(new AMP::Operator::DiffusionLinearFEOperatorParameters( diffusionLinFEOp_db ));
  diffusionOpParams->d_transportModel = transportModel;
  diffusionOpParams->d_elemOp = diffusionLinElem;
  diffusionOpParams->d_Mesh = meshAdapter;
  diffusionOpParams->d_inDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 1, true);
  diffusionOpParams->d_outDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 1, true);
  boost::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> diffusionOp (new AMP::Operator::DiffusionLinearFEOperator( diffusionOpParams ));

  return diffusionOp;
}

  AMP::Operator::Operator::shared_ptr
OperatorBuilder::createVolumeIntegralOperator(AMP::Mesh::Mesh::shared_ptr meshAdapter,
    boost::shared_ptr<AMP::InputDatabase> input_db,
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{
  boost::shared_ptr<AMP::Operator::SourcePhysicsModel> sourcePhysicsModel;

  if(elementPhysicsModel.get()!=NULL)
  {
    sourcePhysicsModel = boost::dynamic_pointer_cast<AMP::Operator::SourcePhysicsModel>(elementPhysicsModel);
  }
  else
  {
    if(input_db->keyExists("SourcePhysicsModel"))
    {
      boost::shared_ptr<AMP::Database> sourceModel_db = input_db->getDatabase("SourcePhysicsModel");
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(sourceModel_db);
      sourcePhysicsModel = boost::dynamic_pointer_cast<SourcePhysicsModel>(elementPhysicsModel);
    }
  }

  // next create a ElementOperation object
  AMP_INSIST(input_db->keyExists("SourceElement"), "Key ''SourceElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> sourceNonlinearElem = ElementOperationFactory::createElementOperation(input_db->getDatabase("SourceElement"));

  // now create the nonlinear source operator
  boost::shared_ptr<AMP::Database> sourceNLinFEOp_db;
  if(input_db->getString("name")=="VolumeIntegralOperator")
  {
    sourceNLinFEOp_db = input_db;
  }
  else
  {
    AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
  }

  boost::shared_ptr<AMP::Operator::VolumeIntegralOperatorParameters> volumeIntegralParameters (
      new AMP::Operator::VolumeIntegralOperatorParameters( input_db ) );
  volumeIntegralParameters->d_sourcePhysicsModel = sourcePhysicsModel;
  volumeIntegralParameters->d_elemOp = sourceNonlinearElem;
  volumeIntegralParameters->d_Mesh = meshAdapter;
  boost::shared_ptr<AMP::Operator::VolumeIntegralOperator> nonlinearSourceOp (new AMP::Operator::VolumeIntegralOperator( volumeIntegralParameters ));

  return nonlinearSourceOp;
}

  AMP::Operator::Operator::shared_ptr
OperatorBuilder::createNonlinearDiffusionOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
    boost::shared_ptr<AMP::InputDatabase> input_db,
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{

  // first create a DiffusionTransportModel
  boost::shared_ptr<AMP::Operator::DiffusionTransportModel> transportModel;

  if(elementPhysicsModel.get()!=NULL)
  {
    transportModel = boost::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>(elementPhysicsModel);
  }
  else
    {
	  boost::shared_ptr<AMP::Database> transportModel_db;
      if (input_db->keyExists("DiffusionTransportModel")) {
    	  transportModel_db = input_db->getDatabase("DiffusionTransportModel");
      } else {
    	  AMP_INSIST(false, "Key ''DiffusionTransportModel'' is missing!");
      }
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
      transportModel = boost::dynamic_pointer_cast<DiffusionTransportModel>(elementPhysicsModel);
    }
  
  AMP_INSIST(transportModel.get()!=NULL,"NULL transport model");
  
  // next create an ElementOperation object
  AMP_INSIST(input_db->keyExists("DiffusionElement"), "Key ''DiffusionElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> diffusionNonlinearElem =
    ElementOperationFactory::createElementOperation(input_db->getDatabase("DiffusionElement"));
  
  // now create the nonlinear diffusion operator parameters
  boost::shared_ptr<AMP::Database> diffusionNLinFEOp_db;
  if(input_db->getString("name")=="DiffusionNonlinearFEOperator")
    {
      diffusionNLinFEOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  AMP_INSIST(diffusionNLinFEOp_db.get()!=NULL, "Error: The database object for DiffusionNonlinearFEOperator is NULL");
  boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperatorParameters> diffusionNLOpParams(
											       new AMP::Operator::DiffusionNonlinearFEOperatorParameters( diffusionNLinFEOp_db ));
  diffusionNLOpParams->d_transportModel = transportModel;
  diffusionNLOpParams->d_elemOp = diffusionNonlinearElem;
  diffusionNLOpParams->d_Mesh = meshAdapter;
  
  // populate the parameters with frozen active variable vectors
  
  // nullify vectors in parameters
  diffusionNLOpParams->d_FrozenTemperature .reset();
  diffusionNLOpParams->d_FrozenConcentration.reset();
  diffusionNLOpParams->d_FrozenBurnup.reset();
  
  // create variables and vectors for frozen material inputs
  boost::shared_ptr<AMP::Database> active_db = diffusionNLinFEOp_db->getDatabase("ActiveInputVariables");
  AMP::Discretization::DOFManager::shared_ptr NodalScalarDOF = AMP::Discretization::simpleDOFManager::create(meshAdapter,AMP::Mesh::Vertex,1,1,true);  
  std::string name;
  AMP::LinearAlgebra::Variable::shared_ptr tVar;
  AMP::LinearAlgebra::Vector::shared_ptr tVec;
  AMP::LinearAlgebra::Variable::shared_ptr cVar;
  AMP::LinearAlgebra::Vector::shared_ptr cVec;
  AMP::LinearAlgebra::Variable::shared_ptr bVar;
  AMP::LinearAlgebra::Vector::shared_ptr bVec;
  name  = active_db->getStringWithDefault("Temperature","not_specified");
  if (name != "not_specified") {
    tVar.reset(new AMP::LinearAlgebra::Variable(name));
    tVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, tVar, true );
    if (diffusionNLinFEOp_db->getBoolWithDefault("FreezeTemperature",false))
      diffusionNLOpParams->d_FrozenTemperature = tVec;
  }
  name  = active_db->getStringWithDefault("Concentration","not_specified");
  if (name != "not_specified") {
    cVar.reset(new AMP::LinearAlgebra::Variable(name));
    cVec =  AMP::LinearAlgebra::createVector( NodalScalarDOF, cVar, true );
    if (diffusionNLinFEOp_db->getBoolWithDefault("FreezeConcentration",false))
      diffusionNLOpParams->d_FrozenConcentration = cVec;
  }
  name  = active_db->getStringWithDefault("Burnup","not_specified");
  if (name != "not_specified") {
    bVar.reset(new AMP::LinearAlgebra::Variable(name));
    bVec =  AMP::LinearAlgebra::createVector( NodalScalarDOF, bVar, true );
    if (diffusionNLinFEOp_db->getBoolWithDefault("FreezeBurnup",false))
      diffusionNLOpParams->d_FrozenBurnup = bVec;
  }
  
  // create the nonlinear diffusion operator
  boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> nonlinearDiffusionOp (
										       new AMP::Operator::DiffusionNonlinearFEOperator( diffusionNLOpParams ));
  
  return nonlinearDiffusionOp;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createNonlinearFickSoretOperator(  AMP::Mesh::Mesh::shared_ptr meshAdapter,
						    std::string operatorName,
						    boost::shared_ptr<AMP::InputDatabase> input_db,
						    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
						    boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory)
{
  boost::shared_ptr<Operator> retOperator;
  AMP_INSIST(input_db.get()!=NULL, "NULL database object passed");
  
  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(operatorName));
  AMP_INSIST(operator_db.get()!=NULL, "NULL database object passed");

  std::string fickOperatorName = operator_db->getString("FickOperator");
  std::string soretOperatorName = operator_db->getString("SoretOperator");

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> fickPhysicsModel;
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> soretPhysicsModel;


  AMP::Operator::Operator::shared_ptr fickOperator = OperatorBuilder::createOperator(meshAdapter,
										     fickOperatorName,
										     input_db,
										     fickPhysicsModel,
										     localModelFactory);
  AMP_INSIST(fickOperator.get()!=NULL, "Error: unable to create Fick operator in OperatorBuilder::createFickSoretOperator");
  
  AMP::Operator::Operator::shared_ptr soretOperator = OperatorBuilder::createOperator(meshAdapter,
										      soretOperatorName,
										      input_db,
										      soretPhysicsModel,
										      localModelFactory);
  
  AMP_INSIST(soretOperator.get()!=NULL, "Error: unable to create Soret operator in OperatorBuilder::createFickSoretOperator");
  
  boost::shared_ptr<AMP::Database> db = boost::dynamic_pointer_cast<AMP::Database>(input_db);
  boost::shared_ptr<FickSoretNonlinearFEOperatorParameters> params(new FickSoretNonlinearFEOperatorParameters(db));
  params->d_Mesh = meshAdapter;
  params->d_FickOperator = boost::dynamic_pointer_cast<DiffusionNonlinearFEOperator>(fickOperator);
  params->d_SoretOperator = boost::dynamic_pointer_cast<DiffusionNonlinearFEOperator>(soretOperator);
  params->d_name = operatorName;
  FickSoretNonlinearFEOperator::shared_ptr fsOp(new FickSoretNonlinearFEOperator(params));
  
  return fsOp;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createLinearMechanicsOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
						boost::shared_ptr<AMP::InputDatabase> input_db,
						boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{
  
  // first create a MechanicsMaterialModel
  if(elementPhysicsModel.get()==NULL)
    {
      AMP_INSIST(input_db->keyExists("MechanicsMaterialModel"), "Key ''MechanicsMaterialModel'' is missing!");
      
      boost::shared_ptr<AMP::Database> materialModel_db = input_db->getDatabase("MechanicsMaterialModel");
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(materialModel_db);
    }
  
  AMP_INSIST(elementPhysicsModel.get()!=NULL,"NULL material model");
  
  // next create a ElementOperation object
  AMP_INSIST(input_db->keyExists("MechanicsElement"), "Key ''MechanicsElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> mechanicsLinElem = ElementOperationFactory::createElementOperation(input_db->getDatabase("MechanicsElement"));
  
  // now create the linear mechanics operator
  boost::shared_ptr<AMP::Database> mechanicsLinFEOp_db;
  if(input_db->getString("name")=="MechanicsLinearFEOperator")
    {
      mechanicsLinFEOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  
  AMP_INSIST(mechanicsLinFEOp_db.get()!=NULL, "Error: The database object for MechanicsLinearFEOperator is NULL");
  
  boost::shared_ptr<AMP::Operator::MechanicsLinearFEOperatorParameters> mechanicsOpParams(
      new AMP::Operator::MechanicsLinearFEOperatorParameters( mechanicsLinFEOp_db ));
  mechanicsOpParams->d_materialModel = boost::dynamic_pointer_cast<MechanicsMaterialModel>(elementPhysicsModel);
  mechanicsOpParams->d_elemOp = mechanicsLinElem;
  mechanicsOpParams->d_Mesh = meshAdapter;
  mechanicsOpParams->d_inDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 3, true);
  mechanicsOpParams->d_outDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 3, true);
  
  boost::shared_ptr<AMP::Operator::MechanicsLinearFEOperator> mechanicsOp (new AMP::Operator::MechanicsLinearFEOperator( mechanicsOpParams ));
  
  return mechanicsOp;
}


AMP::Operator::Operator::shared_ptr
OperatorBuilder::createNonlinearMechanicsOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
						   boost::shared_ptr<AMP::InputDatabase> input_db,
						   boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{
  
  // first create a MechanicsMaterialModel
  if(elementPhysicsModel.get()==NULL)
    {
      AMP_INSIST(input_db->keyExists("MechanicsMaterialModel"), "Key ''MechanicsMaterialModel'' is missing!");
      
      boost::shared_ptr<AMP::Database> transportModel_db = input_db->getDatabase("MechanicsMaterialModel");
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
    }
  
  AMP_INSIST(elementPhysicsModel.get()!=NULL,"NULL material model");
  
  // next create a ElementOperation object
  AMP_INSIST(input_db->keyExists("MechanicsElement"), "Key ''MechanicsElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> mechanicsElem = ElementOperationFactory::createElementOperation(input_db->getDatabase("MechanicsElement"));
  
  // now create the nonlinear mechanics operator
  boost::shared_ptr<AMP::Database> mechanicsFEOp_db;
  if(input_db->getString("name")=="MechanicsNonlinearFEOperator")
    {
      mechanicsFEOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  
  AMP_INSIST(mechanicsFEOp_db.get()!=NULL, "Error: The database object for MechanicsNonlinearFEOperator is NULL");
  
  boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperatorParameters> mechanicsOpParams(new AMP::Operator::MechanicsNonlinearFEOperatorParameters( mechanicsFEOp_db ));
  mechanicsOpParams->d_materialModel = boost::dynamic_pointer_cast<MechanicsMaterialModel>(elementPhysicsModel);
  mechanicsOpParams->d_elemOp = mechanicsElem;
  mechanicsOpParams->d_Mesh = meshAdapter;
  mechanicsOpParams->d_dofMap[Mechanics::DISPLACEMENT] = AMP::Discretization::simpleDOFManager::create(meshAdapter, 
      AMP::Mesh::Vertex, 1, 3, true);
  mechanicsOpParams->d_dofMap[Mechanics::TEMPERATURE] = AMP::Discretization::simpleDOFManager::create(meshAdapter, 
      AMP::Mesh::Vertex, 1, 1, true);
  mechanicsOpParams->d_dofMap[Mechanics::BURNUP] = AMP::Discretization::simpleDOFManager::create(meshAdapter, 
      AMP::Mesh::Vertex, 1, 1, true);
  mechanicsOpParams->d_dofMap[Mechanics::OXYGEN_CONCENTRATION] = AMP::Discretization::simpleDOFManager::create(meshAdapter, 
      AMP::Mesh::Vertex, 1, 1, true);
  mechanicsOpParams->d_dofMap[Mechanics::LHGR] = AMP::Discretization::simpleDOFManager::create(meshAdapter, 
      AMP::Mesh::Vertex, 1, 1, true);
  
  boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperator> mechanicsOp (new AMP::Operator::MechanicsNonlinearFEOperator( mechanicsOpParams ));
  
  return mechanicsOp;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createMassLinearFEOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					     boost::shared_ptr<AMP::InputDatabase> input_db,
					     boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel)
{
  
  // first create a MassDensityModel
  boost::shared_ptr<AMP::Operator::MassDensityModel> densityModel;
  
  if(elementPhysicsModel.get()!=NULL)
    {
      densityModel = boost::dynamic_pointer_cast<AMP::Operator::MassDensityModel>(elementPhysicsModel);
      }
  else
    {
      AMP_INSIST(input_db->keyExists("MassDensityModel"), "Key ''MassDensityModel'' is missing!");
      boost::shared_ptr<AMP::Database> densityModel_db = input_db->getDatabase("MassDensityModel");
      elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(densityModel_db);
      densityModel = boost::dynamic_pointer_cast<MassDensityModel>(elementPhysicsModel);
    }
  
  AMP_INSIST(densityModel.get()!=NULL,"NULL density model");
  
  // next create a ElementOperation object
  AMP_INSIST(input_db->keyExists("MassElement"), "Key ''MassElement'' is missing!");
  boost::shared_ptr<AMP::Operator::ElementOperation> densityLinElem = ElementOperationFactory::createElementOperation(input_db->getDatabase("MassElement"));
  
  // now create the linear density operator
  boost::shared_ptr<AMP::Database> densityLinFEOp_db;
  if(input_db->getString("name")=="MassLinearFEOperator")
    {
      densityLinFEOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  
  AMP_INSIST(densityLinFEOp_db.get()!=NULL, "Error: The database object for MassLinearFEOperator is NULL");
  
  boost::shared_ptr<AMP::Operator::MassLinearFEOperatorParameters> densityOpParams(new AMP::Operator::MassLinearFEOperatorParameters( densityLinFEOp_db ));
  densityOpParams->d_densityModel = densityModel;
  densityOpParams->d_elemOp = densityLinElem;
  densityOpParams->d_Mesh = meshAdapter;
  densityOpParams->d_inDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 1, true);
  densityOpParams->d_outDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 1, true);
  boost::shared_ptr<AMP::Operator::MassLinearFEOperator> densityOp (new AMP::Operator::MassLinearFEOperator( densityOpParams ));
  
  return densityOp;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createLinearBVPOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					  std::string operatorName,
					  boost::shared_ptr<AMP::InputDatabase> input_db,
					  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
					  boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory)
{
  boost::shared_ptr<Operator> retOperator;
  AMP_INSIST(input_db.get()!=NULL, "NULL database object passed");

  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(operatorName));
  AMP_INSIST(operator_db.get()!=NULL, "NULL database object passed");
  
  // create the volume operator
  std::string volumeOperatorName = operator_db->getString("VolumeOperator");
  // if this flag is true the same local physics model will be used for both boundary and volume operators
  bool useSameLocalModelForVolumeAndBoundaryOperators = operator_db->getBoolWithDefault("useSameLocalModelForVolumeAndBoundaryOperators", false);
  
  AMP::Operator::Operator::shared_ptr volumeOperator = OperatorBuilder::createOperator(meshAdapter,
										       volumeOperatorName,
										       input_db,
										       elementPhysicsModel,
										       localModelFactory);

  boost::shared_ptr<AMP::Operator::LinearOperator> volumeLinearOp = boost::dynamic_pointer_cast<AMP::Operator::LinearOperator>(volumeOperator);
  AMP_INSIST(volumeLinearOp.get()!=NULL, "Error: unable to create linear operator in OperatorBuilder::createLinearBVPOperator");
  
  // create the boundary operator
  std::string boundaryOperatorName = operator_db->getString("BoundaryOperator");
  boost::shared_ptr<AMP::InputDatabase> boundaryOperator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(boundaryOperatorName));
  AMP_INSIST(boundaryOperator_db.get()!=NULL, "NULL database object passed for boundary operator");
  
  boundaryOperator_db->putBool("isAttachedToVolumeOperator", true);

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> boundaryLocalModel;

  if(useSameLocalModelForVolumeAndBoundaryOperators)
    {
      boundaryLocalModel=elementPhysicsModel;
    }
  
  boost::shared_ptr<AMP::Operator::BoundaryOperator> boundaryOperator = OperatorBuilder::createBoundaryOperator(meshAdapter,
														boundaryOperatorName,
														input_db,
														volumeLinearOp,
														boundaryLocalModel,
														localModelFactory);
  
  boost::shared_ptr<AMP::Operator::BVPOperatorParameters> bvpOperatorParams (new AMP::Operator::BVPOperatorParameters(input_db));
  bvpOperatorParams->d_volumeOperator = volumeOperator;
  bvpOperatorParams->d_boundaryOperator = boundaryOperator;
  
  retOperator.reset(new AMP::Operator::LinearBVPOperator(bvpOperatorParams));
  
  return retOperator;
}

AMP::Operator::Operator::shared_ptr
OperatorBuilder::createNonlinearBVPOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					     std::string operatorName,
					     boost::shared_ptr<AMP::InputDatabase> input_db,
					     boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
					     boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory	)
{
  boost::shared_ptr<Operator> retOperator;
  AMP_INSIST(input_db.get()!=NULL, "NULL database object passed");
  
  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(operatorName));
  AMP_INSIST(operator_db.get()!=NULL, "NULL database object passed");
  
  // create the volume operator
  std::string volumeOperatorName = operator_db->getString("VolumeOperator");
  // if this flag is true the same local physics model will be used for both boundary and volume operators
  bool useSameLocalModelForVolumeAndBoundaryOperators = operator_db->getBoolWithDefault("useSameLocalModelForVolumeAndBoundaryOperators", false);

  AMP::Operator::Operator::shared_ptr volumeOperator = OperatorBuilder::createOperator(meshAdapter,
										       volumeOperatorName,
										       input_db,
										       elementPhysicsModel,
										       localModelFactory);
  AMP_INSIST(volumeOperator.get()!=NULL, "Error: unable to create nonlinear operator in OperatorBuilder::createNonlinearBVPOperator");
  
  // create the boundary operator
  std::string boundaryOperatorName = operator_db->getString("BoundaryOperator");
  boost::shared_ptr<AMP::InputDatabase> boundaryOperator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(boundaryOperatorName));
  AMP_INSIST(boundaryOperator_db.get()!=NULL, "NULL database object passed for boundary operator");
  
  boundaryOperator_db->putBool("isAttachedToVolumeOperator", true);

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> boundaryLocalModel;

  if(useSameLocalModelForVolumeAndBoundaryOperators)
    {
      boundaryLocalModel=elementPhysicsModel;
    }
  
  boost::shared_ptr<AMP::Operator::BoundaryOperator> boundaryOperator = OperatorBuilder::createBoundaryOperator(meshAdapter,
														boundaryOperatorName,
														input_db,
														volumeOperator,
														boundaryLocalModel,
														localModelFactory);
  
  boost::shared_ptr<AMP::Operator::BVPOperatorParameters> bvpOperatorParams (new AMP::Operator::BVPOperatorParameters(input_db));
  bvpOperatorParams->d_volumeOperator = volumeOperator;
  bvpOperatorParams->d_boundaryOperator = boundaryOperator;
  
  retOperator.reset(new AMP::Operator::NonlinearBVPOperator(bvpOperatorParams));
  
  return retOperator;
}
  
AMP::Operator::Operator::shared_ptr
OperatorBuilder::createFlowFrapconJacobian( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					    boost::shared_ptr<AMP::InputDatabase> input_db)
{
  
  // now create the flow frapcon operator
  boost::shared_ptr<AMP::Database> flowOp_db;
  if(input_db->getString("name")=="FlowFrapconJacobian")
    {
      flowOp_db = input_db;
    }
  else
    {
      AMP_INSIST(input_db->keyExists("name"), "Key ''name'' is missing!");
    }
  
  AMP_INSIST(flowOp_db.get()!=NULL, "Error: The database object for FlowFrapconJacobian is NULL");
  
  boost::shared_ptr<AMP::Operator::FlowFrapconJacobianParameters> flowOpParams(new AMP::Operator::FlowFrapconJacobianParameters( flowOp_db ));
  flowOpParams->d_Mesh = meshAdapter;
  boost::shared_ptr<AMP::Operator::FlowFrapconJacobian> flowOp (new AMP::Operator::FlowFrapconJacobian( flowOpParams ));
  
  return flowOp;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createBoundaryOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					 std::string boundaryOperatorName,
					 boost::shared_ptr<AMP::InputDatabase> input_db,
					 AMP::Operator::Operator::shared_ptr volumeOperator,
					 boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
					 boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  AMP_INSIST(input_db.get()!=NULL, "NULL database object passed");
  
  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(boundaryOperatorName));
  AMP_INSIST(operator_db.get()!=NULL, "Error: OperatorBuilder::createBoundaryOperator(): database object with given name not in database");  
  
  // we create the element physics model if a database entry exists
  // and the incoming element physics model pointer is NULL
//  if( (elementPhysicsModel.get()==NULL) && (operator_db->keyExists("LocalModel" ) ) ) 
//  The above Condition assumes all of the Operators inside column boundary use same Physics Model - SA
  if( (operator_db->keyExists("LocalModel" ) ) )
    {
      // extract the name of the local model from the operator database
      std::string localModelName = operator_db->getString("LocalModel");
      // check whether a database exists in the global database
      // (NOTE: not the operator database) with the given name
      AMP_INSIST(input_db->keyExists(localModelName), "Error:: OperatorBuilder::createOperator(): No local model database entry with given name exists in input database");
      
      boost::shared_ptr<AMP::Database> localModel_db = input_db->getDatabase(localModelName);
      AMP_INSIST(localModel_db.get()!=NULL, "Error:: OperatorBuilder::createOperator(): No local model database entry with given name exists in input database");
      
      //if a non-NULL factory is being supplied through the argument list
      // use it, else call the AMP ElementPhysicsModelFactory interface
      if(localModelFactory.get()!=NULL)
	{
	  elementPhysicsModel = localModelFactory->createElementPhysicsModel(localModel_db);
	}
      else
	{
	  elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel(localModel_db);
	}
      
      AMP_INSIST(elementPhysicsModel.get()!=NULL, "Error:: OperatorBuilder::createOperator(): local model creation failed");
      
    }
  
  std::string boundaryType = operator_db->getString("name");
  
  if(boundaryType=="DirichletMatrixCorrection")
    {
      // in this case the volume operator has to be a linear operator
      retOperator = createDirichletMatrixCorrection(meshAdapter,
						    operator_db,
						    volumeOperator,
						    elementPhysicsModel);
    }
  else if(boundaryType=="MassMatrixCorrection")
    {
      retOperator = createMassMatrixCorrection(meshAdapter,
					       operator_db,
					       volumeOperator,
					       elementPhysicsModel);
    }
  else if(boundaryType=="RobinMatrixCorrection")
    {
      // in this case the volume operator has to be a linear operator
      retOperator = createRobinMatrixCorrection(meshAdapter,
						operator_db,
						volumeOperator,
						elementPhysicsModel);
    }
  else if(boundaryType=="RobinVectorCorrection")
    {
      retOperator = createRobinVectorCorrection(meshAdapter,
						operator_db,
						volumeOperator,
						elementPhysicsModel);
    }
  else if(boundaryType=="NeumannVectorCorrection")
    {
      retOperator = createNeumannVectorCorrection(meshAdapter,
						  operator_db,
						  volumeOperator,
						  elementPhysicsModel);
    }
  else if(boundaryType=="DirichletVectorCorrection")
    {
      // in this case the volume operator has to be a nonlinear operator
      retOperator = createDirichletVectorCorrection(meshAdapter,
						    operator_db,
						    volumeOperator,
						    elementPhysicsModel);
    }
  else if(boundaryType=="PressureBoundaryVectorCorrection")
    {
      retOperator = createPressureBoundaryVectorCorrection(meshAdapter,
							   operator_db,
							   volumeOperator,
							   elementPhysicsModel);
    }
  else if(boundaryType=="ColumnBoundaryOperator")
    {
      // note that the global input database is passed here instead of the operator
      // database
      retOperator = createColumnBoundaryOperator(meshAdapter,
						 boundaryOperatorName,
						 input_db,
						 volumeOperator,
						 elementPhysicsModel,
						 localModelFactory);
    }
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createColumnBoundaryOperator(AMP::Mesh::Mesh::shared_ptr meshAdapter,
					      std::string boundaryOperatorName,
					      boost::shared_ptr<AMP::InputDatabase> input_db,
					      AMP::Operator::Operator::shared_ptr volumeOperator,
					      boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel,
					      boost::shared_ptr<AMP::Operator::ElementPhysicsModelFactory> localModelFactory)
{

  AMP_INSIST(input_db.get()!=NULL, "NULL database object passed");
  
  boost::shared_ptr<AMP::InputDatabase> operator_db = boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase(boundaryOperatorName));
  AMP_INSIST(operator_db.get()!=NULL, "Error: OperatorBuilder::createBoundaryOperator(): database object with given name not in database");  

  int numberOfBoundaryOperators =  operator_db->getIntegerWithDefault("numberOfBoundaryOperators", 1);
  
  std::string *boundaryOps = new std::string[numberOfBoundaryOperators];
  operator_db->getStringArray("boundaryOperators", boundaryOps, numberOfBoundaryOperators);
  
  boost::shared_ptr<OperatorParameters> params(new OperatorParameters(operator_db));
  params->d_Mesh=meshAdapter;
  
  boost::shared_ptr<AMP::Operator::ColumnBoundaryOperator> columnBoundaryOperator(new AMP::Operator::ColumnBoundaryOperator(params));
  
  for(int i=0; i<numberOfBoundaryOperators; i++)
    {
      boost::shared_ptr<BoundaryOperator> bcOperator = OperatorBuilder::createBoundaryOperator(meshAdapter,
											       boundaryOps[i],
											       input_db,
											       volumeOperator,
											       elementPhysicsModel,
											       localModelFactory);
      AMP_ASSERT ( bcOperator );
      columnBoundaryOperator->append(bcOperator);
    }
  delete [] boundaryOps;
  
  return columnBoundaryOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createDirichletMatrixCorrection( AMP::Mesh::Mesh::shared_ptr meshAdapter,
						  boost::shared_ptr<AMP::InputDatabase> input_db,
						  AMP::Operator::Operator::shared_ptr volumeOperator,
						  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearOperator>(volumeOperator);
  boost::shared_ptr<AMP::Operator::DirichletMatrixCorrectionParameters> matrixCorrectionParameters (new AMP::Operator::DirichletMatrixCorrectionParameters( input_db ) );
  matrixCorrectionParameters->d_variable = linearOperator->getOutputVariable();
  matrixCorrectionParameters->d_inputMatrix = linearOperator->getMatrix();
  matrixCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::DirichletMatrixCorrection(matrixCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createMassMatrixCorrection( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					     boost::shared_ptr<AMP::InputDatabase> input_db,
					     AMP::Operator::Operator::shared_ptr volumeOperator,
					     boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearOperator>(volumeOperator);
  boost::shared_ptr<AMP::Operator::DirichletMatrixCorrectionParameters> matrixCorrectionParameters (new AMP::Operator::DirichletMatrixCorrectionParameters( input_db ) );
  matrixCorrectionParameters->d_variable = linearOperator->getOutputVariable();
  matrixCorrectionParameters->d_inputMatrix = linearOperator->getMatrix();
  matrixCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::MassMatrixCorrection(matrixCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createRobinMatrixCorrection( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					      boost::shared_ptr<AMP::InputDatabase> input_db,
					      AMP::Operator::Operator::shared_ptr volumeOperator,
					      boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearOperator>(volumeOperator);
  boost::shared_ptr<AMP::Operator::RobinMatrixCorrectionParameters> matrixCorrectionParameters (new AMP::Operator::RobinMatrixCorrectionParameters( input_db ) );
  matrixCorrectionParameters->d_variable = linearOperator->getOutputVariable();
  matrixCorrectionParameters->d_inputMatrix = linearOperator->getMatrix();
  matrixCorrectionParameters->d_Mesh = meshAdapter;
  matrixCorrectionParameters->d_DofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, 1, 1, true);

  if(elementPhysicsModel.get()!=NULL )
    {
      
      boost::shared_ptr<AMP::Operator::RobinPhysicsModel> robinPhysicsModel;
      robinPhysicsModel = boost::dynamic_pointer_cast<AMP::Operator::RobinPhysicsModel>(elementPhysicsModel);
      matrixCorrectionParameters->d_robinPhysicsModel = robinPhysicsModel ;
    }
  
  retOperator.reset(new AMP::Operator::RobinMatrixCorrection(matrixCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createRobinVectorCorrection( AMP::Mesh::Mesh::shared_ptr meshAdapter,
					      boost::shared_ptr<AMP::InputDatabase> input_db,
					      AMP::Operator::Operator::shared_ptr volumeOperator,
					      boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::NeumannVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::NeumannVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_variable = volumeOperator->getOutputVariable();
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  if(elementPhysicsModel.get()!=NULL )
    {
      
      boost::shared_ptr<AMP::Operator::RobinPhysicsModel> robinPhysicsModel;
      robinPhysicsModel = boost::dynamic_pointer_cast<AMP::Operator::RobinPhysicsModel>(elementPhysicsModel);
      vectorCorrectionParameters->d_robinPhysicsModel = robinPhysicsModel ;
    }
  
  retOperator.reset(new AMP::Operator::RobinVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createNeumannVectorCorrection( AMP::Mesh::Mesh::shared_ptr meshAdapter,
						boost::shared_ptr<AMP::InputDatabase> input_db,
						AMP::Operator::Operator::shared_ptr volumeOperator,
						boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &elementPhysicsModel  )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::NeumannVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::NeumannVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_variable = volumeOperator->getOutputVariable();
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  if(elementPhysicsModel.get()!=NULL && input_db->isDatabase("RobinPhysicsModel"))
    {
      
      boost::shared_ptr<AMP::Operator::RobinPhysicsModel> robinPhysicsModel;
      robinPhysicsModel = boost::dynamic_pointer_cast<AMP::Operator::RobinPhysicsModel>(elementPhysicsModel);
      vectorCorrectionParameters->d_robinPhysicsModel = robinPhysicsModel ;
      
    }
  
  retOperator.reset(new AMP::Operator::NeumannVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createDirichletVectorCorrection(AMP::Mesh::Mesh::shared_ptr meshAdapter,
						 boost::shared_ptr<AMP::InputDatabase> input_db,
						 AMP::Operator::Operator::shared_ptr volumeOperator,
						 boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &
						 )
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::DirichletVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_variable = volumeOperator->getOutputVariable();
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::DirichletVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createDirichletVectorCorrection(AMP::Mesh::Mesh::shared_ptr meshAdapter,
						 boost::shared_ptr<AMP::InputDatabase> input_db,
						 boost::shared_ptr<AMP::Operator::ElementPhysicsModel> &)
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::DirichletVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::DirichletVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createPressureBoundaryVectorCorrection(AMP::Mesh::Mesh::shared_ptr meshAdapter,
							boost::shared_ptr<AMP::InputDatabase> input_db,
							AMP::Operator::Operator::shared_ptr volumeOperator,
							boost::shared_ptr<AMP::Operator::ElementPhysicsModel> & elementPhysicsModel)
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::PressureBoundaryVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::PressureBoundaryVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_variable = volumeOperator->getOutputVariable();
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::PressureBoundaryVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<BoundaryOperator>
OperatorBuilder::createPressureBoundaryVectorCorrection(AMP::Mesh::Mesh::shared_ptr meshAdapter,
							boost::shared_ptr<AMP::InputDatabase> input_db,
							boost::shared_ptr<AMP::Operator::ElementPhysicsModel> & elementPhysicsModel)
{
  boost::shared_ptr<BoundaryOperator> retOperator;
  boost::shared_ptr<AMP::Operator::PressureBoundaryVectorCorrectionParameters> vectorCorrectionParameters (new AMP::Operator::PressureBoundaryVectorCorrectionParameters( input_db ) );
  vectorCorrectionParameters->d_Mesh = meshAdapter;
  
  retOperator.reset(new AMP::Operator::PressureBoundaryVectorCorrection(vectorCorrectionParameters));
  
  return retOperator;
}

boost::shared_ptr<Operator>
OperatorBuilder::createOperator( AMP::Mesh::Mesh::shared_ptr meshAdapter1, AMP::Mesh::Mesh::shared_ptr meshAdapter2, AMP::AMP_MPI comm,
				boost::shared_ptr<AMP::Database> input_db)
{
  boost::shared_ptr<Operator> retOperator;
  
  std::string name = input_db->getString("name");
  if(name=="MapSurface")
    {
      boost::shared_ptr<AMP::Operator::MapOperatorParameters> mapOperatorParameters (new AMP::Operator::MapOperatorParameters( input_db ) );
      
      mapOperatorParameters->d_Mesh = meshAdapter1;
      mapOperatorParameters->d_MapMesh = meshAdapter2;
      mapOperatorParameters->d_MapComm = comm;
      retOperator.reset(new AMP::Operator::MapSurface(mapOperatorParameters));
    }
  return retOperator;
}


}
}

