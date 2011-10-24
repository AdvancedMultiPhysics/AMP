
#include "MassLinearFEOperator.h"
#include "utils/Utilities.h"
#include "ampmesh/MeshVariable.h"

namespace AMP {
namespace Operator {

  MassLinearFEOperator :: MassLinearFEOperator (const boost::shared_ptr<MassLinearFEOperatorParameters> & params)
    : LinearFEOperator (params) 
  {
    AMP_INSIST( ((params.get()) != NULL), "NULL parameter" );

    d_massLinElem = boost::dynamic_pointer_cast<MassLinearElement>(d_elemOp);

    AMP_INSIST( ((d_massLinElem.get()) != NULL), "d_elemOp is not of type MassLinearElement" );

    d_densityModel = params->d_densityModel;

    d_useConstantTemperature = params->d_db->keyExists("FixedTemperature");

    d_useConstantConcentration = params->d_db->keyExists("FixedConcentration");

    d_useConstantBurnup = params->d_db->keyExists("FixedBurnup");

    d_constantTemperatureValue = params->d_db->getDoubleWithDefault(
        "FixedTemperature", 273.0);

    d_constantConcentrationValue = params->d_db->getDoubleWithDefault(
        "FixedConcentration", 0.0);

    d_constantBurnupValue = params->d_db->getDoubleWithDefault(
        "FixedBurnup", 0.0);
    //d_inpVariable.reset(new AMP::Mesh::NodalScalarVariable("inpVar"));
    //d_outVariable.reset(new AMP::Mesh::NodalScalarVariable("outVar"));
    std::string inpVar = params->d_db->getString("InputVariable");
    d_inpVariable.reset(new AMP::Mesh::NodalScalarVariable(inpVar, d_MeshAdapter));    

    std::string outVar = params->d_db->getString("OutputVariable");
    d_outVariable.reset(new AMP::Mesh::NodalScalarVariable(outVar, d_MeshAdapter));

    reset(params);
  }

  void MassLinearFEOperator :: preAssembly(const boost::shared_ptr<AMP::Operator::OperatorParameters>&) 
  {
    d_matrix->zero();

    d_densityModel->preLinearAssembly();
  }

  void MassLinearFEOperator :: postAssembly()
  {
    d_densityModel->postLinearAssembly();

    d_matrix->makeConsistent();
  }

  void MassLinearFEOperator :: preElementOperation( const AMP::Mesh::MeshManager::Adapter::Element & elem,
      const std::vector<AMP::Mesh::DOFMap::shared_ptr> & dof_maps )
  {
    (dof_maps[0])->getDOFs(elem, d_dofIndices);
    unsigned int num_local_dofs = d_dofIndices.size();

    d_elementMassMatrix.resize(num_local_dofs);
    for (unsigned int r = 0; r < num_local_dofs; r++) {
      d_elementMassMatrix[r].resize(num_local_dofs);
      for (unsigned int c = 0; c < num_local_dofs; c++) {
        d_elementMassMatrix[r][c] = 0;
      }
    }

    std::vector<double> localTemperature(num_local_dofs);
    std::vector<double> localConcentration(num_local_dofs);
    std::vector<double> localBurnup(num_local_dofs);

    if (d_useConstantTemperature) {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localTemperature[r] = d_constantTemperatureValue;
      }
    } else {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localTemperature[r] = d_temperature->getValueByGlobalID(
            d_dofIndices[r]);
      }
    }

    if (d_useConstantConcentration) {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localConcentration[r] = d_constantConcentrationValue;
      }
    } else {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localConcentration[r] = d_concentration->getValueByGlobalID(
            d_dofIndices[r]);
      }
    }

    if (d_useConstantBurnup) {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localBurnup[r] = d_constantBurnupValue;
      }
    } else {
      for (unsigned int r = 0; r < num_local_dofs; r++) {
        localBurnup[r] = d_burnup->getValueByGlobalID(
            d_dofIndices[r]);
      }
    }

    const ::Elem* elemPtr = &(elem.getElem());

    d_massLinElem->initializeForCurrentElement(elemPtr, d_densityModel);

    d_massLinElem->setElementMassMatrix(d_elementMassMatrix);

    d_massLinElem->setElementVectors(localTemperature, localConcentration, localBurnup);
  }

  void MassLinearFEOperator :: postElementOperation()
  {

    unsigned int num_local_dofs = d_dofIndices.size();

    for(unsigned int r = 0; r < num_local_dofs; r++) {
      for(unsigned int c = 0; c < num_local_dofs; c++) {
        d_matrix->addValueByGlobalID( d_dofIndices[r], d_dofIndices[c], 
            d_elementMassMatrix[r][c] );
      }
    }

  }

}
}//end namespace




