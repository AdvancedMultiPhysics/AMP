
#include "RobinMatrixCorrection.h"
#include "RobinMatrixCorrectionParameters.h"
#include "utils/Utilities.h"
#include "utils/InputDatabase.h"

/* Libmesh files */

#include "enum_order.h"
#include "enum_fe_family.h"
#include "enum_quadrature_type.h"
#include "auto_ptr.h"
#include "string_to_enum.h"

#include "face_quad4.h"
#include "node.h"

#include <string>

namespace AMP {
namespace Operator {

RobinMatrixCorrection :: RobinMatrixCorrection(const boost::shared_ptr<RobinMatrixCorrectionParameters> & params)
  : BoundaryOperator (params)
{
  std::string feTypeOrderName = (params->d_db)->getStringWithDefault("FE_ORDER", "FIRST");
  std::string feFamilyName = (params->d_db)->getStringWithDefault("FE_FAMILY", "LAGRANGE");
  std::string qruleTypeName = (params->d_db)->getStringWithDefault("QRULE_TYPE", "QGAUSS");
  d_qruleOrderName = (params->d_db)->getStringWithDefault("QRULE_ORDER", "DEFAULT");
  
  d_feTypeOrder = Utility::string_to_enum<libMeshEnums::Order>(feTypeOrderName);
  d_feFamily = Utility::string_to_enum<libMeshEnums::FEFamily>(feFamilyName);
  d_qruleType = Utility::string_to_enum<libMeshEnums::QuadratureType>(qruleTypeName);
  
  d_variable = params->d_variable;
  
  d_dofManager = params->d_DofMap;

  d_NeumannParams.reset(new AMP::Operator::NeumannVectorCorrectionParameters( params->d_db ));
  d_NeumannParams->d_variable = params->d_variable;
  d_NeumannParams->d_Mesh = params->d_Mesh;
  d_NeumannParams->d_variableFlux = params->d_variableFlux;
  d_NeumannCorrection.reset(new NeumannVectorCorrection (d_NeumannParams));
  
  reset(params);
}
  
void RobinMatrixCorrection :: reset(const boost::shared_ptr<OperatorParameters>& params)
{

  boost::shared_ptr<RobinMatrixCorrectionParameters> myparams =
    boost::dynamic_pointer_cast<RobinMatrixCorrectionParameters>(params);
  
  AMP_INSIST( ((myparams.get()) != NULL), "NULL parameters" );
  AMP_INSIST( (((myparams->d_db).get()) != NULL), "NULL database" );
  
  AMP_INSIST( (((myparams->d_db).get()) != NULL), "NULL database" );
  bool skipParams = (myparams->d_db)->getBoolWithDefault("skip_params", true);

  bool d_isFluxGaussPtVector = (myparams->d_db)->getBoolWithDefault("IsFluxGaussPtVector", true);
  
  if(!skipParams)
    {
      AMP_INSIST( (myparams->d_db)->keyExists("alpha"), "Missing key: prefactor alpha" );
      d_alpha    = (myparams->d_db)->getDouble("alpha");
      AMP_INSIST( d_alpha != 0.0, "prefactor alpha must be != 0.0" );
      
      AMP_INSIST( (myparams->d_db)->keyExists("beta"), "Missing key: prefactor beta" );
      d_beta = (myparams->d_db)->getDouble("beta");
      
      AMP_INSIST( (myparams->d_db)->keyExists("gamma"), "Missing key: total prefactor gamma" );
      d_gamma = (myparams->d_db)->getDouble("gamma");
      
      AMP_INSIST( (myparams->d_db)->keyExists("fConductivity"), "Missing key: effective convective coefficient" );
      d_hef   = (myparams->d_db)->getDouble("fConductivity");
      
      AMP_INSIST( (params->d_db)->keyExists("number_of_ids"), "Key ''number_of_ids'' is missing!" );
      int numIds = (params->d_db)->getInteger("number_of_ids");
      
      d_boundaryIds.resize(numIds);
      d_dofIds.resize(numIds);
      d_robinValues.resize(numIds);
      
      char key[100];
      for(int j = 0; j < numIds; j++)
    {
      
      sprintf(key, "id_%d", j);
      AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
      d_boundaryIds[j] = (myparams->d_db)->getInteger(key);
      
      sprintf(key, "number_of_dofs_%d", j);
      AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
      int numDofIds = (myparams->d_db)->getInteger(key);
      
      d_dofIds[j].resize(numDofIds);
      d_robinValues[j].resize(numDofIds);
      for(int i = 0; i < numDofIds; i++)
        {
          sprintf(key, "dof_%d_%d", j, i);
          AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
          d_dofIds[j][i] = (myparams->d_db)->getInteger(key);
          
          sprintf(key, "value_%d_%d", j, i);
          AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
          d_robinValues[j][i] = (myparams->d_db)->getDouble(key);
        }
    }
  }
  
  d_robinPhysicsModel = myparams->d_robinPhysicsModel;
  
  (d_NeumannParams->d_db)->putBool("constant_flux",myparams->d_db->getBoolWithDefault("constant_flux",true));
  d_NeumannParams->d_variableFlux = myparams->d_variableFlux;
  d_NeumannParams->d_robinPhysicsModel = myparams->d_robinPhysicsModel ;
  (d_NeumannParams->d_db)->putDouble("gamma",d_gamma);
  d_NeumannCorrection->reset(d_NeumannParams);
  
  bool skipMatrixCorrection = (myparams->d_db)->getBoolWithDefault("skip_matrix_correction", false);
  if(!skipMatrixCorrection)
  {
    // Create the libmesh elements
    AMP::Mesh::MeshIterator iterator;
    for(unsigned int j = 0; j < d_boundaryIds.size() ; j++) {
       AMP::Mesh::MeshIterator iterator2 = d_Mesh->getBoundaryIDIterator( AMP::Mesh::Face, d_boundaryIds[j], 0 );
       iterator = AMP::Mesh::Mesh::getIterator( AMP::Mesh::Union, iterator, iterator2 );
    }
    libmeshElements.reinit( iterator );

    AMP::LinearAlgebra::Matrix::shared_ptr inputMatrix = myparams->d_inputMatrix;
    AMP_INSIST( ((inputMatrix.get()) != NULL), "NULL matrix" );

/*
    std::vector<std::string> variableNames;
    if(d_robinPhysicsModel.get() != NULL)
    {
      variableNames = d_robinPhysicsModel->getVariableName();
    }
*/
    unsigned int numIds = d_boundaryIds.size();
    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> elementInputVec = myparams->d_elementInputVec;

    std::vector<size_t> gpDofs, dofsElementVec;
    std::vector<size_t> dofIndices; 
    AMP::Discretization::DOFManager::shared_ptr gpDOFManager; 
    if(d_isFluxGaussPtVector && myparams->d_variableFlux.get()!=NULL ){
      gpDOFManager = (myparams->d_variableFlux)->getDOFManager();
    }

    for(unsigned int nid = 0; nid < numIds; nid++)
    {
      AMP::Mesh::MeshIterator bnd1     = d_Mesh->getBoundaryIDIterator( AMP::Mesh::Face, d_boundaryIds[nid], 0 );
      AMP::Mesh::MeshIterator end_bnd1 = bnd1.end();
      for( ; bnd1 != end_bnd1; ++bnd1)
      {

        boost::shared_ptr < ::FEType > d_feType ( new ::FEType(d_feTypeOrder, d_feFamily) );
        boost::shared_ptr < ::FEBase > d_fe( (::FEBase::build(2, (*d_feType))).release() );

        if(d_qruleOrderName == "DEFAULT") {
          d_qruleOrder = d_feType->default_quadrature_order();
        } else {
          d_qruleOrder = Utility::string_to_enum<libMeshEnums::Order>(d_qruleOrderName);
        }
        boost::shared_ptr < ::QBase > d_qrule( (::QBase::build(d_qruleType, 2, d_qruleOrder)).release() );

        // Get the nodes for the element and their global ids
        std::vector<AMP::Mesh::MeshElement> currNodes = bnd1->getElements(AMP::Mesh::Vertex);
        dofIndices.resize(currNodes.size());
        std::vector<AMP::Mesh::MeshElementID> globalIDs(currNodes.size()); 
        for(size_t j=0; j<currNodes.size(); j++)
            globalIDs[j] = currNodes[j].globalID();

        // Get the libmesh element
        d_currElemPtr = libmeshElements.getElement( bnd1->globalID() );

        // Get the DOF indicies for the matrix
        d_dofManager->getDOFs(globalIDs, dofIndices);

        if(d_isFluxGaussPtVector && myparams->d_variableFlux.get()!=NULL ){
          gpDOFManager->getDOFs (bnd1->globalID(), gpDofs);
        }

        d_fe->attach_quadrature_rule( d_qrule.get() );

        d_phi = &(d_fe->get_phi());
        d_JxW = &(d_fe->get_JxW());

        d_fe->reinit ( d_currElemPtr );

        const std::vector<Real> & JxW = (*d_JxW);
        const std::vector<std::vector<Real> > & phi = (*d_phi);
        unsigned int numGaussPts = d_qrule->n_points(); 

        std::vector<std::vector<double> > inputArgs(elementInputVec.size(),std::vector<double>(currNodes.size()));
        std::vector<std::vector<double> > inputArgsAtGpts(elementInputVec.size(),std::vector<double>(numGaussPts));
        std::vector<double> beta(numGaussPts,d_beta);
        std::vector<double> gamma(numGaussPts,d_gamma);
        if(d_robinPhysicsModel.get() != NULL) 
        {
          unsigned int startIdx = 0;
          if(d_isFluxGaussPtVector){
            (myparams->d_variableFlux)->getValuesByGlobalID( gpDofs.size(), &gpDofs[0], &inputArgsAtGpts[0][0] );
            startIdx = 1;
          }

          for(unsigned int m = startIdx ; m < elementInputVec.size(); m++){
            elementInputVec[m]->getDOFManager()->getDOFs( globalIDs, dofsElementVec );
            AMP_ASSERT(dofsElementVec.size()==dofIndices.size());
            elementInputVec[m]->getValuesByGlobalID( dofsElementVec.size(), &dofsElementVec[0], &inputArgs[m][0] );
            for (size_t qp = 0; qp < currNodes.size(); qp++){ 
              for (size_t n = 0; n < currNodes.size(); n++) {
                inputArgsAtGpts[m][qp] += phi[n][qp] * inputArgs[m][n];
              }
            }
          }

          d_robinPhysicsModel->getConductance(beta, gamma, inputArgsAtGpts);
        }

        double temp;
        for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++)
        {
          for (unsigned int j=0; j < currNodes.size(); j++)
          {
            for (unsigned int i=0; i < currNodes.size(); i++)
            {
              temp =  beta[qp] * ( JxW[qp]*phi[j][qp]*phi[i][qp] ) ;
              inputMatrix->addValueByGlobalID ( dofIndices[j], dofIndices[i], temp );
            }//end for i
          }//end for j
        }//end for qp

      }//end for bnd

    }// end for nid

    inputMatrix->makeConsistent();

  }//skip matrix

}


}
}
