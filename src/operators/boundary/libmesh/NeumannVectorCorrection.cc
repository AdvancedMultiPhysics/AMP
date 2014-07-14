
#include "NeumannVectorCorrection.h"
#include "NeumannVectorCorrectionParameters.h"
#include "utils/Utilities.h"
#include "utils/InputDatabase.h"

/* Libmesh files */

#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/auto_ptr.h"
#include "libmesh/string_to_enum.h"

#include "libmesh/face_quad4.h"
#include "libmesh/node.h"

#include <string>

namespace AMP {
namespace Operator {


// Constructor
NeumannVectorCorrection::NeumannVectorCorrection(const boost::shared_ptr<NeumannVectorCorrectionParameters> & params) :
    BoundaryOperator (params)
{
    d_params = params;

    std::string feTypeOrderName = (params->d_db)->getStringWithDefault("FE_ORDER", "FIRST");
    std::string feFamilyName = (params->d_db)->getStringWithDefault("FE_FAMILY", "LAGRANGE");
    std::string qruleTypeName = (params->d_db)->getStringWithDefault("QRULE_TYPE", "QGAUSS");
    d_qruleOrderName = (params->d_db)->getStringWithDefault("QRULE_ORDER", "DEFAULT");

    d_feTypeOrder = Utility::string_to_enum<libMeshEnums::Order>(feTypeOrderName);
    d_feFamily = Utility::string_to_enum<libMeshEnums::FEFamily>(feFamilyName);
    d_qruleType = Utility::string_to_enum<libMeshEnums::QuadratureType>(qruleTypeName);

    d_variable = params->d_variable;

    reset(params);
}


void NeumannVectorCorrection :: reset(const boost::shared_ptr<OperatorParameters>& params)
{
    boost::shared_ptr<NeumannVectorCorrectionParameters> myparams = 
        boost::dynamic_pointer_cast<NeumannVectorCorrectionParameters>(params);

    AMP_INSIST( ((myparams.get()) != NULL), "NULL parameters" );
    AMP_INSIST( (((myparams->d_db).get()) != NULL), "NULL database" );

    AMP_INSIST( (myparams->d_db)->keyExists("number_of_ids"), "Key ''number_of_ids'' is missing!" );
    d_numBndIds = (myparams->d_db)->getInteger("number_of_ids");

    d_isConstantFlux = (myparams->d_db)->getBoolWithDefault("constant_flux", true);
    d_isFluxGaussPtVector = (myparams->d_db)->getBoolWithDefault("IsFluxGaussPtVector", true);

    d_boundaryIds.resize(d_numBndIds);
    d_dofIds.resize(d_numBndIds);
    d_neumannValues.resize(d_numBndIds);
    d_IsCoupledBoundary.resize(d_numBndIds);
    d_numDofIds.resize(d_numBndIds);

    char key[100];
    for(int j = 0; j < d_numBndIds; j++) {
        sprintf(key, "id_%d", j);
        AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
        d_boundaryIds[j] = (myparams->d_db)->getInteger(key);

        sprintf(key, "number_of_dofs_%d", j);
        AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
        d_numDofIds[j] = (myparams->d_db)->getInteger(key);

        sprintf(key, "IsCoupledBoundary_%d", j);
        d_IsCoupledBoundary[j] = (params->d_db)->getBoolWithDefault(key, false);

        d_dofIds[j].resize(d_numDofIds[j]);
        d_neumannValues[j].resize(d_numDofIds[j]);
        for(int i = 0; i < d_numDofIds[j]; i++) {
            sprintf(key, "dof_%d_%d", j, i);
            AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
            d_dofIds[j][i] = (myparams->d_db)->getInteger(key);

            if(d_isConstantFlux){
                sprintf(key, "value_%d_%d", j, i);
                AMP_INSIST( (myparams->d_db)->keyExists(key), "Key is missing!" );
                d_neumannValues[j][i] = (myparams->d_db)->getDouble(key);
            }else{
                d_variableFlux = myparams->d_variableFlux;      
            }
        }//end for i
      }//end for j

    if(myparams->d_robinPhysicsModel) {
        d_robinPhysicsModel = myparams->d_robinPhysicsModel;
    }

    // Create the libmesh elements
    AMP::Mesh::MeshIterator iterator;
    for(unsigned int j = 0; j < d_boundaryIds.size() ; j++) {
        AMP::Mesh::MeshIterator iterator2 = d_Mesh->getBoundaryIDIterator( AMP::Mesh::Face, d_boundaryIds[j], 0 );
        iterator = AMP::Mesh::Mesh::getIterator( AMP::Mesh::Union, iterator, iterator2 );
    }
    libmeshElements.reinit( iterator );
}


void NeumannVectorCorrection :: addRHScorrection(AMP::LinearAlgebra::Vector::shared_ptr rhsCorrection)
{

    AMP::LinearAlgebra::Vector::shared_ptr myRhs = subsetInputVector( rhsCorrection );

    double gammaValue = (d_params->d_db)->getDoubleWithDefault("gamma",1.0);

    AMP::LinearAlgebra::Vector::shared_ptr rInternal = myRhs->cloneVector();
    AMP::Discretization::DOFManager::shared_ptr dofManager = rInternal->getDOFManager();
    rInternal->zero();

    unsigned int numBndIds = d_boundaryIds.size();
    std::vector<size_t> dofs;
    std::vector<std::vector<size_t> > dofIndices;
    std::vector<size_t> fluxDofs;

    for(unsigned int j = 0; j < numBndIds ; j++)
    {
      if(!d_IsCoupledBoundary[j])
      {
        unsigned int numDofIds = d_dofIds[j].size();

        if(!d_isConstantFlux)
        {
          d_variableFlux->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
        }

        for(unsigned int k = 0; k < numDofIds; k++)
        {

          AMP::Mesh::MeshIterator bnd     = d_Mesh->getBoundaryIDIterator( AMP::Mesh::Face, d_boundaryIds[j], 0 );
          AMP::Mesh::MeshIterator end_bnd = bnd.end();

          int count =0;
          for( ; bnd != end_bnd; ++bnd)
          {
            count++;

            const unsigned int dimension = 2;

            boost::shared_ptr < ::FEType > d_feType ( new ::FEType(d_feTypeOrder, d_feFamily) );
            boost::shared_ptr < ::FEBase > d_fe ( (::FEBase::build(dimension, (*d_feType))).release() );

            if(d_qruleOrderName == "DEFAULT") {
              d_qruleOrder = d_feType->default_quadrature_order();
            } else {
              d_qruleOrder = Utility::string_to_enum<libMeshEnums::Order>(d_qruleOrderName);
            }
            boost::shared_ptr < ::QBase > d_qrule( (::QBase::build(d_qruleType, dimension, d_qruleOrder)).release() );


            d_currNodes = bnd->getElements(AMP::Mesh::Vertex);
            unsigned int numNodesInCurrElem = d_currNodes.size();

            dofIndices.resize(numNodesInCurrElem);
            for(unsigned int i = 0; i < numNodesInCurrElem ; i++) {
              dofManager->getDOFs(d_currNodes[i].globalID(), dofIndices[i]);
            }

            AMP::Discretization::DOFManager::shared_ptr fluxDOFManager; 
            if( !d_isConstantFlux && d_isFluxGaussPtVector){
              fluxDOFManager = d_variableFlux->getDOFManager();
              fluxDOFManager->getDOFs (bnd->globalID(), fluxDofs);
            }

            // Get the libmesh element
            ::Elem* currElemPtr = libmeshElements.getElement( bnd->globalID() );

            d_fe->attach_quadrature_rule( d_qrule.get() );

            d_fe->reinit ( currElemPtr );

            const std::vector<std::vector<Real> > phi = d_fe->get_phi();
            const std::vector<Real> djxw = d_fe->get_JxW();

            std::vector<std::vector<double> > temp(1) ;
            std::vector<double> gamma(d_qrule->n_points(), gammaValue);

            dofs.resize(numNodesInCurrElem);
            for (size_t n = 0; n < dofIndices.size() ; n++)
              dofs[n] = dofIndices[n][d_dofIds[j][k]];

            for (size_t qp = 0; qp < d_qrule->n_points(); qp++) {
              if(d_isConstantFlux)
              {
                temp[0].push_back(d_neumannValues[j][k]);
              }else{
                if(d_isFluxGaussPtVector)
                {
                  temp[0].push_back(d_variableFlux->getValueByGlobalID(fluxDofs[qp]));
                }else{
                  Real Tqp = 0.0;
                  for (size_t n = 0; n < dofIndices.size() ; n++) {
                    Tqp += phi[n][qp] * d_variableFlux->getValueByGlobalID(dofs[n]);
                  }
                  temp[0].push_back(Tqp);
                }
              }
            }

            if(d_robinPhysicsModel)
            {
              d_robinPhysicsModel->getConductance(gamma, gamma, temp); 
            }

            std::vector<double> flux( dofIndices.size(), 0.0);

            for(unsigned int i = 0; i < dofIndices.size() ; i++)    // Loop over nodes
            {
              for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) 
              {
                flux[i] +=  (gamma[qp])*djxw[qp]*phi[i][qp]*temp[0][qp];
              }//end for qp
            }//end for i

            rInternal->addValuesByGlobalID((int)dofs.size() , (size_t *)&(dofs[0]), &(flux[0]));

          }//end for bnd

        }//end for k
      }//coupled
    }//end for j

    rInternal->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_ADD );
    myRhs->add(myRhs, rInternal);

}


void NeumannVectorCorrection :: apply(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u, AMP::LinearAlgebra::Vector::shared_ptr r, const double a , const double b )
{
    (void) f; (void) u; (void) r; (void) a; (void) b; 
    //Do Nothing
}

boost::shared_ptr<OperatorParameters> NeumannVectorCorrection :: getJacobianParameters(
    const boost::shared_ptr<AMP::LinearAlgebra::Vector>& ) 
{
    boost::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));

    tmp_db->putString("FE_ORDER","FIRST");
    tmp_db->putString("FE_FAMILY","LAGRANGE");
    tmp_db->putString("QRULE_TYPE","QGAUSS");
    tmp_db->putInteger("DIMENSION",2);
    tmp_db->putString("QRULE_ORDER","DEFAULT");
    tmp_db->putInteger("number_of_ids",d_numBndIds);
    tmp_db->putBool("constant_flux", d_isConstantFlux);

    char key[100];
    for(int j = 0; j < d_numBndIds; j++) {
        sprintf(key, "id_%d", j);
        tmp_db->putInteger(key,d_boundaryIds[j]);
        sprintf(key, "number_of_dofs_%d", j);
        tmp_db->putInteger(key,d_numDofIds[j]);
        sprintf(key, "IsCoupledBoundary_%d", j);
        tmp_db->putBool(key, d_IsCoupledBoundary[j]);

        for(int i = 0; i < d_numDofIds[j]; i++) {
            sprintf(key, "dof_%d_%d", j, i);
            tmp_db->putInteger(key,d_dofIds[j][i]);
            if ( d_isConstantFlux ) {
                sprintf(key, "value_%d_%d", j, i);
                tmp_db->putInteger(key,d_neumannValues[j][i]);
            } else {
                //d_variableFlux ??
            }
        }
    }

    tmp_db->putBool("skip_params", true);

    boost::shared_ptr<NeumannVectorCorrectionParameters> outParams(new NeumannVectorCorrectionParameters(tmp_db));

    return outParams;
}


void NeumannVectorCorrection :: setFrozenVector ( AMP::LinearAlgebra::Vector::shared_ptr f ) 
{
    AMP::LinearAlgebra::Vector::shared_ptr f2 = f;
    if ( d_Mesh.get() != NULL )
        f2 = f->select( AMP::LinearAlgebra::VS_Mesh(d_Mesh), f->getVariable()->getName() );
    if ( f2==NULL )
        return;
    if ( d_Frozen==NULL )
        d_Frozen = AMP::LinearAlgebra::MultiVector::create( "frozenMultiVec", d_Mesh->getComm() );
    d_Frozen->castTo<AMP::LinearAlgebra::MultiVector>().addVector ( f2 );
}


void NeumannVectorCorrection :: setVariableFlux(const AMP::LinearAlgebra::Vector::shared_ptr &flux) {
    if(d_Mesh.get() != NULL) {
        AMP::LinearAlgebra::VS_Mesh meshSelector(d_Mesh);
        AMP::LinearAlgebra::Vector::shared_ptr meshSubsetVec = flux->select(meshSelector, d_variable->getName());
        d_variableFlux = meshSubsetVec->subsetVectorForVariable( d_variable );
    } else {
        d_variableFlux = flux->subsetVectorForVariable( d_variable );
    }
}


}
}

