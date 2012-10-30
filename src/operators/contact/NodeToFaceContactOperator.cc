
#include "operators/contact/NodeToFaceContactOperator.h"
#include "ampmesh/dendro/DendroSearch.h"
#include "ampmesh/MeshID.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace AMP {
  namespace Operator {

    void NodeToFaceContactOperator::initialize() {
      AMP_ASSERT( d_ActiveSet.empty() );
      AMP_ASSERT( d_InactiveSet.empty() );

      /** get all slave boundary vertices and tag them as inactive */
      AMP::Mesh::Mesh::shared_ptr slaveMesh = d_Mesh->Subset(d_SlaveMeshID);
      if (slaveMesh.get() != NULL) {
        AMP::Mesh::MeshIterator slaveMeshIterator = slaveMesh->getBoundaryIDIterator(AMP::Mesh::Vertex, d_SlaveBoundaryID);
        AMP::Mesh::MeshIterator slaveMeshIterator_begin = slaveMeshIterator.begin(), 
          slaveMeshIterator_end = slaveMeshIterator.end();
        d_InactiveSet.resize(slaveMeshIterator.size());
        std::vector<AMP::Mesh::MeshElementID>::iterator slaveVerticesGlobalIDsIterator = d_InactiveSet.begin();
        for (slaveMeshIterator = slaveMeshIterator_begin; slaveMeshIterator != slaveMeshIterator_end; ++slaveMeshIterator) {
          *slaveVerticesGlobalIDsIterator = slaveMeshIterator->globalID();
          ++slaveVerticesGlobalIDsIterator;
        } // end loop over the slave vertices on boundary
        AMP_ASSERT( slaveVerticesGlobalIDsIterator == d_InactiveSet.end() );
      } //end if
    }

    size_t NodeToFaceContactOperator::updateActiveSet() {
      size_t nInactiveSlaveVerticesActivated = 0;
      std::vector<AMP::Mesh::MeshElementID>::iterator inactiveSlaveVerticesGlobalIDsIterator,
        inactiveSlaveVerticesGlobalIDsIterator_begin = d_InactiveSet.begin(),
        inactiveSlaveVerticesGlobalIDsIterator_end = d_InactiveSet.end();
      size_t nInactiveSlaveVertices = d_InactiveSet.size();
      std::vector<double> inactiveSlaveVertexCoord(3);
      AMP::Mesh::MeshElement inactiveSlaveVertex;
      std::vector<double> inactiveSlaveVerticesCoord(3*nInactiveSlaveVertices); 
      std::vector<double>::iterator inactiveSlaveVerticesCoordIterator = inactiveSlaveVerticesCoord.begin();
      for (inactiveSlaveVerticesGlobalIDsIterator = inactiveSlaveVerticesGlobalIDsIterator_begin; 
        inactiveSlaveVerticesGlobalIDsIterator != inactiveSlaveVerticesGlobalIDsIterator_end; 
        ++inactiveSlaveVerticesGlobalIDsIterator) 
      {
        inactiveSlaveVertex = d_Mesh->getElement(*inactiveSlaveVerticesGlobalIDsIterator); 
        inactiveSlaveVertexCoord = inactiveSlaveVertex.coord();
        std::copy(inactiveSlaveVertexCoord.begin(), inactiveSlaveVertexCoord.end(), inactiveSlaveVerticesCoordIterator);
        for (size_t i = 0; i < 3; ++i) { ++inactiveSlaveVerticesCoordIterator; }
      } // end for
      AMP_ASSERT( inactiveSlaveVerticesCoordIterator == inactiveSlaveVerticesCoord.end() );

      size_t nActiveSlaveVerticesDeactivated = 0;

      return nInactiveSlaveVerticesActivated + nActiveSlaveVerticesDeactivated;
    }

    void NodeToFaceContactOperator::reset(const boost::shared_ptr<OperatorParameters> & params) {

      AMP_INSIST( (params != NULL), "NULL parameter" );
      AMP_INSIST( ((params->d_db) != NULL), "NULL database" );

      AMP::Mesh::Mesh::shared_ptr mesh = params->d_Mesh;
      //  AMP::AMP_MPI comm = mesh->getComm(); 
      AMP::AMP_MPI comm = d_GlobalComm;

      /** get the boundary slave vertices coordinates and global IDs */
      size_t nSlaveVertices = 0;
      std::vector<double> tmpSlaveVerticesCoord;
      std::vector<AMP::Mesh::MeshElementID> tmpSlaveVerticesGlobalIDs;
      AMP::Mesh::Mesh::shared_ptr slaveMesh = mesh->Subset(d_SlaveMeshID);
      if (slaveMesh != NULL) {
        AMP::Mesh::MeshIterator slaveMeshIterator = slaveMesh->getBoundaryIDIterator(AMP::Mesh::Vertex, d_SlaveBoundaryID);
        AMP::Mesh::MeshIterator slaveMeshIterator_begin = slaveMeshIterator.begin(), 
          slaveMeshIterator_end = slaveMeshIterator.end();
        nSlaveVertices = slaveMeshIterator.size();
        tmpSlaveVerticesCoord.resize(3*nSlaveVertices);
        tmpSlaveVerticesGlobalIDs.resize(nSlaveVertices);
        std::vector<AMP::Mesh::MeshElementID>::iterator tmpSlaveVerticesGlobalIDsIterator = tmpSlaveVerticesGlobalIDs.begin();
        std::vector<double> tmpCoord(3);
        std::vector<double>::iterator tmpSlaveVerticesCoordIterator = tmpSlaveVerticesCoord.begin();
        for (slaveMeshIterator = slaveMeshIterator_begin; slaveMeshIterator != slaveMeshIterator_end; ++slaveMeshIterator) {
          *tmpSlaveVerticesGlobalIDsIterator = slaveMeshIterator->globalID();
          ++tmpSlaveVerticesGlobalIDsIterator;
          tmpCoord = slaveMeshIterator->coord();
          AMP_ASSERT( tmpCoord.size() == 3 );
          std::copy(tmpCoord.begin(), tmpCoord.end(), tmpSlaveVerticesCoordIterator);
          for (size_t i = 0; i < 3; ++i) { ++tmpSlaveVerticesCoordIterator; }
        } // end loop over the slave vertices on boundary
        AMP_ASSERT( tmpSlaveVerticesGlobalIDsIterator == tmpSlaveVerticesGlobalIDs.end() );
        AMP_ASSERT( tmpSlaveVerticesCoordIterator == tmpSlaveVerticesCoord.end() );
      } //end if

      /** do a dendro search for the boundary slave vertices on the master mesh */
      AMP::Mesh::Mesh::shared_ptr masterMesh = mesh->Subset(d_MasterMeshID);
      AMP::Mesh::DendroSearch dendroSearchOnMaster(masterMesh);
      dendroSearchOnMaster.setTolerance(1.0e-10);
      dendroSearchOnMaster.search(comm, tmpSlaveVerticesCoord);

      std::vector<AMP::Mesh::MeshElementID> tmpMasterVerticesGlobalIDs;
      std::vector<double> tmpSlaveVerticesShift, tmpSlaveVerticesLocalCoordOnFace;
      std::vector<int> flags;

      dendroSearchOnMaster.projectOnBoundaryID(comm, d_MasterBoundaryID,
          tmpMasterVerticesGlobalIDs, tmpSlaveVerticesShift, tmpSlaveVerticesLocalCoordOnFace, flags);

      AMP_ASSERT( nSlaveVertices == tmpMasterVerticesGlobalIDs.size() / 4 );
      AMP_ASSERT( nSlaveVertices == tmpSlaveVerticesShift.size() / 3 );
      AMP_ASSERT( nSlaveVertices == tmpSlaveVerticesLocalCoordOnFace.size() / 2 );
      AMP_ASSERT( nSlaveVertices == flags.size() );

      /** build the constraints */
      const unsigned int nConstraints = std::count(flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundOnBoundary);

      unsigned int localPtsNotFound = std::count(flags.begin(), flags.end(), AMP::Mesh::DendroSearch::NotFound);
      unsigned int localPtsFoundNotOnBoundary = std::count(flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundNotOnBoundary);
      unsigned int localPtsFoundOnBoundary = std::count(flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundOnBoundary);
      unsigned int globalPtsNotFound = comm.sumReduce(localPtsNotFound);
      unsigned int globalPtsFoundNotOnBoundary = comm.sumReduce(localPtsFoundNotOnBoundary);
      unsigned int globalPtsFoundOnBoundary = comm.sumReduce(localPtsFoundOnBoundary);
      d_fout<<"Global number of points not found is "<<globalPtsNotFound<<" (local was "<<localPtsNotFound<<")"<<std::endl;
      d_fout<<"Global number of points found not on boundary is "<<globalPtsFoundNotOnBoundary<<" (local was "<<localPtsFoundNotOnBoundary<<")"<<std::endl;
      d_fout<<"Global number of points found on boundary is "<<globalPtsFoundOnBoundary<<" (local was "<<localPtsFoundOnBoundary<<")"<<std::endl;
      d_fout<<"Total number of points is "<<globalPtsNotFound+globalPtsFoundNotOnBoundary+globalPtsFoundOnBoundary<<std::endl;

      AMP_ASSERT( std::count(flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundNotOnBoundary) == 0 ); // DendroSearch::FoundNotOnBoundary is not acceptable

      d_SlaveVerticesGlobalIDs.resize(nConstraints);
      std::fill(d_SlaveVerticesGlobalIDs.begin(), d_SlaveVerticesGlobalIDs.end(), AMP::Mesh::MeshElementID());
      std::vector<AMP::Mesh::MeshElementID> masterVerticesGlobalIDs(4*nConstraints);
      std::fill(masterVerticesGlobalIDs.begin(), masterVerticesGlobalIDs.end(), AMP::Mesh::MeshElementID());
      d_MasterVerticesOwnerRanks.resize(4*nConstraints);
      std::fill(d_MasterVerticesOwnerRanks.begin(), d_MasterVerticesOwnerRanks.end(), comm.getSize());
      d_MasterShapeFunctionsValues.resize(4*nConstraints);
      std::fill(d_MasterShapeFunctionsValues.begin(), d_MasterShapeFunctionsValues.end(), 0.0);
      d_SlaveVerticesShift.resize(3*nConstraints);
      std::fill(d_SlaveVerticesShift.begin(), d_SlaveVerticesShift.end(), 0.0);

      std::vector<AMP::Mesh::MeshElementID>::const_iterator tmpSlaveVerticesGlobalIDsConstIterator = tmpSlaveVerticesGlobalIDs.begin();
      std::vector<AMP::Mesh::MeshElementID>::const_iterator tmpMasterVerticesGlobalIDsConstIterator = tmpMasterVerticesGlobalIDs.begin();
      double const * tmpSlaveVerticesLocalCoordOnFacePointerToConst = &(tmpSlaveVerticesLocalCoordOnFace[0]);
      std::vector<double>::const_iterator tmpSlaveVerticesShiftConstIterator = tmpSlaveVerticesShift.begin();
      double * masterShapeFunctionsValuesPointer = &(d_MasterShapeFunctionsValues[0]);
      std::vector<double>::iterator slaveVerticesShiftIterator = d_SlaveVerticesShift.begin();
      std::vector<AMP::Mesh::MeshElementID>::iterator slaveVerticesGlobalIDsIterator = d_SlaveVerticesGlobalIDs.begin();
      std::vector<AMP::Mesh::MeshElementID>::iterator masterVerticesGlobalIDsIterator = masterVerticesGlobalIDs.begin();
      std::vector<size_t>::iterator masterVerticesOwnerRanksIterator = d_MasterVerticesOwnerRanks.begin();

      //  double basis_functions_values_on_face[4];
      std::vector<int>::const_iterator flagsIterator = flags.begin(),
        flagsIterator_end = flags.end();
      for ( ; flagsIterator != flagsIterator_end; ++flagsIterator) {
        // TODO: the following if statement is debug only
        if (*flagsIterator == AMP::Mesh::DendroSearch::NotFound) {
          std::vector<double> blackSheepCoord = (mesh->getElement(*tmpSlaveVerticesGlobalIDsConstIterator)).coord();
          d_fout<<blackSheepCoord[0]<<"  "
              <<blackSheepCoord[1]<<"  "
              <<blackSheepCoord[2]<<"\n";
        } // end if
        //    AMP_ASSERT( (*flagsIterator == AMP::Mesh::DendroSearch::NotFound) || (*flagsIterator == AMP::Mesh::DendroSearch::FoundOnBoundary) );
        if (*flagsIterator == AMP::Mesh::DendroSearch::FoundOnBoundary) {
          hex8_element_t::get_basis_functions_values_on_face(tmpSlaveVerticesLocalCoordOnFacePointerToConst, masterShapeFunctionsValuesPointer);
          for (size_t d = 0; d < 2; ++d) { ++tmpSlaveVerticesLocalCoordOnFacePointerToConst; }
          for (size_t v = 0; v < 4; ++v) { ++masterShapeFunctionsValuesPointer; }
          *slaveVerticesGlobalIDsIterator = *tmpSlaveVerticesGlobalIDsConstIterator;
          ++slaveVerticesGlobalIDsIterator;
          ++tmpSlaveVerticesGlobalIDsConstIterator;
          for (size_t d = 0; d < 3; ++d) { 
            *slaveVerticesShiftIterator = *tmpSlaveVerticesShiftConstIterator;
            ++slaveVerticesShiftIterator;
            ++tmpSlaveVerticesShiftConstIterator;
          } // end for d
          for (size_t v = 0; v < 4; ++v) {
            *masterVerticesGlobalIDsIterator = *tmpMasterVerticesGlobalIDsConstIterator;
            *masterVerticesOwnerRanksIterator = masterVerticesGlobalIDsIterator->owner_rank();
            ++masterVerticesGlobalIDsIterator;
            ++tmpMasterVerticesGlobalIDsConstIterator;
            ++masterVerticesOwnerRanksIterator;
          } // end for v
        } else {
          for (size_t d = 0; d < 2; ++d) { ++tmpSlaveVerticesLocalCoordOnFacePointerToConst; }
          ++tmpSlaveVerticesGlobalIDsConstIterator;
          for (size_t d = 0; d < 3; ++d) { ++tmpSlaveVerticesShiftConstIterator; }
          for (size_t v = 0; v < 4; ++v) { ++tmpMasterVerticesGlobalIDsConstIterator; }
        } // end if
      } // end for
      AMP_ASSERT( tmpSlaveVerticesLocalCoordOnFacePointerToConst == &(tmpSlaveVerticesLocalCoordOnFace[0])+2*nSlaveVertices );
      AMP_ASSERT( tmpSlaveVerticesShiftConstIterator == tmpSlaveVerticesShift.end() );
      AMP_ASSERT( tmpSlaveVerticesGlobalIDsConstIterator == tmpSlaveVerticesGlobalIDs.end() );
      AMP_ASSERT( tmpMasterVerticesGlobalIDsConstIterator == tmpMasterVerticesGlobalIDs.end() );
      AMP_ASSERT( slaveVerticesShiftIterator == d_SlaveVerticesShift.end() );
      AMP_ASSERT( slaveVerticesGlobalIDsIterator == d_SlaveVerticesGlobalIDs.end() );
      AMP_ASSERT( masterShapeFunctionsValuesPointer == &(d_MasterShapeFunctionsValues[0])+4*nConstraints );
      AMP_ASSERT( masterVerticesGlobalIDsIterator == masterVerticesGlobalIDs.end() );
      AMP_ASSERT( masterVerticesOwnerRanksIterator == d_MasterVerticesOwnerRanks.end() );
      tmpSlaveVerticesGlobalIDs.clear();
      tmpMasterVerticesGlobalIDs.clear();
      tmpSlaveVerticesShift.clear();
      tmpSlaveVerticesLocalCoordOnFace.clear();
      flags.clear();

      /** setup for apply */
      size_t npes = comm.getSize();
      d_SendCnts.resize(npes);
      std::fill(d_SendCnts.begin(), d_SendCnts.end(), 0);
      for (size_t i = 0; i < 4*nConstraints; ++i) {
        ++d_SendCnts[d_MasterVerticesOwnerRanks[i]]; 
      } // end for i
      d_SendDisps.resize(npes);
      d_SendDisps[0] = 0;
      for (size_t i = 1; i < npes; ++i) {
        d_SendDisps[i] = d_SendDisps[i-1] + d_SendCnts[i-1]; 
      } // end for i
      AMP_ASSERT( d_SendDisps[npes-1] + d_SendCnts[npes-1] == 4*nConstraints );

      std::vector<int> tmpSendCnts(npes, 0);
      d_MasterVerticesMap.resize(4*nConstraints, nConstraints);
      std::vector<AMP::Mesh::MeshElementID> sendMasterVerticesGlobalIDs(d_SendDisps[npes-1]+d_SendCnts[npes-1], AMP::Mesh::MeshElementID());
      for (size_t i = 0; i < 4*nConstraints; ++i) {
        size_t sendToRank = d_MasterVerticesOwnerRanks[i];
        d_MasterVerticesMap[i] = d_SendDisps[sendToRank] + tmpSendCnts[sendToRank];
        sendMasterVerticesGlobalIDs[d_MasterVerticesMap[i]] = masterVerticesGlobalIDs[i];
        ++tmpSendCnts[sendToRank];
      } // end for i 
      AMP_ASSERT( std::equal(tmpSendCnts.begin(), tmpSendCnts.end(), d_SendCnts.begin()) );
      tmpSendCnts.clear();

      d_RecvCnts.resize(npes);
      comm.allToAll(1, &(d_SendCnts[0]), &(d_RecvCnts[0]));
      d_RecvDisps.resize(npes);
      d_RecvDisps[0] = 0;
      for (size_t i = 1; i < npes; ++i) {
        d_RecvDisps[i] = d_RecvDisps[i-1] + d_RecvCnts[i-1];
      } // end for i
      d_RecvMasterVerticesGlobalIDs.resize(d_RecvDisps[npes-1]+d_RecvCnts[npes-1]);
      comm.allToAll((!(sendMasterVerticesGlobalIDs.empty()) ? &(sendMasterVerticesGlobalIDs[0]) : NULL), &(d_SendCnts[0]), &(d_SendDisps[0]),
          (!(d_RecvMasterVerticesGlobalIDs.empty()) ? &(d_RecvMasterVerticesGlobalIDs[0]) : NULL), &(d_RecvCnts[0]), &(d_RecvDisps[0]), true);

      d_TransposeSendCnts.resize(npes);
      d_TransposeSendDisps.resize(npes);
      d_TransposeRecvCnts.resize(npes);
      d_TransposeRecvDisps.resize(npes); 
      std::copy(d_SendCnts.begin(), d_SendCnts.end(), d_TransposeSendCnts.begin());
      std::copy(d_SendDisps.begin(), d_SendDisps.end(), d_TransposeSendDisps.begin());
      std::copy(d_RecvCnts.begin(), d_RecvCnts.end(), d_TransposeRecvCnts.begin());
      std::copy(d_RecvDisps.begin(), d_RecvDisps.end(), d_TransposeRecvDisps.begin());

      std::swap_ranges(d_RecvCnts.begin(), d_RecvCnts.end(), d_SendCnts.begin());
      std::swap_ranges(d_RecvDisps.begin(), d_RecvDisps.end(), d_SendDisps.begin());

      getVectorIndicesFromGlobalIDs(d_SlaveVerticesGlobalIDs, d_SlaveIndices);
      getVectorIndicesFromGlobalIDs(d_RecvMasterVerticesGlobalIDs, d_RecvMasterIndices);

      for (size_t i = 0; i < npes; ++i) {
        d_SendCnts[i] *= d_DOFsPerNode; 
        d_SendDisps[i] *= d_DOFsPerNode; 
        d_RecvCnts[i] *= d_DOFsPerNode; 
        d_RecvDisps[i] *= d_DOFsPerNode; 
        d_TransposeSendCnts[i] *= d_DOFsPerNode; 
        d_TransposeSendDisps[i] *= d_DOFsPerNode; 
        d_TransposeRecvCnts[i] *= d_DOFsPerNode; 
        d_TransposeRecvDisps[i] *= d_DOFsPerNode; 
      } // end for i

      d_fout<<std::setprecision(15);
      for (size_t i = 0; i < d_SlaveVerticesGlobalIDs.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
          d_fout<<"i="<<i<<"  "
            <<"j="<<j<<"  "
            <<"4*i+j="<<4*i+j<<"  "
            <<"d_MasterShapeFunctionsValues[4*i+j]="<<d_MasterShapeFunctionsValues[4*i+j]<<"\n";
        } // end for j
        for (size_t k = 0; k < d_DOFsPerNode; ++k) {
          d_fout<<"i="<<i<<"  "
            <<"k="<<k<<"  "
            <<"d_DOFsPerNode*i+k="<<d_DOFsPerNode*i+k<<"  "
            <<"d_SlaveIndices[d_DOFsPerNode*i+k]="<<d_SlaveIndices[d_DOFsPerNode*i+k]<<"  "
            <<"d_SlaveVerticesShift[d_DOFsPerNode*i+k]="<<d_SlaveVerticesShift[d_DOFsPerNode*i+k]<<"\n";     
        } // end for k
      } // end for i

      // this is really ugly
      d_SlaveShift = d_SlaveVerticesShift;

    }

    void NodeToFaceContactOperator::getVectorIndicesFromGlobalIDs(const std::vector<AMP::Mesh::MeshElementID> & globalIDs, 
        std::vector<size_t> & vectorIndices) {
      std::vector<size_t> tmpIndices;
      std::vector<AMP::Mesh::MeshElementID>::const_iterator globalIDsConstIterator = globalIDs.begin(), 
        globalIDsConstIterator_end = globalIDs.end();
      vectorIndices.resize(globalIDs.size()*d_DOFsPerNode);
      std::vector<size_t>::iterator vectorIndicesIterator = vectorIndices.begin();
      for ( ; globalIDsConstIterator != globalIDsConstIterator_end; ++globalIDsConstIterator) {
        d_DOFManager->getDOFs(*globalIDsConstIterator, tmpIndices);
        AMP_ASSERT( *globalIDsConstIterator != AMP::Mesh::MeshElementID() );
        AMP_ASSERT( tmpIndices.size() == d_DOFsPerNode );
        std::copy(tmpIndices.begin(), tmpIndices.end(), vectorIndicesIterator);
        for (size_t i = 0; i < d_DOFsPerNode; ++i) { ++vectorIndicesIterator; }
      } // end for
      AMP_ASSERT( vectorIndicesIterator == vectorIndices.end() );
    }

    void NodeToFaceContactOperator::copyMasterToSlave(AMP::LinearAlgebra::Vector::shared_ptr u) {
      /** send and receive the master values */
      AMP::AMP_MPI comm = d_GlobalComm;
      //  AMP::AMP_MPI comm = u->getComm();
      size_t npes = comm.getSize();
      //  size_t rank = comm.getRank();

      std::vector<double> sendMasterValues(d_SendDisps[npes-1]+d_SendCnts[npes-1]);
      for (size_t i = 0; i < npes; ++i) {
        for (size_t j = 0; j < d_SendCnts[i]; j += d_DOFsPerNode) {
          size_t k = d_SendDisps[i] + j;
          u->getLocalValuesByGlobalID(d_DOFsPerNode, &(d_RecvMasterIndices[k]), &(sendMasterValues[k]));
        } // end for j
      } // end for i

      std::vector<double> recvMasterValues(d_RecvDisps[npes-1]+d_RecvCnts[npes-1]);
      comm.allToAll((!(sendMasterValues.empty()) ? &(sendMasterValues[0]) : NULL), &(d_SendCnts[0]), &(d_SendDisps[0]),
          (!(recvMasterValues.empty()) ? &(recvMasterValues[0]) : NULL), &(d_RecvCnts[0]), &(d_RecvDisps[0]), true);
      sendMasterValues.clear();

      /** compute slave values */
      std::vector<double> slaveValues(d_SlaveIndices.size(), 0.0);

      for (size_t i = 0; i < d_SlaveVerticesGlobalIDs.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
          for (size_t k = 0; k < d_DOFsPerNode; ++k) {
            slaveValues[d_DOFsPerNode*i+k] += d_MasterShapeFunctionsValues[4*i+j] * recvMasterValues[d_DOFsPerNode*d_MasterVerticesMap[4*i+j]+k];
          } // end for k
        } // end for j
      } // end for i

      if (!d_SlaveVerticesGlobalIDs.empty()) { 
        u->setLocalValuesByGlobalID(d_SlaveIndices.size(), &(d_SlaveIndices[0]), &(slaveValues[0]));
      } // end if
    }

    void NodeToFaceContactOperator::addSlaveToMaster(AMP::LinearAlgebra::Vector::shared_ptr u) {
      /** send and receive slave value times shape functions values */
      AMP::AMP_MPI comm = d_GlobalComm;
      //  AMP::AMP_MPI comm = r->getComm();
      size_t npes = comm.getSize();

      std::vector<double> sendAddToMasterValues(d_TransposeSendDisps[npes-1]+d_TransposeSendCnts[npes-1]);
      for (size_t i = 0; i < d_SlaveVerticesGlobalIDs.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
          u->getLocalValuesByGlobalID(d_DOFsPerNode, &(d_SlaveIndices[d_DOFsPerNode*i]), &(sendAddToMasterValues[d_DOFsPerNode*d_MasterVerticesMap[4*i+j]])); 
          for (size_t k = 0; k < d_DOFsPerNode; ++k) {
            sendAddToMasterValues[d_DOFsPerNode*d_MasterVerticesMap[4*i+j]+k] *= d_MasterShapeFunctionsValues[4*i+j];
          } // end for k
        } // end for j
      } // end for i

      std::vector<double> recvAddToMasterValues(d_TransposeRecvDisps[npes-1]+d_TransposeRecvCnts[npes-1]);
      comm.allToAll((!(sendAddToMasterValues.empty()) ? &(sendAddToMasterValues[0]) : NULL), &(d_TransposeSendCnts[0]), &(d_TransposeSendDisps[0]),
          (!(recvAddToMasterValues.empty()) ? &(recvAddToMasterValues[0]) : NULL), &(d_TransposeRecvCnts[0]), &(d_TransposeRecvDisps[0]), true);
      sendAddToMasterValues.clear();

      /** add slave value times shape functions values to master values and set slave values to zero */
      if (!d_RecvMasterVerticesGlobalIDs.empty()) {
        u->addLocalValuesByGlobalID(d_RecvMasterIndices.size(), &(d_RecvMasterIndices[0]), &(recvAddToMasterValues[0]));
      } // end if
    }

  } // end namespace Operator
} // end namespace AMP

