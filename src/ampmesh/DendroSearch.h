
#ifndef DENDRO_SEARCH
#define DENDRO_SEARCH

#include "mpi.h"

#include "utils/AMPManager.h"
#include "utils/Utilities.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"

#include "vectors/Vector.h"


#include "ampmesh/Mesh.h"

#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

#include <vector>

#include "sys/sys.h"
#include "par/parUtils.h"
#include "binOps/binUtils.h"
#include "oct/TreeNode.h"
#include "oct/octUtils.h"
#include "oct/nodeAndValues.h"
#include "oct/nodeAndRanks.h"
#include "externVars.h"
#include "dendro.h"

#include "hex8_element_t.h"

void createLocalMeshElementArray(std::vector<AMP::Mesh::MeshElement>& localElemArr, 
    AMP::Mesh::Mesh::shared_ptr meshAdapter) {
  localElemArr.clear();
  AMP::Mesh::MeshIterator el = meshAdapter->getIterator(AMP::Mesh::Volume, 0);
  AMP::Mesh::MeshIterator end_el = el.end();
  AMP_ASSERT(el != end_el);
  for(; el != end_el; ++el) {
    localElemArr.push_back(*el);
  }//end el
}

void setupDSforSearchType(unsigned int & BoxLevel, std::vector<ot::TreeNode>& nodeList, std::vector<int>& stIdxList,
    std::vector<ot::TreeNode>& mins, std::vector<int>& rankList,
    std::vector<int>& elemIdList, std::vector<AMP::Mesh::MeshElement>& localElemArr,
    double* minCoords, double* maxCoords, double* ScalingFactor, 
    AMP::Mesh::Mesh::shared_ptr meshAdapter, AMP::AMP_MPI globalComm) {
  int rank = globalComm.getRank();
  int npes = globalComm.getSize();

  std::vector<double> box = meshAdapter->getBoundingBox();
  for(int i=0; i<meshAdapter->getDim(); ++i) {
    minCoords[i] = box[2*i+0];
    maxCoords[i] = box[2*i+1];
    ScalingFactor[i] = 1.0/(1.0e-10 + maxCoords[i] - minCoords[i]);
  }

  createLocalMeshElementArray(localElemArr, meshAdapter);

  const unsigned int MaxDepth = 30;

  unsigned int totalNumElems = meshAdapter->numGlobalElements(AMP::Mesh::Volume);

  double avgHboxInv = std::pow(totalNumElems, (1.0/3.0));
  assert(avgHboxInv > 1.0);
  BoxLevel = binOp::fastLog2(static_cast<unsigned int>(std::ceil(avgHboxInv)));
  assert(BoxLevel < MaxDepth);

  if(!rank) {
    std::cout<<"BoxLevel = "<<BoxLevel<<std::endl;
  }

  const unsigned int ITPMD = (1u << MaxDepth);
  const double DTPMD = static_cast<double>(ITPMD);
  const double hBox = 1.0/(static_cast<double>(1u<<BoxLevel));

  std::vector< ot::NodeAndValues<int, 1> > nodeAndElemIdList;

  assert(!(localElemArr.empty()));

  for(int eId = 0; eId < localElemArr.size(); ++eId) {
    std::vector<AMP::Mesh::MeshElement> currNodes = localElemArr[eId].getElements(AMP::Mesh::Vertex);
    int minId[3];
    int maxId[3];
    for(size_t i = 0; i < currNodes.size(); ++i) {
      std::vector<double> pt = currNodes[i].coord();
      double scaledPt[3];
      for(int j = 0; j < 3; ++j) {
        scaledPt[j] = ((pt[j] - minCoords[j])*ScalingFactor[j]);
        int id = static_cast<int>(scaledPt[j]/hBox);
        if(i == 0) {
          minId[j] = id;
          maxId[j] = id;
        } else {
          if(minId[j] > id) {
            minId[j] = id;
          }
          if(maxId[j] < id) {
            maxId[j] = id;
          }
        }
      }//end j
    }//end i
    //Performance Improvement: We can skip the boxes that lie
    //completely outside the element.
    for(int k = minId[2]; k <= maxId[2]; ++k) {
      for(int j = minId[1]; j <= maxId[1]; ++j) {
        for(int i = minId[0]; i <= maxId[0]; ++i) {
          unsigned int bX = i*(1u<<(MaxDepth - BoxLevel));
          unsigned int bY = j*(1u<<(MaxDepth - BoxLevel));
          unsigned int bZ = k*(1u<<(MaxDepth - BoxLevel));
          ot::TreeNode box(bX, bY, bZ, BoxLevel, 3, MaxDepth);
          box.setWeight(rank);
          ot::NodeAndValues<int, 1> obj;
          obj.node = box;
          obj.values[0] = eId;
          nodeAndElemIdList.push_back(obj);
        }//end i 
      }//end j 
    }//end k 
  }//end eId

  std::vector< ot::NodeAndValues<int, 1> > tmpList;
  par::sampleSort< ot::NodeAndValues<int, 1> >(
      nodeAndElemIdList, tmpList, (globalComm.getCommunicator()));
  swap(nodeAndElemIdList, tmpList);
  tmpList.clear();

  int numLocalOcts = nodeAndElemIdList.size();
  int numGlobalOcts = globalComm.sumReduce<int>(numLocalOcts);
  if(!rank) {
    std::cout<<"Total num initial octants = "<<numGlobalOcts <<std::endl;
  }

  assert(rankList.empty());
  assert(elemIdList.empty());

  for(int i = 0; i < numLocalOcts; ++i) {
    ot::TreeNode currNode = nodeAndElemIdList[i].node;
    rankList.push_back(currNode.getWeight());
    elemIdList.push_back(nodeAndElemIdList[i].values[0]);
  }//end i

  assert(numLocalOcts > 0);
  assert(nodeList.empty());

  //Local Merge
  {
    ot::TreeNode currNode = nodeAndElemIdList[0].node;
    currNode.setWeight(1);
    nodeList.push_back(currNode);
  }
  for(size_t i = 1; i < nodeAndElemIdList.size(); ++i) {
    ot::TreeNode currNode = nodeAndElemIdList[i].node;
    if( nodeList[nodeList.size() - 1] == currNode ) {
      nodeList[nodeList.size() - 1].addWeight(1);
    } else {
      currNode.setWeight(1);
      nodeList.push_back(currNode);
    }
  }//end i
  nodeAndElemIdList.clear();

  int localFlag = 0;
  if( (rank > 0) && (rank < (npes - 1)) && ((nodeList.size()) == 1) ) {
    localFlag = 1;
  }

  int globalFlag;
  MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_SUM, (globalComm.getCommunicator()));

  int prevRank = rank - 1;
  int nextRank = rank + 1;

  if(globalFlag > 0) {
    int gatherSendBuf = 0;
    if( (rank > 0) && (rank < (npes - 1)) && (nodeList.size() == 1) ) {
      gatherSendBuf = rankList.size();
    }

    int* gatherList = new int[npes];

    MPI_Allgather((&gatherSendBuf), 1, MPI_INT, gatherList, 1, MPI_INT, (globalComm.getCommunicator()));

    if(rank > 0) {
      while(gatherList[prevRank] > 0) {
        --prevRank;
      }//end while
    }

    if(rank < (npes - 1)) {
      while(gatherList[nextRank] > 0) {
        ++nextRank;
      }//end while
    }

    int* sendBoxCnts = new int[npes];
    int* recvBoxCnts = new int[npes];

    int* sendSourceCnts = new int[npes];
    int* recvSourceCnts = new int[npes];

    for(int i = 0; i < npes; ++i) {
      sendBoxCnts[i] = 0;
      recvBoxCnts[i] = 0;
      sendSourceCnts[i] = 0;
      recvSourceCnts[i] = 0;
    }//end i

    if(gatherSendBuf > 0) {
      sendBoxCnts[prevRank] = 1;
      sendSourceCnts[prevRank] = gatherSendBuf;
    }
    for(int i = rank + 1; i < nextRank; ++i) {
      recvBoxCnts[i] = 1;
      recvSourceCnts[i] = gatherList[i];
    }//end i

    delete [] gatherList;

    int* sendBoxDisps = new int[npes];
    int* recvBoxDisps = new int[npes];
    sendBoxDisps[0] = 0;
    recvBoxDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendBoxDisps[i] = sendBoxDisps[i - 1] + sendBoxCnts[i - 1];
      recvBoxDisps[i] = recvBoxDisps[i - 1] + recvBoxCnts[i - 1];
    }//end i

    std::vector<ot::TreeNode> tmpBoxList(recvBoxDisps[npes - 1] + recvBoxCnts[npes - 1]);

    ot::TreeNode* recvBoxBuf = NULL;
    if(!(tmpBoxList.empty())) {
      recvBoxBuf = (&(tmpBoxList[0]));
    }

    MPI_Alltoallv( (&(nodeList[0])), sendBoxCnts, sendBoxDisps, par::Mpi_datatype<ot::TreeNode>::value(),
        recvBoxBuf, recvBoxCnts, recvBoxDisps, par::Mpi_datatype<ot::TreeNode>::value(), (globalComm.getCommunicator()));

    if(gatherSendBuf > 0) {
      nodeList.clear();
    } else {
      for(int i = 0; i < tmpBoxList.size(); ++i) {
        if(tmpBoxList[i] == nodeList[nodeList.size() - 1]) {
          nodeList[nodeList.size() - 1].addWeight(tmpBoxList[i].getWeight());
        } else {
          nodeList.push_back(tmpBoxList[i]);
        }
      }//end i
    }

    delete [] sendBoxCnts;
    delete [] recvBoxCnts;
    delete [] sendBoxDisps;
    delete [] recvBoxDisps;

    int* sendSourceDisps = new int[npes];
    int* recvSourceDisps = new int[npes];
    sendSourceDisps[0] = 0;
    recvSourceDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendSourceDisps[i] = sendSourceDisps[i - 1] + sendSourceCnts[i - 1];
      recvSourceDisps[i] = recvSourceDisps[i - 1] + recvSourceCnts[i - 1];
    }//end i

    std::vector<int> tmpRankList(recvSourceDisps[npes - 1] + recvSourceCnts[npes - 1]);
    std::vector<int> tmpElemIdList(recvSourceDisps[npes - 1] + recvSourceCnts[npes - 1]);

    int* recvRankBuf = NULL;
    int* recvElemIdBuf = NULL;
    if(!(tmpRankList.empty())) {
      recvRankBuf = (&(tmpRankList[0]));
      recvElemIdBuf = (&(tmpElemIdList[0]));
    }

    MPI_Alltoallv( (&(rankList[0])), sendSourceCnts, sendSourceDisps, MPI_INT,
        recvRankBuf, recvSourceCnts, recvSourceDisps, MPI_INT, (globalComm.getCommunicator()));
    MPI_Alltoallv( (&(elemIdList[0])), sendSourceCnts, sendSourceDisps, MPI_INT,
        recvElemIdBuf, recvSourceCnts, recvSourceDisps, MPI_INT, (globalComm.getCommunicator()));

    if(gatherSendBuf > 0) {
      rankList.clear();
      elemIdList.clear();
    } else {
      if(!(tmpRankList.empty())) {
        rankList.insert(rankList.end(), tmpRankList.begin(), tmpRankList.end());
        elemIdList.insert(elemIdList.end(), tmpElemIdList.begin(), tmpElemIdList.end());
      }
    }

    delete [] sendSourceCnts;
    delete [] recvSourceCnts;
    delete [] sendSourceDisps;
    delete [] recvSourceDisps;
  }

  if(!(nodeList.empty())) {
    assert(nodeList.size() >= 2);

    ot::TreeNode prevBox;
    ot::TreeNode nextBox;
    ot::TreeNode firstBox = nodeList[0];
    ot::TreeNode lastBox = nodeList[nodeList.size() - 1];
    MPI_Request recvPrevReq;
    MPI_Request recvNextReq;
    MPI_Request sendFirstReq;
    MPI_Request sendLastReq;
    if(rank > 0) {
      MPI_Irecv(&prevBox, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 1, (globalComm.getCommunicator()), &recvPrevReq);
      MPI_Isend(&firstBox, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 2, (globalComm.getCommunicator()), &sendFirstReq);
    }
    if(rank < (npes - 1)) {
      MPI_Irecv(&nextBox, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 2, (globalComm.getCommunicator()), &recvNextReq);
      MPI_Isend(&lastBox, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 1, (globalComm.getCommunicator()), &sendLastReq);
    }

    if(rank > 0) {
      MPI_Status status;
      MPI_Wait(&recvPrevReq, &status);
      MPI_Wait(&sendFirstReq, &status);
    }
    if(rank < (npes - 1)) {
      MPI_Status status;
      MPI_Wait(&recvNextReq, &status);
      MPI_Wait(&sendLastReq, &status);
    }

    bool removeFirst = false;
    bool addToLast = false;
    if(rank > 0) {
      if(prevBox == firstBox) {
        removeFirst = true;
      }
    }
    if(rank < (npes - 1)) {
      if(nextBox == lastBox) {
        addToLast = true;
      }
    }

    MPI_Request recvRankReq;
    MPI_Request recvElemIdReq;
    if(addToLast) {
      int numPts = rankList.size();
      rankList.resize(numPts + (nextBox.getWeight()));
      elemIdList.resize(numPts + (nextBox.getWeight()));
      nodeList[nodeList.size() - 1].addWeight(nextBox.getWeight());
      MPI_Irecv((&(rankList[numPts])), ((nextBox.getWeight())), MPI_INT, nextRank,
          3, (globalComm.getCommunicator()), &recvRankReq);
      MPI_Irecv((&(elemIdList[numPts])), ((nextBox.getWeight())), MPI_INT, nextRank,
          4, (globalComm.getCommunicator()), &recvElemIdReq);
    }
    if(removeFirst) {
      MPI_Send((&(rankList[0])), ((firstBox.getWeight())), MPI_INT, prevRank, 3, (globalComm.getCommunicator()));
      MPI_Send((&(elemIdList[0])), ((firstBox.getWeight())), MPI_INT, prevRank, 4, (globalComm.getCommunicator()));
      nodeList.erase(nodeList.begin());
    }
    if(addToLast) {
      MPI_Status status;
      MPI_Wait(&recvRankReq, &status);
      MPI_Wait(&recvElemIdReq, &status);
    }
    if(removeFirst) {
      rankList.erase(rankList.begin(), rankList.begin() + ((firstBox.getWeight())));
      elemIdList.erase(elemIdList.begin(), elemIdList.begin() + ((firstBox.getWeight())));
    }

    stIdxList.resize(nodeList.size());

    stIdxList[0] = 0;
    for(int i = 1; i < nodeList.size(); ++i) {
      stIdxList[i] = stIdxList[i - 1] + nodeList[i - 1].getWeight();
    }//end i
  }

  ot::TreeNode firstNode;
  if(!(nodeList.empty())) {
    firstNode = nodeList[0];
    firstNode.setWeight(rank);
  }
  mins.resize(npes);
  MPI_Allgather(&firstNode, 1, par::Mpi_datatype<ot::TreeNode>::value(), 
      &(mins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), globalComm.getCommunicator() );

  std::vector<ot::TreeNode> tmpMins;
  for(int i = 0; i < npes; ++i) {
    if(mins[i].getDim() > 0) {
      tmpMins.push_back(mins[i]);
    }
  }//end i
  swap(mins, tmpMins);
  tmpMins.clear();

  int minFineListLen = nodeList[0].getWeight();
  int maxFineListLen = nodeList[0].getWeight();
  for(int i = 1; i < nodeList.size(); ++i) {
    if(minFineListLen > nodeList[i].getWeight()) {
      minFineListLen = nodeList[i].getWeight();
    }
    if(maxFineListLen < nodeList[i].getWeight()) {
      maxFineListLen = nodeList[i].getWeight();
    }
  }//end i

  int globalMinFineListLen = globalComm.minReduce<int>(minFineListLen);
  int globalMaxFineListLen = globalComm.maxReduce<int>(maxFineListLen);

  numLocalOcts = nodeList.size();
  numGlobalOcts = globalComm.sumReduce<int>(numLocalOcts);

  if(!rank) {
    std::cout<<"Total num final octants = "<<numGlobalOcts <<std::endl;
    std::cout<<"Global Min Fine List Length = "<<globalMinFineListLen <<std::endl;
    std::cout<<"Global Max Fine List Length = "<<globalMaxFineListLen <<std::endl;
  }
}

class DendroSearch {
  public:
    DendroSearch(AMP::AMP_MPI comm, AMP::Mesh::Mesh::shared_ptr mesh) : globalComm(comm), meshAdapter(mesh) {
      verbose = true;
      rank = globalComm.getRank();
      npes = globalComm.getSize();
      setupDendro();
    }

    void interpolate(AMP::LinearAlgebra::Vector::shared_ptr vectorField, const std::vector<double> & pts, 
        std::vector<double> & results, std::vector<bool> & foundPt) {
      int numLocalPts = (pts.size())/3;

      double searchBeginTime, searchStep1Time, searchStep2Time, searchStep3Time, 
             searchStep4Time, searchStep5Time, searchStep6Time, searchStep7Time;
      if(verbose) {
        globalComm.barrier();
        searchBeginTime = MPI_Wtime();
      }

      const unsigned int MaxDepth = 30;
      const unsigned int ITPMD = (1u << MaxDepth);
      const double DTPMD = static_cast<double>(ITPMD);

      std::vector<ot::NodeAndValues<double, 4> > ptsWrapper(numLocalPts);
      for(int i = 0; i < numLocalPts; ++i) {
        double x = pts[3*i];
        double y = pts[(3*i) + 1];
        double z = pts[(3*i) + 2];
        double scaledX = ((x - minCoords[0])*ScalingFactor[0]);
        double scaledY = ((y - minCoords[1])*ScalingFactor[1]);
        double scaledZ = ((z - minCoords[2])*ScalingFactor[2]);
        unsigned int pX = static_cast<unsigned int>(scaledX*DTPMD);
        unsigned int pY = static_cast<unsigned int>(scaledY*DTPMD);
        unsigned int pZ = static_cast<unsigned int>(scaledZ*DTPMD);

        ptsWrapper[i].node =  ot::TreeNode(pX, pY, pZ, MaxDepth, 3, MaxDepth);
        ptsWrapper[i].node.setWeight(rank);
        ptsWrapper[i].values[0] = x;
        ptsWrapper[i].values[1] = y;
        ptsWrapper[i].values[2] = z;
        ptsWrapper[i].values[3] = i;     
      }//end i

      if(verbose) {
        globalComm.barrier();
        searchStep1Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-1 of search: "<<(searchStep1Time - searchBeginTime)<<" seconds."<<std::endl;
        }
      }

      //Performance Question: Should PtsWrapper be sorted or not?
      //If PtsWrapper is sorted (even just a local sort), we can skip the
      //binary searches and use binning instead.  Binning is amortized constant
      //time and using binary searches would be logarithmic. This is just a matter
      //of constants since sorting is also logarithmic.

      int* sendCnts = new int[npes];
      for(int i = 0; i < npes; ++i) {
        sendCnts[i] = 0;
      }//end i

      std::vector<int> part(numLocalPts, -1);
      for(int i = 0; i < numLocalPts; ++i) {
        unsigned int retIdx;
        bool found = seq::maxLowerBound<ot::TreeNode>(mins, (ptsWrapper[i].node), retIdx, NULL, NULL);
        if(found) {
          part[i] = mins[retIdx].getWeight();
          sendCnts[part[i]]++;
        }
      }//end i

      if(verbose) {
        globalComm.barrier();
        searchStep2Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-2 of search: "<<(searchStep2Time - searchStep1Time)<<" seconds."<<std::endl;
        }
      }

      int* recvCnts = new int[npes];
      MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, (globalComm.getCommunicator()));

      int* sendDisps = new int[npes];
      int* recvDisps = new int[npes];
      sendDisps[0] = 0;
      recvDisps[0] = 0;
      for(int i = 1; i < npes; ++i) {
        sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
        recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
      }//end i

      std::vector<ot::NodeAndValues<double, 4> > sendList(sendDisps[npes - 1] + sendCnts[npes - 1]);
      ot::NodeAndValues<double, 4>* sendListPtr = NULL;
      if(!(sendList.empty())) {
        sendListPtr = &(sendList[0]);
      }

      for(int i = 0; i < npes; ++i) {
        sendCnts[i] = 0;
      }//end i

      for(int i = 0; i < numLocalPts; ++i) {
        if(part[i] >= 0) {
          sendListPtr[sendDisps[part[i]] + sendCnts[part[i]]] = ptsWrapper[i];
          sendCnts[part[i]]++;
        }
      }//end i
      ptsWrapper.clear();

      std::vector<ot::NodeAndValues<double, 4> > recvList(recvDisps[npes - 1] + recvCnts[npes - 1]);
      ot::NodeAndValues<double, 4>* recvListPtr = NULL;
      if(!(recvList.empty())) {
        recvListPtr = &(recvList[0]);
      }
      MPI_Alltoallv( sendListPtr, sendCnts, sendDisps, par::Mpi_datatype<ot::NodeAndValues<double, 4> >::value(),
          recvListPtr, recvCnts, recvDisps, par::Mpi_datatype<ot::NodeAndValues<double, 4> >::value(),
          (globalComm.getCommunicator()) );
      sendList.clear();

      if(verbose) {
        globalComm.barrier();
        searchStep3Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-3 of search: "<<(searchStep3Time - searchStep2Time)<<" seconds."<<std::endl;
        }
      }

      for(int i = 0; i < npes; ++i) {
        sendCnts[i] = 0;
      }//end i

      std::vector<int> ptToOctMap((recvList.size()), -1);
      int* ptToOctMapPtr = &(ptToOctMap[0]);
      int* rankListPtr = &(rankList[0]);
      ot::TreeNode* nodeListPtr = &(nodeList[0]);
      int* stIdxListPtr = &(stIdxList[0]);
      for(int i = 0; i < recvList.size(); ++i) {
        unsigned int retIdx;
        seq::maxLowerBound<ot::TreeNode>(nodeList, (recvListPtr[i].node), retIdx, NULL, NULL);
        if( nodeListPtr[retIdx].isAncestor(recvListPtr[i].node) ) {
          ptToOctMapPtr[i] = retIdx;
          int stIdx = stIdxListPtr[retIdx];
          for(int j = 0; j < nodeListPtr[retIdx].getWeight(); ++j) {
            sendCnts[rankListPtr[stIdx + j]]++;
          }//end j
        }
      }//end i

      if(verbose) {
        globalComm.barrier();
        searchStep4Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-4 of search: "<<(searchStep4Time - searchStep3Time)<<" seconds."<<std::endl;
        }
      }

      MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, (globalComm.getCommunicator()));

      sendDisps[0] = 0;
      recvDisps[0] = 0;
      for(int i = 1; i < npes; ++i) {
        sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
        recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
      }//end i

      std::vector<double> sendPtsList(6*(sendDisps[npes - 1] + sendCnts[npes - 1]));
      double* sendPtsPtr = NULL;
      if(!(sendPtsList.empty())) {
        sendPtsPtr = &(sendPtsList[0]);
      }

      for(int i = 0; i < npes; ++i) {
        sendCnts[i] = 0;
      }//end i

      int* elemIdListPtr = &(elemIdList[0]);
      for(int i = 0; i < ptToOctMap.size(); ++i) {
        if(ptToOctMapPtr[i] >= 0) {
          int stIdx = stIdxListPtr[ptToOctMapPtr[i]];
          for(int j = 0; j < nodeListPtr[ptToOctMapPtr[i]].getWeight(); ++j) {
            int recvRank = rankListPtr[stIdx + j];
            int currIdx = 6*(sendDisps[recvRank] + sendCnts[recvRank]);
            //Local Id of this element on the processor that owns this element
            sendPtsPtr[currIdx] = elemIdListPtr[stIdx + j];
            //Pt's x coordinate
            sendPtsPtr[currIdx + 1] = recvListPtr[i].values[0];
            //Pt's y coordinate
            sendPtsPtr[currIdx + 2] = recvListPtr[i].values[1];
            //Pt's z coordinate
            sendPtsPtr[currIdx + 3] = recvListPtr[i].values[2];
            //Local Id of Pt on the processor that owns this Pt
            sendPtsPtr[currIdx + 4] = recvListPtr[i].values[3];
            //rank of processor that owns Pt
            sendPtsPtr[currIdx + 5] = recvListPtr[i].node.getWeight();
            sendCnts[recvRank]++;
          }//end j
        }
      }//end i
      recvList.clear();

      for(int i = 0; i < npes; ++i) {
        sendCnts[i] *= 6;
        sendDisps[i] *= 6;
        recvCnts[i] *= 6;
        recvDisps[i] *= 6;
      }//end i

      std::vector<double> recvPtsList(recvDisps[npes - 1] + recvCnts[npes - 1]);
      double* recvPtsPtr = NULL;
      if(!(recvPtsList.empty())) {
        recvPtsPtr = &(recvPtsList[0]);
      }
      MPI_Alltoallv( sendPtsPtr, sendCnts, sendDisps, MPI_DOUBLE,
          recvPtsPtr, recvCnts, recvDisps, MPI_DOUBLE, (globalComm.getCommunicator()) );
      sendPtsList.clear();

      if(verbose) {
        globalComm.barrier();
        searchStep5Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-5 of search: "<<(searchStep5Time - searchStep4Time)<<" seconds."<<std::endl;
        }
      }

      vectorField->makeConsistent(  AMP::LinearAlgebra::Vector::CONSISTENT_SET );

      //dofMap = vectorField->getDOFmap();

      for(int i = 0; i < npes; ++i) {
        sendCnts[i] = 0;
      }//end i

      std::vector<std::vector<double> > tmpSendResults(npes);
      int numRecvPts = recvPtsList.size()/6;
      std::vector<double> tmpPt(3);
      std::vector<double> dummy(24);
      for(int i = 0; i < numRecvPts; ++i) {
        int eId = static_cast<int>(recvPtsList[6*i]);
        AMP::Mesh::MeshElement* el = &(localElemArr[eId]);
        tmpPt[0] = recvPtsList[(6*i) + 1];
        tmpPt[1] = recvPtsList[(6*i) + 2];
        tmpPt[2] = recvPtsList[(6*i) + 3];
        const std::vector<AMP::Mesh::MeshElement> support_points = el->getElements(AMP::Mesh::Vertex);
        for (unsigned int j = 0; j < 8; ++j) {
          std::vector<double> point_coord = support_points[j].coord();
          dummy[3*j+0] = point_coord[0];
          dummy[3*j+1] = point_coord[1];
          dummy[3*j+2] = point_coord[2];
        } // end for j
        hex8_element_t volume_element(dummy);
        bool found = (volume_element.project_on_face(tmpPt).first != 99);
        if(found) {
          //Get DofIds for this element.
          //Interpolate and compute the value
          double value;
          unsigned int ptLocalId = static_cast<unsigned int>(recvPtsList[(6*i) + 4]);
          unsigned int ptProcId = static_cast<unsigned int>(recvPtsList[(6*i) + 5]);
          sendCnts[ptProcId] += 2;
          tmpSendResults[ptProcId].push_back(static_cast<double>(ptLocalId));
          tmpSendResults[ptProcId].push_back(value);
        }
      }//end i
      recvPtsList.clear();

      if(verbose) {
        globalComm.barrier();
        searchStep6Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-6 of search: "<<(searchStep6Time - searchStep5Time)<<" seconds."<<std::endl;
        }
      }

      MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, (globalComm.getCommunicator()));

      sendDisps[0] = 0;
      recvDisps[0] = 0;
      for(int i = 1; i < npes; ++i) {
        sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
        recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
      }//end i

      std::vector<double> sendResults(sendDisps[npes - 1] + sendCnts[npes - 1]);
      for(int i = 0; i < npes; ++i) {
        for(int j = 0; j < sendCnts[i]; ++j) {
          sendResults[sendDisps[i] + j] = tmpSendResults[i][j];
        }//end j
      }//end i
      tmpSendResults.clear();

      std::vector<double> recvResults(recvDisps[npes - 1] + recvCnts[npes - 1]);

      double* sendResultsPtr;
      if(!(sendResults.empty())) {
        sendResultsPtr = &(sendResults[0]);
      }
      double* recvResultsPtr;
      if(!(recvResults.empty())) {
        recvResultsPtr = &(recvResults[0]);
      }

      MPI_Alltoallv(sendResultsPtr, sendCnts, sendDisps, MPI_DOUBLE, 
          recvResultsPtr, recvCnts, recvDisps, (globalComm.getCommunicator()));
      sendResults.clear();

      delete [] sendCnts;
      delete [] sendDisps;
      delete [] recvCnts;
      delete [] recvDisps;

      //Points that are not found will have a result = 0. 
      results.clear();
      results.resize(numLocalPts, 0.0);
      foundPt.clear();
      foundPt.resize(numLocalPts, false);
      for(size_t i = 0; i < recvResults.size(); i+= 2) {
        unsigned int locId = static_cast<unsigned int>(recvResults[i]);
        double val = recvResults[i + 1];
        results[locId] = val;
        foundPt[locId] = true;
      }//end i

      if(verbose) {
        globalComm.barrier();
        searchStep7Time = MPI_Wtime();
        if(!rank) {
          std::cout<<"Time for step-7 of search: "<<(searchStep7Time - searchStep6Time)<<" seconds."<<std::endl;
        }
      }
    }

  private:
    bool verbose;
    AMP::AMP_MPI globalComm;
    AMP::Mesh::Mesh::shared_ptr meshAdapter;
    int rank, npes;
    double minCoords[3];
    double maxCoords[3];
    double ScalingFactor[3];
    std::vector<ot::TreeNode> nodeList;
    std::vector<int> stIdxList;
    std::vector<int> rankList;
    std::vector<int> elemIdList;
    std::vector<AMP::Mesh::MeshElement> localElemArr;
    std::vector<ot::TreeNode> mins;
    unsigned int BoxLevel;

    void setupDendro() {
      double setupBeginTime, setupEndTime;
      if(verbose) {
        globalComm.barrier();
        setupBeginTime = MPI_Wtime();
      }
      setupDSforSearchType(BoxLevel, nodeList, stIdxList, mins, rankList, elemIdList, localElemArr,
          minCoords, maxCoords, ScalingFactor, meshAdapter, globalComm );
      if(verbose) {
        globalComm.barrier();
        setupEndTime = MPI_Wtime();
        if(!rank) {
          std::cout<<"Finished setting up DS for search in "<<(setupEndTime - setupBeginTime)<<" seconds."<<std::endl;
        }
      }
    }

};

#endif // DENDRO_SEARCH