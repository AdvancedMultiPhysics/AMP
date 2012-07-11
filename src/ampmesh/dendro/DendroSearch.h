
#ifndef DENDRO_SEARCH
#define DENDRO_SEARCH

#include "utils/AMPManager.h"
#include "utils/Utilities.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"

#include "vectors/Vector.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/MeshElement.h"
#include "ampmesh/hex8_element_t.h"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

#include "sys/sys.h"
#include "par/parUtils.h"
#include "binOps/binUtils.h"
#include "oct/TreeNode.h"
#include "oct/octUtils.h"
#include "oct/nodeAndValues.h"
#include "oct/nodeAndRanks.h"
#include "dendro.h"


class DendroSearch {
  public:
    struct ProjectOnBoundaryData; 

    enum SearchStatus { NotFound = 0, Found, FoundOnBoundary, FoundNotOnBoundary };

    DendroSearch(AMP::AMP_MPI comm, AMP::Mesh::Mesh::shared_ptr mesh);

    void searchAndInterpolate(AMP::LinearAlgebra::Vector::shared_ptr vectorField, const unsigned int dofsPerNode,
        const std::vector<double> & pts, std::vector<double> & results, std::vector<bool> & foundPt);

    void search(const std::vector<double> & pts);

    void interpolate(AMP::LinearAlgebra::Vector::shared_ptr vectorField, const unsigned int dofsPerNode,
        std::vector<double> & results, std::vector<bool> & foundPt);

//    void projectOnBoundaryID(const int boundaryID, //std::vector<ProjectOnBoundaryStatus> & projectOnBoundaryStatuses);
//        const unsigned int dofsPerNode, AMP::Discretization::DOFManager::shared_ptr dofManager,
//        std::vector<size_t> & nodeIDs, std::vector<size_t> & nodeOwnerRanks, std::vector<double> & localCoords, std::vector<int> & flags);

    void projectOnBoundaryID(const int boundaryID, std::vector<AMP::Mesh::MeshElementID> & faceVerticesGlobalIDs, 
        std::vector<double> & shiftGlobalCoords, std::vector<double> & projectionLocalCoordsOnFace, std::vector<int> & flags);

  private:
    AMP::AMP_MPI d_globalComm;
    AMP::Mesh::Mesh::shared_ptr d_meshAdapter;
    std::vector<AMP::Mesh::MeshElement> d_localElemArr;
    std::vector<ot::TreeNode> d_nodeList;
    std::vector<ot::TreeNode> d_mins;
    std::vector<double> d_minCoords;
    std::vector<double> d_maxCoords;
    std::vector<double> d_scalingFactor;
    std::vector<double> d_foundPts;
    std::vector<int> d_stIdxList;
    std::vector<int> d_rankList;
    std::vector<int> d_elemIdList;
    std::vector<int> d_sendCnts;
    std::vector<int> d_sendDisps;
    std::vector<int> d_recvCnts;
    std::vector<int> d_recvDisps;
    unsigned int d_boxLevel;
    int d_numLocalPts;
    int d_rank;
    int d_npes;
    bool d_verbose;
  
    std::vector<hex8_element_t> d_volume_elements;

    void setupDendro();
    void setupDSforSearch();
    void createLocalMeshElementArray();
};

struct DendroSearch::ProjectOnBoundaryData {
  size_t d_PointLocalID;
  SearchStatus d_SearchStatus;
  AMP::Mesh::MeshElementID d_FaceVerticesIDs[4];
  double d_ProjectionLocalCoordsOnFace[2]; 
  double d_ShiftGlobalCoords[3];

};

#endif // DENDRO_SEARCH


