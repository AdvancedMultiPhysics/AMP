#include "AMP/operators/map/NodeToNodeMap.h"
/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.hpp"
template int AMP::AMP_MPI::allGather<AMP::Operator::NodeToNodeMap::Point>(
    AMP::Operator::NodeToNodeMap::Point const *,
    int,
    AMP::Operator::NodeToNodeMap::Point *,
    int *,
    int *,
    bool ) const;
template void AMP::Utilities::quicksort<AMP::Operator::NodeToNodeMap::Point>(
    std::vector<AMP::Operator::NodeToNodeMap::Point> & );
template size_t AMP::Utilities::findfirst<AMP::Operator::NodeToNodeMap::Point>(
    std::vector<AMP::Operator::NodeToNodeMap::Point> const &,
    AMP::Operator::NodeToNodeMap::Point const & );
