#ifndef included_AMP_MLoptions
#define included_AMP_MLoptions

#include "AMP/utils/Database.h"

#include <memory>
#include <string>


namespace AMP::Solver {

class MLoptions
{
public:
    explicit MLoptions( std::shared_ptr<AMP::Database> db );

    ~MLoptions() {}

    static void addDefaults( const std::string &problemType, std::shared_ptr<AMP::Database> db );

    std::string d_problemType;
    int d_maxLevels;
    int d_pdeEquations;
    std::string d_precType;
    std::string d_increasingDecreasing;
    std::string d_aggregationType;
    double d_aggregationDampingFactor;
    double d_aggregationThreshold;
    int d_nodesPerAggregate;
    int d_nextLevelAggregatesPerProcess;
    std::string d_eigenAnalysisType;
    int d_eigenAnalysisIterations;
    bool d_enableEnergyMinimization;
    std::string d_smootherType;
    int d_smootherSweeps;
    double d_smootherDampingFactor;
    std::string d_prePost;
    int d_coarseMaxSize;
    std::string d_coarseType;
    bool d_aggregationAuxEnable;
    double d_aggregationAuxThreshold;
    std::string d_nullSpaceType;
    int d_nullSpaceDimension;
    bool d_nullSpaceAddDefaultVectors;
};
} // namespace AMP::Solver

#endif
