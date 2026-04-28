#ifndef included_AMP_SAMRBuilder
#define included_AMP_SAMRBuilder

#include "AMP/mesh/SAMRAI/SAMRHierarchy.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"

#include <memory>


namespace SAMRAI::hier {
class PatchHierarchy;
}
namespace SAMRAI::tbox {
class Database;
}
namespace SAMRAI::mesh {
class GriddingAlgorithm;
class StandardTagAndInitStrategy;
class GriddingAlgorithmStrategy;
} // namespace SAMRAI::mesh


namespace AMP::Mesh {


class SAMRBuilder
{
public:
    /*!
     * This function creates the patch hierarchy from the input database,
     * and optionally a user provided TagAndIntStrategy or GriddingAlgorithm.
     *
     * There are several advanced options based on the input file:
     *
     * If the ManualBoxes database is present, it allows the users to explicitly
     * create a hierarchy with the desired boxes and the ranks specified:
     * \verbatim
     *    ManualBoxes{
     *       manually_create_levels = FALSE
     *       level_0 {
     *          boxes =  [(0,0,0), (63,127,0)]
     *          rank = 0
     *       }
     *       level_1 {
     *       }
     * \endverbatim
     *
     * BoxGeneratorStrategy - This is an optional field that specifies the desired
     * BoxGeneratorStrategy:
     *   "BergerRigoutsos", "TileClustering"
     *
     * LoadBalanceStrategy - This is an optional field that specifies the desired
     * LoadBalanceStrategy:
     *   "TreeLoadBalancer", "ChopAndPackLoadBalancer", "SimpleLoadBalancer"
     *
     * @param db            Input database
     * @param test_object   Optional StandardTagAndInitStrategy object.
     *                      If this is not nullptr on input, the provided StandardTagAndInitStrategy
     *                      will be used.  If it is nullptr on input, then a BogusTagAndInitStrategy
     *                      will be created and a pointer returned.  If this is nullptr and
     *                      gridding_algorithm is not nullptr, then the StandardTagAndInitStrategy
     *                      in the GriddingAlgorithm will be returned instead.
     * @param gridding_algorithm    Optional GriddingAlgorithm object.
     *                      If this is not nullptr on input, the provided GriddingAlgorithm
     *                      will be used.  If it is nullptr on input, then a default
     *                      GriddingAlgorithm will be created and a pointer returned.
     * @param max_gcw       Maximum ghost cell width
     * @param restrict_gcw_domain Restrict chost cell width?
     * @param mpi           Communicator to use
     */
    static std::shared_ptr<AMP::Mesh::SAMRHierarchy>
    buildHierarchy( std::shared_ptr<AMP::Database> db,
                    std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> &test_object,
                    std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_algorithm,
                    int max_gcw              = 0,
                    bool restrict_gcw_domain = true,
                    const AMP::AMP_MPI &mpi  = AMP::AMP_MPI( AMP_COMM_WORLD ) );

    static std::shared_ptr<AMP::Mesh::SAMRHierarchy>
    buildHierarchy( std::shared_ptr<AMP::Database> db,
                    int max_gcw             = 0,
                    const AMP::AMP_MPI &mpi = AMP::AMP_MPI( AMP_COMM_WORLD ) );

    /*!
     * This function creates the gridding algorithm from the input database,
     * patch hierarchy, and optionally a user provided TagAndIntStrategy.
     *
     * There are several advanced options based on the input file:
     *
     * BoxGeneratorStrategy - This is an optional field that specifies the desired
     * BoxGeneratorStrategy:
     *   "BergerRigoutsos", "TileClustering"
     *
     * LoadBalanceStrategy - This is an optional field that specifies the desired
     * LoadBalanceStrategy:
     *   "TreeLoadBalancer", "ChopAndPackLoadBalancer", "SimpleLoadBalancer"
     *
     * @param hierarchy     Input hierarchy
     * @param amp_db        Input database
     * @param test_object   Optional StandardTagAndInitStrategy object.
     *                      If this is not nullptr on input, the provided StandardTagAndInitStrategy
     *                      will be used.  If it is nullptr on input, then a BogusTagAndInitStrategy
     *                      will be created and a pointer returned.
     */
    static std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> buildGriddingAlgorithm(
        const std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy,
        const std::shared_ptr<AMP::Database> amp_db,
        std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> &test_object );
    /*!
     * This function will check the command line for valid inputs and return the input
     * file name and the log file name.
     * @param argc          Number of input arguments
     * @param argv          Input argument list
     * @param input_file    The input file name
     * @param log_file      The log file name
     */
    static void
    processCommandLine( int argc, char *argv[], std::string &input_file, std::string &log_file );
};

} // namespace AMP::Mesh

#endif
