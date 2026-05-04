#ifndef included_AMP_SimpleLoadBalancer
#define included_AMP_SimpleLoadBalancer

#include "SAMRAI/mesh/LoadBalanceStrategy.h"

#include <vector>

namespace AMP::Mesh {


/*!
 * @brief Provides load balancing routines for AMR hierarchy
 *
 * @see mesh::LoadBalanceStrategy
 */
class SimpleLoadBalancer : public SAMRAI::mesh::LoadBalanceStrategy
{
public:
    /*!
     * @brief Initializing constructor sets object state to default or,
     * if database provided, to parameters in database.
     *
     * @param[in] dim
     *
     * @param[in] name User-defined identifier used for error reporting
     * and timer names.
     *
     * @param[in] input_db (optional) database pointer providing
     * parameters from input file.  This pointer may be null indicating
     * no input is used.
     *
     * @pre !name.empty()
     */
    SimpleLoadBalancer( SAMRAI::tbox::Dimension dim,
                        const std::string &name,
                        std::shared_ptr<SAMRAI::tbox::Database> input_db = {} );

    //! Destructor
    virtual ~SimpleLoadBalancer();

    virtual void loadBalanceBoxLevel(
        SAMRAI::hier::BoxLevel &balance_box_level,
        SAMRAI::hier::Connector *balance_to_anchor,
        const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
        const int level_number,
        const SAMRAI::hier::IntVector &min_size,
        const SAMRAI::hier::IntVector &max_size,
        const SAMRAI::hier::BoxLevel &domain_box_level,
        const SAMRAI::hier::IntVector &bad_interval,
        const SAMRAI::hier::IntVector &cut_factor,
        const SAMRAI::tbox::RankGroup &rank_group = SAMRAI::tbox::RankGroup() ) const;

    /*!
     * @brief Return true if load balancing procedure for given level
     * depends on patch data on mesh; otherwise return false.
     *
     * @param[in] level_number  Integer patch level number.
     */
    virtual bool getLoadBalanceDependsOnPatchData( [[maybe_unused]] int level_number ) const
    {
        return false;
    }

    void setWorkloadPatchDataIndex( int data_id, int level_number );

private:
    /*
     * Values for workload estimate data, workload factor, and bin pack method
     * that will be used for all levels unless specified for individual levels.
     */
    int d_master_workload_data_id;
    /*
     * Values for workload estimate data, workload factor, and bin pack method
     * used on individual levels when specified as such.
     */
    std::vector<int> d_workload_data_id;
};

} // namespace AMP::Mesh

#endif
