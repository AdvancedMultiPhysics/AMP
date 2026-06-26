#ifndef included_AMP_GroupedRedistributionPlan
#define included_AMP_GroupedRedistributionPlan

#include "AMP/utils/AMP_MPI.h"

#include <cstdint>

namespace AMP::Utilities {


class GroupedRedistributionPlan
{
public:
    GroupedRedistributionPlan() = default;

    GroupedRedistributionPlan( const AMP_MPI &comm, int new_nprocs )
        : d_parent_comm( comm ), d_new_nprocs( new_nprocs )
    {
        AMP_INSIST( !comm.isNull(), "Grouped redistribution requires a valid communicator" );
        AMP_INSIST( new_nprocs > 0 && new_nprocs <= comm.getSize(),
                    "Grouped redistribution invalid processor count" );

        const int parent_rank = comm.getRank();
        const int parent_size = comm.getSize();
        d_group_color =
            ( new_nprocs == parent_size ) ?
                parent_rank :
                static_cast<int>( ( static_cast<std::int64_t>( parent_rank ) * new_nprocs ) /
                                  parent_size );
        d_group_comm   = comm.split( d_group_color, parent_rank );
        d_is_root      = d_group_comm.getRank() == 0;
        d_reduced_comm = comm.split( d_is_root ? 0 : -1, parent_rank );
    }

    int numRanks() const { return d_new_nprocs; }
    int groupColor() const { return d_group_color; }
    bool isActive() const { return d_is_root; }

    const AMP_MPI &parentComm() const { return d_parent_comm; }
    const AMP_MPI &groupComm() const { return d_group_comm; }
    const AMP_MPI &reducedComm() const { return d_reduced_comm; }

private:
    AMP_MPI d_parent_comm;
    int d_new_nprocs  = 0;
    int d_group_color = -1;
    bool d_is_root    = false;
    AMP_MPI d_group_comm;
    AMP_MPI d_reduced_comm;
};

inline GroupedRedistributionPlan createGroupedRedistributionPlan( const AMP_MPI &comm,
                                                                  int new_nprocs )
{
    return GroupedRedistributionPlan( comm, new_nprocs );
}

} // namespace AMP::Utilities

#endif
