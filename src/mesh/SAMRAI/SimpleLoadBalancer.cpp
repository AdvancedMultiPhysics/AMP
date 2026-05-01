#include "AMP/mesh/SAMRAI/SimpleLoadBalancer.h"
#include "AMP/utils/UtilityMacros.h"

#include "SAMRAI/hier/BoxLevelConnectorUtils.h"
#include "SAMRAI/hier/Connector.h"
#include "SAMRAI/hier/OverlapConnectorAlgorithm.h"
#include "SAMRAI/pdat/CellDataFactory.h"

#include <algorithm>


namespace AMP::Mesh {


// Function to find the index containing the minimum value
static inline size_t minIndex( const std::vector<size_t> &size )
{
    size_t i = 0;
    size_t v = size[0];
    for ( size_t j = 1; j < size.size(); j++ ) {
        if ( size[j] < v ) {
            v = size[j];
            i = j;
        }
    }
    return i;
}


// Function to compare the size of two boxes
static bool compare_box_size( const SAMRAI::hier::Box &i, const SAMRAI::hier::Box &j )
{
    return i.size() < j.size();
}


// Function to split N cells as evenly as possible within the min/max contraints
int splitCells( int N, int min, int max )
{
    if ( N < min )
        AMP_ERROR( "Internal error" );
    if ( N <= max )
        return N;
    double best = 1e100;
    int val     = N;
    for ( int i = max; i >= min; i-- ) {
        int N1         = i;
        int N2         = N % N1;
        double quality = static_cast<double>( std::abs( N1 - N2 ) ) / static_cast<double>( N1 );
        if ( N2 == 0 )
            quality = 0;
        else if ( N2 < min || N2 > max )
            quality = 1e100;
        if ( quality < best ) {
            best = quality;
            val  = i;
        }
    }
    return val;
}


// Function to split a box
static std::vector<SAMRAI::hier::Box> splitBox( const SAMRAI::hier::Box &box,
                                                const SAMRAI::hier::IntVector &min_size,
                                                const SAMRAI::hier::IntVector &max_size )
{
    int lower[3] = { 0, 0, 0 };
    int upper[3] = { 0, 0, 0 };
    int min[3]   = { 1, 1, 1 };
    int max[3]   = { 1, 1, 1 };
    auto dim     = box.getDim();
    for ( int d = 0; d < dim.getValue(); d++ ) {
        lower[d] = box.lower( d );
        upper[d] = box.upper( d );
        min[d]   = min_size( d );
        max[d]   = max_size( d );
    }
    int N[3]  = {};
    int Nb[3] = {};
    for ( int d = 0; d < 3; d++ ) {
        int Nd = upper[d] - lower[d] + 1;
        N[d]   = splitCells( Nd, min[d], max[d] );
        Nb[d]  = ( Nd + N[d] - 1 ) / N[d];
    }
    std::vector<SAMRAI::hier::Box> box_list;
    box_list.reserve( Nb[0] * Nb[1] * Nb[2] );
    for ( int i = 0; i < Nb[0]; i++ ) {
        for ( int j = 0; j < Nb[1]; j++ ) {
            for ( int k = 0; k < Nb[2]; k++ ) {
                int lb[3] = { lower[0] + i * N[0], lower[1] + j * N[1], lower[2] + k * N[2] };
                int ub[3] = { std::min( lb[0] + N[0] - 1, upper[0] ),
                              std::min( lb[1] + N[1] - 1, upper[1] ),
                              std::min( lb[2] + N[2] - 1, upper[2] ) };
                SAMRAI::hier::Box box2( box );
                box2.setLower( SAMRAI::hier::Index( dim, lb ) );
                box2.setUpper( SAMRAI::hier::Index( dim, ub ) );
                box_list.push_back( box2 );
            }
        }
    }
    return box_list;
}


// Constructor
SimpleLoadBalancer::SimpleLoadBalancer( SAMRAI::tbox::Dimension,
                                        const std::string &,
                                        const std::shared_ptr<SAMRAI::tbox::Database> )
{
}


// Destructor
SimpleLoadBalancer::~SimpleLoadBalancer() = default;


// loadBalanceBoxLevel
void SimpleLoadBalancer::loadBalanceBoxLevel(
    SAMRAI::hier::BoxLevel &balance_box_level,
    SAMRAI::hier::Connector *balance_to_anchor,
    const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
    const int,
    const SAMRAI::hier::IntVector &min_size,
    const SAMRAI::hier::IntVector &max_size,
    const SAMRAI::hier::BoxLevel &,
    const SAMRAI::hier::IntVector &,
    const SAMRAI::hier::IntVector &,
    const SAMRAI::tbox::RankGroup & ) const
{
    // Get the global boxes
    auto mpi         = balance_box_level.getMPI();
    auto globalBoxes = balance_box_level.getGlobalizedVersion().getGlobalBoxes();
    globalBoxes.removePeriodicImageBoxes();

    // Check for small boxes and break larger boxes
    std::vector<SAMRAI::hier::Box> boxes;
    boxes.reserve( globalBoxes.size() + 10 );
    for ( auto it = globalBoxes.begin(); it != globalBoxes.end(); ++it ) {
        auto box  = *it;
        auto size = box.numberCells();
        if ( size < min_size ) {
            // Box is < min_size, throw an error
            AMP_ERROR( "Small box detected" );
        } else if ( size <= max_size ) {
            // Box is a valid size, add it to the list
            boxes.push_back( box );
        } else {
            // Box is larger than the max_size, break it and add the boxes
            auto new_boxes = splitBox( box, min_size, max_size );
            for ( auto &new_boxe : new_boxes )
                boxes.push_back( new_boxe );
        }
    }

    // Check that we conserved the total number of cells
    size_t N1 = balance_box_level.getGlobalNumberOfCells();
    size_t N2 = 0;
    for ( auto &boxe : boxes )
        N2 += boxe.size();
    AMP_INSIST( N1 == N2, "Cells were not conserved" );

    // Sort the boxes (largest first)
    std::stable_sort( boxes.begin(), boxes.end(), compare_box_size );

    // Distribute the boxes among the processors (keeping only my boxes)
    size_t size = mpi.getSize();
    size_t rank = mpi.getRank();
    std::vector<size_t> N_cells( size, 0 );
    SAMRAI::hier::BoxContainer my_boxes;
    for ( size_t i = 0, j = 0; i < boxes.size(); i++ ) {
        size_t k = minIndex( N_cells );
        N_cells[k] += boxes[i].size();
        if ( k == rank ) {
            auto box = boxes[i];
            box.setId( SAMRAI::hier::BoxId( SAMRAI::hier::LocalId( j ), rank ) );
            my_boxes.pushBack( box );
            j++;
        }
    }
    boxes.clear();

    // Update the box level with the new boxes
    auto ratio = balance_box_level.getRefinementRatio();
    auto geom  = balance_box_level.getGridGeometry();
    balance_box_level.initialize( my_boxes, ratio, geom, mpi, SAMRAI::hier::BoxLevel::DISTRIBUTED );
    balance_box_level.finalize();
    SAMRAI::hier::BoxLevelConnectorUtils blcu;
    auto one = SAMRAI::hier::IntVector::getOne( ratio.getDim() );
    blcu.addPeriodicImages(
        balance_box_level,
        hierarchy->getGridGeometry()->getDomainSearchTree(),
        SAMRAI::hier::IntVector::max( one, hierarchy->getRequiredConnectorWidth( 0, 0, true ) ) );
    balance_box_level.finalize();
    balance_box_level.cacheGlobalReducedData();
    balance_box_level.setParallelState( SAMRAI::hier::BoxLevel::GLOBALIZED );

    // Reinitialize Connectors due to changed balance_box_level.
    if ( balance_to_anchor ) {
        auto &anchor_to_balance = balance_to_anchor->getTranspose();
        balance_to_anchor->clearNeighborhoods();
        balance_to_anchor->setBase( balance_box_level, true );
        anchor_to_balance.clearNeighborhoods();
        anchor_to_balance.setHead( balance_box_level, true );
        SAMRAI::hier::OverlapConnectorAlgorithm oca;
        oca.findOverlaps( *balance_to_anchor );
        oca.findOverlaps( anchor_to_balance, balance_box_level );
        balance_to_anchor->removePeriodicRelationships();
        anchor_to_balance.removePeriodicRelationships();
    }
}

void SimpleLoadBalancer::setWorkloadPatchDataIndex( int data_id, int level_number )
{

    if ( level_number >= 0 ) {
        auto asize = static_cast<int>( d_workload_data_id.size() );
        if ( asize < level_number + 1 ) {
            d_workload_data_id.resize( level_number + 1 );
            for ( int i = asize; i < level_number - 1; ++i ) {
                d_workload_data_id[i] = d_master_workload_data_id;
            }
            d_workload_data_id[level_number] = data_id;
        }
    } else {
        d_master_workload_data_id = data_id;
        for ( int &ln : d_workload_data_id ) {
            ln = d_master_workload_data_id;
        }
    }
}


} // namespace AMP::Mesh
