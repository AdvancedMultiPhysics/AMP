#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

// This test creates a few small local CSR matrices and concatenates them
// the result is compared entrywise with what should have been created

template<typename Config>
std::map<int, std::shared_ptr<AMP::LinearAlgebra::CSRLocalMatrixData<Config>>>
vertBlocks( std::vector<typename Config::lidx_t> &row_starts,
            std::vector<typename Config::gidx_t> &cols,
            std::vector<typename Config::scalar_t> &coeffs )
{
    using Allocator = typename Config::allocator_type;
    using lidx_t    = typename Config::lidx_t;
    using gidx_t    = typename Config::gidx_t;
    using scalar_t  = typename Config::scalar_t;

    using LocalData    = AMP::LinearAlgebra::CSRLocalMatrixData<Config>;
    const auto mem_loc = AMP::Utilities::getAllocatorMemoryType<Allocator>();

    std::map<int, std::shared_ptr<LocalData>> blocks;

    // create a few blocks and record entries as they get built
    // a little tedious, but useful nevertheless

    // first block, 3x10, diag plus last element in each row
    auto it =
        blocks.insert( { 1, std::make_shared<LocalData>( nullptr, mem_loc, 0, 3, 0, 10, true ) } )
            .first;
    auto block = ( *it ).second;
    lidx_t *rs_b, *cols_loc_b;
    gidx_t *cols_b;
    scalar_t *coeffs_b;
    rs_b = block->getRowStarts();

    row_starts.push_back( 2 );
    rs_b[0] = row_starts.back();
    row_starts.push_back( 2 );
    rs_b[1] = row_starts.back();
    row_starts.push_back( 2 );
    rs_b[2] = row_starts.back();

    block->setNNZ( true );
    std::tie( rs_b, cols_b, cols_loc_b, coeffs_b ) = block->getDataFields();

    cols.push_back( 0 );
    cols_b[0] = cols.back();
    cols.push_back( 9 );
    cols_b[1] = cols.back();
    cols.push_back( 1 );
    cols_b[2] = cols.back();
    cols.push_back( 9 );
    cols_b[3] = cols.back();
    cols.push_back( 2 );
    cols_b[4] = cols.back();
    cols.push_back( 9 );
    cols_b[5] = cols.back();

    coeffs.push_back( 1.0 );
    coeffs_b[0] = coeffs.back();
    coeffs.push_back( 2.0 );
    coeffs_b[1] = coeffs.back();
    coeffs.push_back( 3.0 );
    coeffs_b[2] = coeffs.back();
    coeffs.push_back( 4.0 );
    coeffs_b[3] = coeffs.back();
    coeffs.push_back( 5.0 );
    coeffs_b[4] = coeffs.back();
    coeffs.push_back( 6.0 );
    coeffs_b[5] = coeffs.back();

    // second block, 2x10, 1's in column 7
    it = blocks.insert( { 3, std::make_shared<LocalData>( nullptr, mem_loc, 0, 2, 0, 10, true ) } )
             .first;
    block = ( *it ).second;
    rs_b  = block->getRowStarts();

    row_starts.push_back( 1 );
    rs_b[0] = row_starts.back();
    row_starts.push_back( 1 );
    rs_b[1] = row_starts.back();

    block->setNNZ( true );
    std::tie( rs_b, cols_b, cols_loc_b, coeffs_b ) = block->getDataFields();

    cols.push_back( 7 );
    cols_b[0] = cols.back();
    cols.push_back( 7 );
    cols_b[1] = cols.back();

    coeffs.push_back( 1.0 );
    coeffs_b[0] = coeffs.back();
    coeffs.push_back( 1.0 );
    coeffs_b[1] = coeffs.back();

    // third block, same as first with coeffs negated
    it = blocks.insert( { 17, std::make_shared<LocalData>( nullptr, mem_loc, 0, 3, 0, 10, true ) } )
             .first;
    block = ( *it ).second;
    rs_b  = block->getRowStarts();

    row_starts.push_back( 2 );
    rs_b[0] = row_starts.back();
    row_starts.push_back( 2 );
    rs_b[1] = row_starts.back();
    row_starts.push_back( 2 );
    rs_b[2] = row_starts.back();

    block->setNNZ( true );
    std::tie( rs_b, cols_b, cols_loc_b, coeffs_b ) = block->getDataFields();

    cols.push_back( 0 );
    cols_b[0] = cols.back();
    cols.push_back( 9 );
    cols_b[1] = cols.back();
    cols.push_back( 1 );
    cols_b[2] = cols.back();
    cols.push_back( 9 );
    cols_b[3] = cols.back();
    cols.push_back( 2 );
    cols_b[4] = cols.back();
    cols.push_back( 9 );
    cols_b[5] = cols.back();

    coeffs.push_back( 1.0 );
    coeffs_b[0] = coeffs.back();
    coeffs.push_back( 2.0 );
    coeffs_b[1] = coeffs.back();
    coeffs.push_back( 3.0 );
    coeffs_b[2] = coeffs.back();
    coeffs.push_back( 4.0 );
    coeffs_b[3] = coeffs.back();
    coeffs.push_back( 5.0 );
    coeffs_b[4] = coeffs.back();
    coeffs.push_back( 6.0 );
    coeffs_b[5] = coeffs.back();

    // the running record of row_starts needs to be accumulated
    // to match what should come from the concatenation process
    row_starts.push_back( 0 );
    std::exclusive_scan( row_starts.begin(), row_starts.end(), row_starts.begin(), 0 );

    return blocks;
}

void testVertical( AMP::UnitTest *ut )
{
    using DefaultCSRConfig = AMP::LinearAlgebra::DefaultHostCSRConfig;
    using lidx_t           = typename DefaultCSRConfig::lidx_t;
    using gidx_t           = typename DefaultCSRConfig::gidx_t;
    using scalar_t         = typename DefaultCSRConfig::scalar_t;

    // get the blocks to concatenate
    std::vector<lidx_t> row_starts;
    std::vector<gidx_t> cols;
    std::vector<scalar_t> coeffs;
    auto blocks = vertBlocks<DefaultCSRConfig>( row_starts, cols, coeffs );

    // form concatenated matrix
    auto cat = AMP::LinearAlgebra::CSRLocalMatrixData<DefaultCSRConfig>::ConcatVertical( blocks );

    // test overall dimensions of matrix, and total NNZ
    bool shape_pass = true;
    if ( cat->beginRow() == 0 && cat->endRow() == 8 && cat->beginCol() == 0 &&
         cat->endCol() == 10 ) {
        ut->passes( "Vertical concatenate matches dimensions" );
    } else {
        shape_pass = false;
        AMP::pout << "Got row extents [" << cat->beginRow() << "," << cat->endRow()
                  << "), expected [0,8)" << std::endl;
        AMP::pout << "Got columns extents [" << cat->beginCol() << "," << cat->endCol()
                  << "), expected [0,10)" << std::endl;
        ut->failure( "Vertical concatenate does not match dimensions" );
    }

    const auto tot_nnz = static_cast<lidx_t>( cols.size() );
    if ( cat->numberOfNonZeros() == tot_nnz ) {
        ut->passes( "Vertical concatenate matches NNZ" );
    } else {
        shape_pass = false;
        AMP::pout << "Got " << cat->numberOfNonZeros() << " total NNZ, expected " << tot_nnz
                  << std::endl;
        ut->failure( "Vertical concatenate does not match NNZ" );
    }

    // fail early if shape doesn't match
    if ( !shape_pass ) {
        ut->failure( "Vertical concatenate can not match entries" );
        return;
    }

    // pull out fields and compare entrywise
    auto [cat_rs, cat_cols, cat_cols_loc, cat_coeffs] = cat->getDataFields();

    bool entries_pass = true;
    for ( gidx_t row = 0; row <= cat->endRow(); row++ ) {
        if ( cat_rs[row] != row_starts[row] ) {
            entries_pass = false;
            break;
        }
    }
    if ( !entries_pass ) {
        ut->failure( "Vertical concatenate does not match entries (row starts)" );
        return;
    }

    for ( gidx_t n = 0; n < tot_nnz; n++ ) {
        if ( cat_cols[n] != cols[n] ) {
            entries_pass = false;
            break;
        }
    }
    if ( !entries_pass ) {
        ut->failure( "Vertical concatenate does not match entries (cols)" );
        return;
    }

    for ( gidx_t n = 0; n < tot_nnz; n++ ) {
        if ( !AMP::Utilities::approx_equal( cat_coeffs[n], coeffs[n] ) ) {
            entries_pass = false;
            break;
        }
    }
    if ( !entries_pass ) {
        ut->failure( "Vertical concatenate does not match entries (coeffs)" );
        return;
    }

    ut->passes( "Vertical concatenate matches entries" );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    testVertical( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
