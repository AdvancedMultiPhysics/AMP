// Test coarsening of SAMRAI hierarchies
#include "AMP/IO/PIO.h"
#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRBuilder.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include "SAMRAI/SAMRAI_config.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"

#include <string>
#include <vector>


// Run all the tests
void run_tests( const std::string &input_file, AMP::UnitTest &ut )
{
    AMP::AMP_MPI mpi( AMP_COMM_WORLD );

    // Create input database and parse all data in input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
    input_db->print( AMP::plog );

    // Create the patch hierarchy
    std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> taginit_object;
    std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> gridding_algorithm;
    auto samrHierarchy =
        AMP::Mesh::SAMRBuilder::buildHierarchy( input_db, taginit_object, gridding_algorithm, 2 );
    auto hierarchy_adaptor =
        std::dynamic_pointer_cast<AMP::Mesh::SAMRAIHierarchyAdaptor>( samrHierarchy );
    AMP_ASSERT( hierarchy_adaptor );
    auto hierarchy = std::dynamic_pointer_cast<SAMRAI::hier::PatchHierarchy>(
        hierarchy_adaptor->getSAMRAIHierarchy() );
    AMP_ASSERT( hierarchy );

    // Print the hierarchy
    AMP::plog << "Initial hierarchy:\n";
    hierarchy->recursivePrint( AMP::plog, "   ", 3 );

    // Create a coarsened patch hierarchy from boxlevels on current hierarchy
    // new/old hierarchies
    auto finest_ln    = hierarchy->getFinestLevelNumber();
    auto ratio        = hierarchy->getRatioToCoarserLevel( finest_ln );
    auto fineGeometry = hierarchy->getGridGeometry();
    std::string name{ "CoarseHierarchy" };
    auto coarseGeometry  = fineGeometry->makeCoarsenedGridGeometry( name + "GridGeometry", ratio );
    auto coarseHierarchy = std::make_shared<SAMRAI::hier::PatchHierarchy>(
        name, coarseGeometry, std::shared_ptr<SAMRAI::tbox::Database>() );

    coarseHierarchy->setMaxNumberOfLevels( hierarchy->getMaxNumberOfLevels() );
    std::vector<SAMRAI::hier::IntVector> ratio_to_coarser( hierarchy->getMaxNumberOfLevels(),
                                                           ratio );
    coarseGeometry->setUpRatios( ratio_to_coarser );

    for ( int ln = 1; ln <= finest_ln; ++ln ) {
        coarseHierarchy->setRatioToCoarserLevel( ratio, ln );
    }
    for ( int ln = 0; ln <= finest_ln; ++ln ) {
        auto patchLevel = hierarchy->getPatchLevel( ln );
        auto boxLevel   = patchLevel->getBoxLevel();
        auto coarsenedBoxes( boxLevel->getBoxes() );
        coarsenedBoxes.coarsen( ratio );
        auto coarsenedBoxLevel = std::make_shared<SAMRAI::hier::BoxLevel>(
            boxLevel->getRefinementRatio(), coarseGeometry, boxLevel->getMPI() );
        coarsenedBoxLevel->initialize(
            coarsenedBoxes, boxLevel->getRefinementRatio(), coarseGeometry, boxLevel->getMPI() );
        coarseHierarchy->makeNewPatchLevel( ln, coarsenedBoxLevel );
    }

    for ( int ln = 0; ln <= finest_ln; ++ln ) {
        auto constructedLevel = coarseHierarchy->getPatchLevel( ln );
        auto fineLevel        = hierarchy->getPatchLevel( ln );
        auto restrictedLevel  = std::make_shared<SAMRAI::hier::PatchLevel>( hierarchy->getDim() );
        restrictedLevel->setCoarsenedPatchLevel( fineLevel, ratio );

        auto restrictedBoxes  = restrictedLevel->getBoxLevel()->getBoxes();
        auto constructedBoxes = constructedLevel->getBoxLevel()->getBoxes();

        if ( restrictedBoxes.size() != constructedBoxes.size() ) {

            ut.failure( "Coarsen of hierarchy fails to preserve number of boxes for " +
                        input_file );

        } else {

            auto it1 = restrictedBoxes.begin();
            auto it2 = constructedBoxes.begin();

            while ( it1 != restrictedBoxes.end() && it2 != constructedBoxes.end() ) {
                if ( !( *it1 ).isSpatiallyEqual( *it2 ) ) {
                    ut.failure( "Coarsen of hierarchy fails to preserve boxes for " + input_file );
                    break;
                }
                ++it1;
                ++it2;
            }
            ut.passes( "Coarsen of hierarchy preserves boxes for " + input_file );
        }
    }
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );

    std::vector<std::string> inputs;
    // Process command line arguments
    if ( argc > 1 ) {
        if ( ( argc != 2 ) ) {
            AMP::pout << "USAGE:  " << argv[0] << " <input file> " << std::endl;
            exit( -1 );
        } else {
            inputs.push_back( argv[1] );
        }
    } else {
        inputs.push_back( "input_3d_patch_1d" );
        inputs.push_back( "input_3d_2level" );
        inputs.push_back( "input_3d_2level_2" );
        inputs.push_back( "input_3d_2level_3" );
        inputs.push_back( "input_3d_32_2level" );
        inputs.push_back( "input_3d_128_2level" );
        inputs.push_back( "input_3d_bug1" );
        inputs.push_back( "input_3d_bug2" );
        inputs.push_back( "input_3d_wall" );
        inputs.push_back( "input_3d_nrdf_16b4l" );
        inputs.push_back( "input_3d_3level_coarsen_bug" );
        inputs.push_back( "input_3d_7level_coarsen_bug" );
    }

    AMP::UnitTest ut;

    for ( const auto &input_file : inputs )
        run_tests( input_file, ut );

    auto N_errors = ut.NumFailGlobal();
    ut.report();

    // That's all, folks!
    AMP::AMPManager::shutdown();
    return N_errors;
}
