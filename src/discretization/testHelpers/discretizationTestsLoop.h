#ifndef included_AMP_DiscretizationTestsLoop
#define included_AMP_DiscretizationTestsLoop


#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/UnitTest.h"


namespace AMP::unit_test {


// Function to test subsetting a DOF manager
void testSubsetDOFManager( std::shared_ptr<MeshGenerator> generator,
                           bool split,
                           AMP::UnitTest &ut );

// Function to test the creation/destruction of a simpleDOFManager
void testSimpleDOFManager( std::shared_ptr<MeshGenerator> generator, AMP::UnitTest &ut );

// Function to test the creation/destruction of a multiDOFManager
void testMultiDOFManager( std::shared_ptr<MeshGenerator> generator, AMP::UnitTest &ut );

// Function to test the creation/destruction of a structuredFaceDOFManager
void testStructureDOFManager(
    std::shared_ptr<MeshGenerator> generator, int Nx, int Ny, int Nz, int GCW, AMP::UnitTest &ut );


} // namespace AMP::unit_test


#endif
