#ifndef included_AMP_DiscretizationTests
#define included_AMP_DiscretizationTests


#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/UnitTest.h"


namespace AMP::unit_test {


void testGetDOFIterator( AMP::UnitTest &ut,
                         const AMP::Mesh::MeshIterator &iterator,
                         std::shared_ptr<AMP::Discretization::DOFManager> DOF );


// Function to test some very basic properites
void testBasics( std::shared_ptr<AMP::Discretization::DOFManager> DOF, AMP::UnitTest &ut );


// Test subsetting for different comms
void testSubsetComm( std::shared_ptr<AMP::Discretization::DOFManager> DOF, AMP::UnitTest &ut );


// Test subsetting for different meshes
void testSubsetMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                     std::shared_ptr<AMP::Discretization::DOFManager> DOF,
                     bool is_nodal,
                     int DOFsPerNode,
                     int gcw,
                     AMP::UnitTest &ut );


// Function to test the index conversion given a multDOFManager
void testMultiDOFMap( AMP::UnitTest &ut,
                      std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOF );


// Function to test that a multivector with a DOFManager repeated correctly sets the values
void testMultiDOFVector( AMP::UnitTest &ut, std::shared_ptr<AMP::Discretization::DOFManager> DOF );


// Test SimpleDOFManager / boxMeshDOFManager
void testLogicalDOFMap( AMP::UnitTest &ut );

// Test SimpleDOFManager / boxMeshDOFManager
void testLogicalDOFMap( int ndim, AMP::UnitTest &ut );

// Test SimpleDOFManager / boxMeshDOFManager
void testLogicalDOFMap( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                        bool isboxMesh,
                        AMP::Mesh::GeomType type,
                        int gcw,
                        int DOFsPerElement );


} // namespace AMP::unit_test


#endif
