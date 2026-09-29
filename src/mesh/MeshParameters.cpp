#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
MeshParameters::MeshParameters() : comm( AMP_COMM_WORLD ), MAX_GCW_WIDTH( 1 ) {}
MeshParameters::MeshParameters( std::shared_ptr<AMP::Database> db )
    : d_db( db ), comm( AMP_COMM_WORLD ), MAX_GCW_WIDTH( 1 )
{
}


/********************************************************
 * De-constructor                                        *
 ********************************************************/
MeshParameters::~MeshParameters() = default;


/********************************************************
 * Set the desired communicator                          *
 ********************************************************/
void MeshParameters::setComm( const AMP::AMP_MPI &comm_in ) { comm = comm_in; }


/********************************************************
 * Return the database                                   *
 ********************************************************/
std::shared_ptr<AMP::Database> MeshParameters::getDatabase() { return d_db; }
std::shared_ptr<const AMP::Database> MeshParameters::getDatabase() const { return d_db; }


} // namespace AMP::Mesh
