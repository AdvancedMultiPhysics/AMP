#include "AMP/operators/map/MapOperator.h"
#include "AMP/utils/Utilities.h"

namespace AMP::Operator {


void MapOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );

    AMP_INSIST( myparams, "NULL parameter" );
    AMP_INSIST( myparams->d_db, "NULL database" );
    AMP_INSIST( myparams->d_db->keyExists( "BoundaryId" ), "Key ''tflow_id'' is missing!" );
    d_boundaryId = myparams->d_db->getScalar<int>( "BoundaryId" );
    d_MapComm    = myparams->d_MapComm;
    d_MapMesh    = myparams->d_MapMesh;
    if ( d_MapComm.isNull() && d_MapMesh )
        d_MapComm = d_MapMesh->getComm();
    if ( d_MapComm.isNull() && d_Mesh )
        d_MapComm = d_Mesh->getComm();
    AMP_INSIST( !d_MapComm.isNull(), "NULL communicator" );
}


} // namespace AMP::Operator
