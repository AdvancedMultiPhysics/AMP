#ifndef included_AMP_ManufacturedDiffusionTransportModel
#define included_AMP_ManufacturedDiffusionTransportModel

#include <map>
#include <string>
#include <vector>

#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/operators/diffusion/DiffusionConstants.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/utils/shared_ptr.h"


namespace AMP {
namespace Operator {

class ManufacturedDiffusionTransportModel : public DiffusionTransportModel
{
public:
    explicit ManufacturedDiffusionTransportModel(
        const AMP::shared_ptr<DiffusionTransportModelParameters> &params )
        : DiffusionTransportModel( params )
    {
    }

    virtual ~ManufacturedDiffusionTransportModel() {}


    virtual void getTransport( std::vector<double> &result,
                               std::map<std::string, AMP::shared_ptr<std::vector<double>>> &args,
                               const std::vector<libMesh::Point> &Coordinates )
    {
        AMP_ASSERT( ( Coordinates.size() == result.size() ) );
        auto it = args.find( "temperature" );
        AMP_ASSERT( it != args.end() );
        std::vector<double> &T( *( it->second ) );
        AMP_ASSERT( T.size() == result.size() );

        for ( unsigned int qp = 0; qp < Coordinates.size(); qp++ ) {
            double x = Coordinates[qp]( 0 );
            double y = Coordinates[qp]( 1 );
            double z = Coordinates[qp]( 2 );
            double r = sqrt( x * x + y * y + z * z );

            double temp = exp( -r * T[qp] );

            result[qp] = temp;
        }
    }

protected:
private:
};
} // namespace Operator
} // namespace AMP

#endif
