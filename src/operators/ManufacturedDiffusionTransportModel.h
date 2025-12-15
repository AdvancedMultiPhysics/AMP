#ifndef included_AMP_ManufacturedDiffusionTransportModel
#define included_AMP_ManufacturedDiffusionTransportModel

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


namespace AMP::Operator {

class ManufacturedDiffusionTransportModel : public DiffusionTransportModel
{
public:
    explicit ManufacturedDiffusionTransportModel(
        std::shared_ptr<const DiffusionTransportModelParameters> params )
        : DiffusionTransportModel( params )
    {
    }

    virtual ~ManufacturedDiffusionTransportModel() {}


    void getTransport( std::vector<double> &result,
                       std::map<std::string, std::shared_ptr<std::vector<double>>> &args,
                       const std::vector<libMesh::Point> &Coordinates ) override
    {
        AMP_ASSERT( ( Coordinates.size() == result.size() ) );
        auto it = args.find( "temperature" );
        AMP_ASSERT( it != args.end() );
        const auto &T = *( it->second );
        AMP_ASSERT( T.size() == result.size() );
        for ( unsigned int qp = 0; qp < Coordinates.size(); qp++ ) {
            double x    = Coordinates[qp]( 0 );
            double y    = Coordinates[qp]( 1 );
            double z    = Coordinates[qp]( 2 );
            double r    = std::sqrt( x * x + y * y + z * z );
            double temp = exp( -r * T[qp] );
            result[qp]  = temp;
        }
    }

protected:
private:
};
} // namespace AMP::Operator

#endif
