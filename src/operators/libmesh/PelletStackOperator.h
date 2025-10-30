
#ifndef included_AMP_PelletStackOperator
#define included_AMP_PelletStackOperator

#include "AMP/operators/libmesh/PelletStackOperatorParameters.h"

namespace AMP::Operator {

class PelletStackOperator final : public Operator
{
public:
    explicit PelletStackOperator( std::shared_ptr<const PelletStackOperatorParameters> params );

    virtual ~PelletStackOperator() {}

    std::string type() const override { return "PelletStackOperator"; }

    int getLocalIndexForPellet( unsigned int pellId );

    auto getTotalNumberOfPellets() { return d_totalNumberOfPellets; }

    const auto &getLocalMeshes() { return d_meshes; }

    const auto &getLocalPelletIds() { return d_pelletIds; }

    bool useSerial() { return d_useSerial; }

    bool onlyZcorrection() { return d_onlyZcorrection; }

    bool useScaling() { return d_useScaling; }

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    void applyUnscaling( AMP::LinearAlgebra::Vector::shared_ptr f );

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

protected:
    void applySerial( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr &f );

    void applyOnlyZcorrection( AMP::LinearAlgebra::Vector::shared_ptr &u );

    void applyXYZcorrection( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr &f );

    void computeZscan( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                       std::vector<double> &finalMaxZdispsList );

    unsigned int d_totalNumberOfPellets;
    unsigned int d_currentPellet;
    bool d_useSerial;
    bool d_onlyZcorrection;
    bool d_useScaling;
    double d_scalingFactor;
    short int d_masterId;
    short int d_slaveId;
    std::vector<std::shared_ptr<AMP::Mesh::Mesh>> d_meshes;
    std::vector<unsigned int> d_pelletIds;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_var;
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVectorForMaps;
    bool d_frozenVectorSet;
    AMP_MPI d_pelletStackComm;
    std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> d_n2nMaps;
};
} // namespace AMP::Operator

#endif
