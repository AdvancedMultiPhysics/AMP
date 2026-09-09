#ifndef included_AMP_IntergridRedist
#define included_AMP_IntergridRedist

#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/GroupedRedistributionPlan.h"

namespace AMP::Solver::AMG {

struct IntergridRedist : AMP::Operator::LinearOperator {
    using redist_context = Utilities::GroupedRedistributionPlan;
    enum class direction { up, down };

    IntergridRedist( std::shared_ptr<AMP::Operator::OperatorParameters> params,
                     direction dir,
                     const redist_context &ctx,
                     std::shared_ptr<LinearAlgebra::Vector> interface_in  = nullptr,
                     std::shared_ptr<LinearAlgebra::Vector> interface_out = nullptr );
    IntergridRedist( std::shared_ptr<AMP::Operator::OperatorParameters> params,
                     direction dir,
                     const redist_context &ctx,
                     std::shared_ptr<AMP::Operator::Operator> transfer,
                     std::shared_ptr<LinearAlgebra::Vector> interface_in  = nullptr,
                     std::shared_ptr<LinearAlgebra::Vector> interface_out = nullptr );

    virtual void apply( std::shared_ptr<const LinearAlgebra::Vector> x,
                        std::shared_ptr<LinearAlgebra::Vector> b ) override;

    std::shared_ptr<LinearAlgebra::Vector> createInputVector() const override;
    std::shared_ptr<LinearAlgebra::Vector> createOutputVector() const override;

private:
    direction d_direction;
    redist_context d_redist_context;
    std::shared_ptr<AMP::Operator::Operator> d_transfer;
    std::shared_ptr<LinearAlgebra::Vector> d_interface_in;
    std::shared_ptr<LinearAlgebra::Vector> d_interface_out;
    std::shared_ptr<LinearAlgebra::Vector> tmp;
};

} // namespace AMP::Solver::AMG

#endif
