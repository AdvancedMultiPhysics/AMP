#ifndef included_AMP_DiffusionTransportModel
#define included_AMP_DiffusionTransportModel

#include "AMP/materials/Material.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/ElementPhysicsModel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/point.h"
ENABLE_WARNINGS


namespace AMP::Operator {

typedef ElementPhysicsModelParameters DiffusionTransportModelParameters;

class DiffusionTransportModel : public ElementPhysicsModel
{
public:
    explicit DiffusionTransportModel(
        std::shared_ptr<const DiffusionTransportModelParameters> params );

    virtual ~DiffusionTransportModel() {}

    // For LinearOperator's Reset/Init

    virtual void preLinearAssembly() {}

    virtual void postLinearAssembly() {}

    virtual void preLinearElementOperation() {}

    virtual void postLinearElementOperation() {}

    virtual void preLinearGaussPointOperation() {}

    virtual void postLinearGaussPointOperation() {}

    // For NonlinearOperator's Init

    virtual void preNonlinearInitElementOperation() {}

    virtual void postNonlinearInitElementOperation() {}

    virtual void nonlinearInitGaussPointOperation() {}

    // For NonlinearOperator's Apply

    virtual void preNonlinearAssembly() {}

    virtual void postNonlinearAssembly() {}

    virtual void preNonlinearAssemblyElementOperation() {}

    virtual void postNonlinearAssemblyElementOperation() {}

    virtual void preNonlinearAssemblyGaussPointOperation() {}

    virtual void postNonlinearAssemblyGaussPointOperation() {}

    // For NonlinearOperator's Reset

    virtual void globalReset() {}

    virtual void preNonlinearReset() {}

    virtual void postNonlinearReset() {}

    virtual void preNonlinearResetElementOperation() {}

    virtual void postNonlinearResetElementOperation() {}

    virtual void preNonlinearResetGaussPointOperation() {}

    virtual void postNonlinearResetGaussPointOperation() {}

    virtual void nonlinearResetGaussPointOperation( const double * ) {}

    // For NonlinearOperator's GetJacobianParameters

    virtual void preNonlinearJacobian() {}

    virtual void postNonlinearJacobian() {}

    virtual void preNonlinearJacobianElementOperation() {}

    virtual void postNonlinearJacobianElementOperation() {}

    virtual void preNonlinearJacobianGaussPointOperation() {}

    virtual void postNonlinearJacobianGaussPointOperation() {}

    virtual void nonlinearJacobianGaussPointOperation( const double * ) {}

    static std::shared_ptr<std::vector<double>>
    bilogTransform( const std::vector<double> &u, const double a, const double b );

    static void bilogScale( std::vector<double> &u, const double a, const double b );

    virtual void getTransport( std::vector<double> &result,
                               std::map<std::string, std::shared_ptr<std::vector<double>>> &args,
                               const std::vector<libMesh::Point> &Coordinates = d_DummyCoords );

    std::shared_ptr<AMP::Materials::Material> getMaterial() { return d_material; }
    std::shared_ptr<AMP::Materials::Property> getProperty() { return d_property; }

    bool isaTensor() { return d_IsTensor; }

protected:
    std::shared_ptr<AMP::Materials::Material> d_material;

    std::shared_ptr<AMP::Materials::Property> d_property;

    /**
     * \brief Use a bilogarithmic scaling of material arguments
     *
     * The chemical diffusion equation is
     * \f[ \partial_t u = \nabla \cdot (K \nabla u). \f]
     * Sometimes the variable \f$u\f$ is physically restricted to be within a certain range, such as
     * \f$ u \in [0,.2] \f$ for hyper non-stoichiometry. These bounds can wreak havoc with the
     * convergence of nonlinear solvers. We can employ the transformation \f$ U=\log( (u-a)/(b-u) )
     * \f$ or
     * \f$ u = (a+ b \exp(U)/(1+\exp(U)) \f$ to change to a variable \f$U\f$ that ranges
     * from \f$(-\infty, \infty)\f$. For any value of \f$ U \f$ it is rigorously true that \f$ a<u<b
     * \f$,
     * alleviating the problem of solver iterations sending the solution into forbidden territory.
     * The equation for \f$U\f$ is
     * \f[ f(U) \partial_t U = \nabla \cdot (f(U) K \nabla U), \f]
     * where \f$ f(U) = (b-a) \exp(U) / (1+\exp(U))^2 \f$.
     * This adds a well-behaved mass matrix to the chemical diffusion equation and modifies the
     * diffusion coefficient by
     * the same factor.
     * When using this transformation be sure to transform initial and boundary conditions as well
     * in the input files.
     * You have to specify this transformation in the mass operator for the chemical equation.
     * The transformation must be specified within all operators using materials that depend on the
     * transformed
     * variable.
     * When such an operator acts on a variable other than BilogVariable,
     * you must specify "BilogScaleCoefficient=FALSE".
     * Be sure to also to transform the default variables specified for this
     * DiffusionTransportModel.
     *
     * The thermal diffusion equation is
     * \f[ \rho C_P \partial_t T = \nabla \cdot (D \nabla T) \f]
     * The bilogarthmic transform may also be used with this equation to give
     * \f[ \rho C_P f(T) \partial_t T = \nabla \cdot (f(T) D \nabla T), \f]
     * with similar considerations as for the chemical equation.
     */
    bool d_UseBilogScaling;

    /**
     * \brief the material argument to which the bilogarithmic transformation applies
     *
     * This must be one of the values returned by AMP::Materials::Material::get_arg_names(). The
     * material argument
     * bounds
     * reported by AMP::Materials::Material::get_arg_range() will be used for \f$ a\f$ and \f$b\f$.
     */
    std::string d_BilogVariable;

    /**
     * \brief scale the output coefficient as well as the variable.
     */
    bool d_BilogScaleCoefficient;

protected: // used to be private
    std::array<double, 2> d_BilogRange;
    double d_BilogEpsilonRangeLimit;

    static const std::vector<libMesh::Point> d_DummyCoords;

    bool d_IsTensor;
};
} // namespace AMP::Operator

#endif
