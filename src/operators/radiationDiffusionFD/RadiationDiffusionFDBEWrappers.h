#ifndef RAD_DIF_FD_BE_WRAPPERS
#define RAD_DIF_FD_BE_WRAPPERS

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/LinearOperator.h"

#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"

namespace AMP::Operator {

// Classes declared here
class BERadDifOpPJac;
class BERadDifOp;
struct BERadDifOpPJacData;


/** The classes in this file either are (or are associated with) wrapping spatial radiation 
 * diffusion operators, i.e., RadiationDiffusionFD and its Jacobian, as backward Euler operators, i.
 * e., multiplying them by some time-step size and adding an identity. These operators here are
 * closely related to AMP::TimeIntegrator::TimeOperator  
 */


/** ---------------------------------------------------------------- *
 *  ---- Class wrapping a RadDifOp as a backward Euler operator ---- *
 * ----------------------------------------------------------------- */
/** Implements the Operator u + gamma*L(u) where L is a RadDifOp. 
 * This operator arises from the BDF discretization of the ODEs 
 *      u'(t) + L(u) = s(t). 
 * The incoming OperatorParameters are used to create operator L. 
 */ 
class BERadDifOp : public AMP::Operator::Operator {

public:
    //! Time-step size (up to scaling by BDF constants)
    double                    d_gamma    = -1.0;
    //! The underlying radiation diffusion spatial operator
    std::shared_ptr<RadDifOp> d_RadDifOp = nullptr;

    //! Constructor
    BERadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ );

    //! Destructor
    virtual ~BERadDifOp() {};

    //! Compute r <- u + gamma*L(u)
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    //! Used to register this operator in a factory
    std::string type() const override { return "BERadDifOp"; }

    //! Returns RadDifOp's isValidVector
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override {
        return d_RadDifOp->isValidVector( ET );
    }

    //! Set the scaled time-step size of the operator
    void setGamma( AMP::Scalar gamma_ ); 


    //! Set multiphysics scalings
    void setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> f );

    
protected:

    //! Vectors for multiphysics scaling
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSolutionScaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pFunctionScaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pScratchSolVector;

    /** \brief Returns a parameter object that may be used to reset the associated RadDifPJacOp
     * operator. 
     * \param[in] u_in The current nonlinear iterate. 
     * \details Note that the base class's getParameters() get's redirected to this function. 
     */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;

};


/** ------------------------------------------------------------------ *
 *  --- Class wrapping a RadDifOpPJac as a backward Euler operator --- *
 * ------------------------------------------------------------------- */
/** Implements the Operator I + gamma*hat{L} where hat{L} is a 
 * RadDifOpPJac, i.e., a Picard linearization of a radiation diffusion 
 * operator, RadDifOp  
 */
class BERadDifOpPJac : public AMP::Operator::LinearOperator {

public:

    //! Time-step size (up to scaling by BDF constants)
    double                              d_gamma    = -1.0;
    //! The underlying linearized radiation diffusion spatial operator
    std::shared_ptr<RadDifOpPJac>       d_RadDifOpPJac;
    //! Data structure for storing block 2x2 matrix
    std::shared_ptr<BERadDifOpPJacData> d_data = nullptr;


    //! Constructor
    BERadDifOpPJac( std::shared_ptr<AMP::Operator::OperatorParameters> params );

    //! Destructor
    virtual ~BERadDifOpPJac() {};

    //! Used by OperatorFactory to create a BERadDifOpPJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<BERadDifOpPJac>( params ); };

    //! Compute r <- (I + gamma*hat{L})*u
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;
    
    //! Reset the operator based on the incoming parameters 
    void reset(std::shared_ptr<const AMP::Operator::OperatorParameters> params) override;

    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override; 

    
private:
    //! Create new d_data based on my RadDifOpPJac's d_data
    void setData(); 
    
};


/** Data structure for storing the 2x2 block matrix associated with the BERadDifOpPJac, I + 
 * gamma*hat{L}, where hat{L} is a RadDifOpPJac.
 * 
 * Specifically, the constructor here takes in the data structure used to store the 2x2 block matrix associated with hat{L}, i.e., 
 *  [ d_E 0   ] + [ diag(r_EE) diag(r_ET) ]
 *  [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * and overwrites it to create
 *  ([I 0]         [ d_E 0   ])         [ diag(r_EE) diag(r_ET) ]
 *  ([0 I] + gamma*[ 0   d_T ]) + gamma*[ diag(r_TE) diag(r_TT) ]
 *      ==
 *  [ d_E_BE  0    ]   [ diag(r_EE_BE) diag(r_ET_BE) ]
 *  [ 0      d_T_BE] + [ diag(r_TE_BE) diag(r_TT_BE) ]
 * 
 * Note: We store the data in the above format for two reasons:
 * 1. It allows an operator-split preconditioner to be built, wherein the diffusion blocks must
 * contain the identity perturbation since AMG is applied to them, and it's easy enough to add an
 * identity perturbation on the fly to the reaction blocks when decoupled 2x2 solves are done on 
 * them
 * 2. The modification that we make to the data means that our underlying RadDifOpPJac's apply will 
 * actually be an apply of a BERadDifOpPJac (we would have to write another apply routine if we 
 * also added an identity perturbation into the reaction block).
 */
struct BERadDifOpPJacData {
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT_BE = nullptr;

    BERadDifOpPJacData( ) { };
    BERadDifOpPJacData( std::shared_ptr<RadDifOpPJacData> data, double gamma );
};

} // namespace AMP::Operator




#endif