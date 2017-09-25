
#ifndef included_AMP_MechanicsElement
#define included_AMP_MechanicsElement

#include <vector>

#include "utils/shared_ptr.h"

/* AMP files */
#include "MechanicsMaterialModel.h"
#include "operators/ElementOperation.h"

/* Libmesh files */
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"

namespace AMP {
namespace Operator {

/**
  An abstract base class for representing the element level computation performed within a
  finite element operator for modelling solid mechanics (elasticity/plasticity).
  This class just handles some of the data and operations that is required by both the linear and
  nonlinear
  mechanics operators to perform their respective element level computations.
  The actual computation is implemented in the apply() function in the derived classes.
  @see MechanicsLinearFEOperator
  @see MechanicsNonlinearFEOperator
  */
class MechanicsElement : public ElementOperation
{
public:
    /**
      Constructor. This builds the finite element shape functions. This reads the
      values for the following keys from the database object contained in
      the parameter object, params:
      1) USE_REDUCED_INTEGRATION (false by default)
      2) FE_ORDER (FIRST by default) - Order of the polynomial used in the shape functions.
      3) FE_FAMILY (LAGRANGE by default) - Family of the polynomial used in the shape functions.
      4) QRULE_TYPE (QGAUSS by default) - Type of numerical integration scheme used.
      5) QRULE_ORDER (DEFAULT by default) - Order of the numerical integration scheme.
      */
    explicit MechanicsElement( const AMP::shared_ptr<ElementOperationParameters> &params );

    /**
      Destructor.
      */
    virtual ~MechanicsElement() {}

    /**
      This function is used by the Linear and Nonlinear mechanics FEOperators to pass
      the current element and material model to this class during the finite element
      assembly operation.
      @param [in] elem Pointer to the current element within a finite element assembly.
      @param [in] materialModel Shared pointer to the mechanics material model used in the current
      element.
      */
    void initializeForCurrentElement( const ::Elem *elem,
                                      const AMP::shared_ptr<MechanicsMaterialModel> &materialModel )
    {
        d_elem          = elem;
        d_materialModel = materialModel;
    }

    /**
      @return Number of gauss quadrature points per element
      */
    unsigned int getNumberOfGaussPoints() { return ( d_qrule->n_points() ); }

protected:
    bool d_useReducedIntegration; /**< A flag that is true if reduced integration
                                    scheme is used and false otherwise. */

    AMP::shared_ptr<::FEType> d_feType; /**< Type of polynomial used for the
                                            finite element shape functions. This includes
                                            both the polynomial order:
                                            First order/Second order etc. and polynomial family:
                                            Lagrange/Hierarchic/Hermite etc.  */

    AMP::shared_ptr<::FEBase> d_fe; /**< Finite element shape functions. */

    AMP::shared_ptr<::QBase> d_qrule; /**< Quadtrature rule used for numerical integration. */

    const ::Elem *d_elem; /**< Pointer to the current element within the finite element assembly. */

    AMP::shared_ptr<MechanicsMaterialModel>
        d_materialModel; /**< Shared pointer to
                             the mechanics material
                             model used in the current element. */

    bool d_useJaumannRate; /**< A flag that checks whether to use Jaumann Rate in Updated Lagrangian
                            * formulation or not.
                            */

    bool d_useFlanaganTaylorElem; /** < Inside Green-Naghdi stress-rate whether to use Flanagan
                                     Taylor stress-srate or
                                     not. */

    int d_iDebugPrintInfoLevel;

private:
};
} // namespace Operator
} // namespace AMP

#endif
