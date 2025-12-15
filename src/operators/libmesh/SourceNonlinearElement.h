#ifndef included_AMP_SourceNonlinearElement
#define included_AMP_SourceNonlinearElement

#include <vector>

#include <memory>

// AMP files
#include "AMP/operators/ElementOperation.h"
#include "AMP/operators/libmesh/SourcePhysicsModel.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature_gauss.h"
ENABLE_WARNINGS


namespace AMP::Operator {

/**
  A class for representing the element level computation performed within a
  nonlinear volume integral operator.
*/
class SourceNonlinearElement : public ElementOperation
{
public:
    /**
      Constructor. This builds the finite element shape functions. Since this derived
      directly from the class ElementOperation the constructor reads the
      values for the following keys from the database object contained in
      the parameter object, params:
      1) FE_ORDER (FIRST by default) - Order of the polynomial used in the shape functions.
      2) FE_FAMILY (LAGRANGE by default) - Family of the polynomial used in the shape functions.
      3) QRULE_TYPE (QGAUSS by default) - Type of numerical integration scheme used.
      4) QRULE_ORDER (DEFAULT by default) - Order of the numerical integration scheme.
      5) INTEGRATEVOLUME (TRUE by default)- Bool to choose to Integrate (Contradiction to the class
      ???).
     */
    explicit SourceNonlinearElement( std::shared_ptr<const ElementOperationParameters> params );

    //! Destructor.
    virtual ~SourceNonlinearElement() {}

    /**
      This function is used by the VolumeIntegralOperators to pass
      the current element and source physics model to this class during the finite element
      assembly operation.
      @param [in] elem Pointer to the current element within a finite element assembly.
      @param [in] sourceTransportModel Shared pointer to the Source Physics Model used in the
      current element.
     */
    void initializeForCurrentElement( const libMesh::Elem *elem,
                                      std::shared_ptr<SourcePhysicsModel> sourceTransportModel );

    void setElementInputVector( const std::vector<std::vector<double>> &elementInputVector )
    {
        d_elementInputVector = elementInputVector;
    }

    /**
      This function is used by VolumeIntegralOperator to pass the address
      of the element Input, Auxillary and Output vector to this class.
      @param [in] elementInputVector Element input vector
      @param [in] elementAuxVector Element Auxillary vector
      @param [out] elementOutputVector Element residual vector
     */
    void setElementVectors( const std::vector<std::vector<double>> &elementInputVector,
                            const std::vector<std::vector<double>> &elementAuxVector,
                            std::vector<double> &elementOutputVector )
    {
        d_elementInputVector  = elementInputVector;
        d_elementAuxVector    = elementAuxVector;
        d_elementOutputVector = &( elementOutputVector );
    }

    /**
      This function is used to by the VolumeIntegralOperatorset input
      variable type.
     */
    void setElementFlags( const std::string &inputVariableType )
    {
        d_isInputType = inputVariableType;
    }

    /**
      Element residual vector computation.
      */
    void apply() override;

    std::shared_ptr<libMesh::FEBase> getFEBase() { return d_fe; }

    unsigned int getNumberOfGaussPoints() { return ( d_qrule->n_points() ); }


protected:
    std::vector<std::vector<double>> d_elementInputVector;

    std::vector<std::vector<double>> d_elementAuxVector;

    std::vector<double> *d_elementOutputVector;

    std::vector<std::vector<double>> d_elementOtherVectors;

    std::shared_ptr<libMesh::FEType> d_feType;

    std::shared_ptr<libMesh::FEBase> d_fe;

    std::shared_ptr<libMesh::QBase> d_qrule;

    const std::vector<libMesh::Real> *d_JxW;

    const std::vector<std::vector<libMesh::Real>> *d_phi;

    const std::vector<std::vector<libMesh::RealGradient>> *d_dphi;

    std::string d_isInputType;

    const libMesh::Elem *d_elem;

    bool d_integrateVolume;

    std::shared_ptr<SourcePhysicsModel> d_sourcePhysicsModel;

private:
};
} // namespace AMP::Operator

#endif
