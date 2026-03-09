#ifndef included_AMP_DiffusionElement
#define included_AMP_DiffusionElement

#include <vector>

#include "AMP/operators/ElementOperation.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/operators/diffusion/DiffusionTransportTensorModel.h"
#include <memory>

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Operator {

class DiffusionElement : public ElementOperation
{
public:
    explicit DiffusionElement( std::shared_ptr<const ElementOperationParameters> params );

    virtual ~DiffusionElement() {}

    void initializeForCurrentElement( const libMesh::Elem *elem,
                                      std::shared_ptr<DiffusionTransportModel> transportModel );

protected:
    std::shared_ptr<libMesh::FEType> d_feType;

    std::shared_ptr<libMesh::FEBase> d_fe;

    std::shared_ptr<libMesh::QBase> d_qrule;

    const std::vector<libMesh::Real> *d_JxW;

    const std::vector<std::vector<libMesh::Real>> *d_phi;

    const std::vector<std::vector<libMesh::RealGradient>> *d_dphi;

    const libMesh::Elem *d_elem;

    std::shared_ptr<DiffusionTransportModel> d_transportModel;
    std::shared_ptr<DiffusionTransportTensorModel> d_transportTensorModel;

private:
};
} // namespace AMP::Operator

#endif
