#ifndef included_AMP_MassElement
#define included_AMP_MassElement

#include <memory>
#include <vector>

// AMP files
#include "AMP/operators/ElementOperation.h"
#include "AMP/operators/libmesh/MassDensityModel.h"

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

class MassElement : public ElementOperation
{
public:
    explicit MassElement( std::shared_ptr<const ElementOperationParameters> params );

    virtual ~MassElement() {}

    void initializeForCurrentElement( const libMesh::Elem *elem,
                                      std::shared_ptr<MassDensityModel> densityModel );

protected:
    std::shared_ptr<libMesh::FEType> d_feType;

    std::shared_ptr<libMesh::FEBase> d_fe;

    std::shared_ptr<libMesh::QBase> d_qrule;

    const std::vector<libMesh::Real> *d_JxW;

    const std::vector<std::vector<libMesh::Real>> *d_phi;

    const libMesh::Elem *d_elem;

    std::shared_ptr<MassDensityModel> d_densityModel;

private:
};
} // namespace AMP::Operator

#endif
