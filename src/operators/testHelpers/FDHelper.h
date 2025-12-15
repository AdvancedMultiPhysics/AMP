#ifndef included_AMP_FDHelper
#define included_AMP_FDHelper

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

/** Compute discrete L^p norms of vector u, for p = 1,2,inf on a regular box-shaped mesh with mesh
 * spacings h in each dimension
 */
std::array<double, 3> getDiscreteNorms( const std::vector<double> &h,
                                        std::shared_ptr<const AMP::LinearAlgebra::Vector> u );


//! Populate the given multivector a function of the given type. Assumes two multivector has 2
//! components, each of which can be handled with the given scalar DOFManager
void fillMultiVectorWithFunction(
    std::shared_ptr<const AMP::Mesh::Mesh> Mesh,
    AMP::Mesh::GeomType geom,
    std::shared_ptr<const AMP::Discretization::DOFManager> scalarDOFMan,
    std::shared_ptr<AMP::LinearAlgebra::Vector> vec_,
    const std::function<double( size_t component, AMP::Mesh::Point &point )> &fun );


#endif