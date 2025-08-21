#ifndef included_AMP_DiffusionFD
#define included_AMP_DiffusionFD

#include "AMP/IO/PIO.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

namespace AMP::Operator {


/* ------------------------------------------------------------------------------------------
    Implementation of a finite-difference discretization of a constant-coefficient diffusion
operator
------------------------------------------------------------------------------------------- */

/* Specifically, given a set of coefficients {cij}, the discretized operators are:
    - In 1D:
        -cxx*u_xx (3-point stencil)
    - In 2D:
        -(cxx*u_xx + cyy*u_yy + cyx*u_yx) (9-point stencil)
    - In 3D:
        -(cxx*u_xx + cyy*u_yy + czz*u_zz + cyx*u_yx + czx*u_zx + czy*u_yz) (19-point stencil)

    - Mixed derivatives are discretized with central finite differences (as opposed to upwind)
    - The incoming database must specify the constants cij
    - The mesh in the incoming OperatorParamaters must be a BoxMesh build from a cube generator
    - This class provides the functionality to construct a RHS vector, given the source term on the
   RHS of the PDE and Dirichlet boundary function
    - Dirichlet boundary conditions are not eliminated; instead, boundary rows in the matrix are
   some scaled identity (with the constant equal to the diagonal entry in the discretization
   stencil)
    - Note that, technically, any second-order, constant-coefficient PDEs of the form specific above
   can be discretized with this class. I.e., this class does not enforce any requirement that the
   PDE be elliptic (determinant[PDE operator] < 0).
*/

class DiffusionFDOperator : public LinearOperator
{

    //
private:
    size_t d_dim                                  = 0;
    std::shared_ptr<AMP::Mesh::BoxMesh> d_BoxMesh = nullptr;
    // Convenience member
    static constexpr auto VertexGeom = AMP::Mesh::GeomType::Vertex;

    // Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;
    // FD coefficients
    std::shared_ptr<std::vector<double>> d_stencil = nullptr;
    // Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;
    // Auxilliary variable for creating CSR data
    std::shared_ptr<std::vector<AMP::Mesh::MeshElementID>> d_ids = nullptr;

    //
public:
    std::shared_ptr<AMP::Database> d_db;
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFMan;

    // Constructor
    DiffusionFDOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ );

    // Destructor
    virtual ~DiffusionFDOperator() {}

    // Used by OperatorFactory to create a DiffusionFDOperator
    static std::unique_ptr<AMP::Operator::Operator>
    create( std::shared_ptr<AMP::Operator::OperatorParameters> params )
    {
        return std::make_unique<DiffusionFDOperator>( params );
    };

    // Used to register this operator in a factory
    std::string type() const override { return "DiffusionFDOperator"; }

    /* Create RHS vector consistent with the linear operator. Here:
    1. PDESourceFun is a function returning the PDE source term
    2. boundaryFun  is a function returning the Dirichlet boundary value at the given location on
    the given boundary */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createRHSVector(
        std::function<double( AMP::Mesh::MeshElement & )> PDESourceFun,
        std::function<double( AMP::Mesh::MeshElement &, int boundary_id )> boundaryFun );

    // Populate a vector with the given function
    void fillVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> u,
                                 std::function<double( AMP::Mesh::MeshElement & )> fun ) const;

    // Vector of hx, hy, hz
    std::vector<double> getMeshSize() const;

    //
private:
    // Build and set d_DOFMan
    void setDOFManager();

    // FD stencil
    std::vector<double> createStencil() const;

    // Pack cols+data vectors in given row
    void getCSRData( size_t row, std::vector<size_t> &cols, std::vector<double> &data ) const;

    // Fill CSR matrix with data
    void fillMatrixWithCSRData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix ) const;

    // Assemble matrix
    std::shared_ptr<AMP::LinearAlgebra::Matrix> createDiscretizationMatrix();

    // Map from grid index i, or i,j, or i,j,k to a MeshElementIndex to a MeshElementId and then to
    // the corresponding DOF
    size_t gridIndsToDOF( int i, int j = 0, int k = 0 ) const;

    // Map from a grid index to a mesh element index
    AMP::Mesh::MeshElementID gridIndsToMeshElementIndex( int i, int j = 0, int k = 0 ) const;

    // Map from a grid index to a mesh element index
    AMP::Mesh::MeshElementID gridIndsToMeshElementIndex( std::array<int, 3> ijk ) const;

    size_t DOFToGridInds( size_t dof, int component = 0 ) const;

    // Convert a global element box to a global node box.
    // Modified from src/mesh/test/test_BoxMeshIndex.cpp by removing the possibility of any of the
    // grid dimensions being periodic.
    AMP::Mesh::BoxMesh::Box getGlobalNodeBox() const;
};
} // namespace AMP::Operator


#endif