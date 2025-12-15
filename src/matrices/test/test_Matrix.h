#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <stdio.h>
#include <string>

using namespace AMP::unit_test;


namespace AMP::LinearAlgebra {


// Class to create a matrix from a simpleDOFManager
template<int NUM_DOF_ROW, int NUM_DOF_COL, class GENERATOR>
class DOFMatrixTestFactory : public MatrixFactory
{
public:
    DOFMatrixTestFactory( const std::string &type ) : MatrixFactory( type )
    {
        PROFILE( "DOFMatrixTestFactory" );
        // Create the mesh
        GENERATOR meshGenerator;
        mesh = meshGenerator.getMesh();
        // Create the DOF_Manager
        if ( NUM_DOF_ROW != NUM_DOF_COL )
            AMP_ERROR( "DOF Manager is not finished to the point we can have "
                       "different number of DOFs for the two matrix variables" );
        DOFs = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, NUM_DOF_ROW );
    }

    ~DOFMatrixTestFactory()
    {
        DOFs.reset();
        mesh.reset();
    }

    std::string name() const override
    {
        GENERATOR meshGenerator;
        auto name = meshGenerator.name();
        return AMP::Utilities::stringf( "DOFMatrixTestFactory<%i,%i,%s,%s>",
                                        NUM_DOF_ROW,
                                        NUM_DOF_COL,
                                        name.c_str(),
                                        d_type.data() );
    }

    std::shared_ptr<AMP::Mesh::Mesh> getMesh() const override { return mesh; }

    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE( "getVector" );
        auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "a" );
        auto vector   = AMP::LinearAlgebra::createVector( DOFs, variable );
        return vector;
    }

    std::shared_ptr<AMP::LinearAlgebra::Matrix> getMatrix() const override
    {
        PROFILE( "getMatrix" );
        auto variable_a = std::make_shared<AMP::LinearAlgebra::Variable>( "a" );
        auto variable_b = std::make_shared<AMP::LinearAlgebra::Variable>( "b" );
        auto vector_a   = AMP::LinearAlgebra::createVector( DOFs, variable_a );
        auto vector_b   = AMP::LinearAlgebra::createVector( DOFs, variable_b );
        auto matrix     = AMP::LinearAlgebra::createMatrix( vector_a, vector_b, d_type );
        return matrix;
    }

    std::shared_ptr<AMP::Discretization::DOFManager> getDOFMap() const override { return DOFs; }

    std::shared_ptr<AMP::Discretization::DOFManager> getDOFMapL() const override { return DOFs; }

private:
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    std::shared_ptr<AMP::Discretization::DOFManager> DOFs;
};


} // namespace AMP::LinearAlgebra
