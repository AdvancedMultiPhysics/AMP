#include "test_Matrix.h"

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/vectors/VectorBuilder.h"

#include <chrono>


#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()


using namespace AMP::LinearAlgebra;


class OddEvenMatrixFactory : public MatrixFactory
{
public:
    OddEvenMatrixFactory( const std::string &type, size_t N_local, bool oddEven )
        : MatrixFactory( type )
    {
        AMP::AMP_MPI comm( AMP_COMM_WORLD );
        size_t N     = N_local * comm.getSize();
        size_t start = N_local * comm.getRank();
        std::vector<size_t> remotes;
        remotes.reserve( ( N + 1 ) / 2 );
        i0 = oddEven ? 1 : 0;
        for ( size_t i = i0; i < N; i += 2 ) {
            if ( i < start || i >= start + N_local )
                remotes.push_back( i );
        }
        DOFs =
            std::make_shared<AMP::Discretization::DOFManager>( N_local, AMP_COMM_WORLD, remotes );
    }

    std::string name() const override { return "OddEvenMatrixFactory"; }

    std::shared_ptr<AMP::Mesh::Mesh> getMesh() const override { return nullptr; }

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
        size_t N        = DOFs->numGlobalDOF();
        auto getRow     = [&i0 = this->i0, N]( size_t row ) {
            std::vector<size_t> x;
            if ( row % 2 != i0 )
                return x;
            x.reserve( ( N + 1 ) / 2 );
            for ( size_t i = i0; i < N; i += 2 )
                x.push_back( i );
            return x;
        };
        auto matrix = AMP::LinearAlgebra::createMatrix( vector_a, vector_b, d_type, getRow );
        return matrix;
    }

    std::shared_ptr<AMP::Discretization::DOFManager> getDOFMap() const override { return DOFs; }

    std::shared_ptr<AMP::Discretization::DOFManager> getDOFMapL() const override { return DOFs; }

private:
    size_t i0 = 0;
    std::shared_ptr<AMP::Discretization::DOFManager> DOFs;
};


void runTests( AMP::UnitTest &ut, const std::string &type )
{
    OddEvenMatrixFactory factory1( type, 20, true );
    OddEvenMatrixFactory factory2( type, 20, false );
    auto A = factory1.getMatrix();
    auto B = factory2.getMatrix();
    // auto AA = AMP::LinearAlgebra::Matrix::matMatMult( A, A );
    auto AB = AMP::LinearAlgebra::Matrix::matMatMult( A, B );
    auto BA = AMP::LinearAlgebra::Matrix::matMatMult( B, A );
    // auto BB = AMP::LinearAlgebra::Matrix::matMatMult( B, B );
    std::vector<size_t> cols;
    std::vector<double> values;
    bool pass = true;
    for ( size_t row = A->beginRow(); row < A->endRow(); row++ ) {
        AB->getRowByGlobalID( row, cols, values );
        pass = pass && cols.empty();
        BA->getRowByGlobalID( row, cols, values );
        pass = pass && cols.empty();
    }
    if ( pass )
        ut.passes( type );
    else
        ut.passes( type );
}


int main( int argc, char **argv )
{

    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

#ifdef AMP_USE_TRILINOS
    runTests( ut, "ManagedEpetraMatrix" );
#endif

    runTests( ut, "CSRMatrix" );

    ut.report();
    int num_failed = ut.NumFailGlobal();
    ut.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
