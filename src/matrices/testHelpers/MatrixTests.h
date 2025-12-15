#ifndef included_AMP_test_MatrixTests
#define included_AMP_test_MatrixTests

#include "AMP/matrices/Matrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

#include <memory>
#include <string>


namespace AMP::LinearAlgebra {

void fillWithPseudoLaplacian( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
                              std::shared_ptr<AMP::Discretization::DOFManager> dofmap );


/**
 * \class MatrixFactory
 * \brief A helper class to generate matricies
 */
class MatrixFactory
{
public:
    virtual ~MatrixFactory() {}
    virtual std::shared_ptr<AMP::Mesh::Mesh> getMesh() const                    = 0;
    virtual std::shared_ptr<AMP::LinearAlgebra::Vector> getVector() const       = 0;
    virtual std::shared_ptr<AMP::LinearAlgebra::Matrix> getMatrix() const       = 0;
    virtual std::shared_ptr<AMP::Discretization::DOFManager> getDOFMap() const  = 0;
    virtual std::shared_ptr<AMP::Discretization::DOFManager> getDOFMapL() const = 0;
    virtual std::string name() const                                            = 0;
    const std::string &type() const { return d_type; }

protected:
    MatrixFactory( const std::string &type ) : d_type( type ) {}
    MatrixFactory( const MatrixFactory & );

protected:
    const std::string d_type;
};


/**
 * \class MatrixTests
 * \brief A helper class to store/run tests for a matrix
 */
class MatrixTests
{
public:
    explicit MatrixTests( std::shared_ptr<const MatrixFactory> factory,
                          std::shared_ptr<const MatrixFactory> copy_factory = nullptr )
        : d_factory( factory ), d_copy_factory( copy_factory )
    {
    }

public:
    void InstantiateMatrix( AMP::UnitTest *ut );
    void VerifyGetLeftRightVector( AMP::UnitTest *ut );
    void VerifyGetSetValuesMatrix( AMP::UnitTest *ut );
    void VerifyAXPYMatrix( AMP::UnitTest *ut );
    void VerifyCopyMatrix( AMP::UnitTest *ut );
    void VerifyScaleMatrix( AMP::UnitTest *ut );
    void VerifyExtractDiagonal( AMP::UnitTest *ut );
    void VerifyMultMatrix( AMP::UnitTest *ut );
    void VerifyMatMultMatrix( AMP::UnitTest *ut );
    void VerifyMatMultMatrix_IA( AMP::UnitTest *ut );
    void VerifyMatMultMatrix_AI( AMP::UnitTest *ut );
    void VerifyMatMultMatrix_AA( AMP::UnitTest *ut );
    void VerifyAddElementNode( AMP::UnitTest *ut );

private:
    std::shared_ptr<AMP::LinearAlgebra::Matrix>
    getCopyMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix );
    std::shared_ptr<const MatrixFactory> d_factory;
    std::shared_ptr<const MatrixFactory> d_copy_factory;
};

void test_matrix_loop( AMP::UnitTest &ut, std::shared_ptr<MatrixTests> tests );


void test_matrix_loop( AMP::UnitTest &ut,
                       std::shared_ptr<MatrixFactory> factory,
                       std::shared_ptr<MatrixFactory> copy_factory = nullptr );


void testBasics( AMP::UnitTest &ut, const std::string &type );


} // namespace AMP::LinearAlgebra

#endif
