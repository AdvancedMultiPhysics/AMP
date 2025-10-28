#ifndef included_AMP_Matrix_MatrixFactory
#define included_AMP_Matrix_MatrixFactory

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {

class Matrix;

template<class MATRIX>
std::unique_ptr<MATRIX> createMatrixFromRestart( int64_t fid, AMP::IO::RestartManager *manager )
{
    return std::make_unique<MATRIX>( fid, manager );
}


//! Matrix factory class
class MatrixFactory
{
public:
    //! get a singleton instance of the factory
    static MatrixFactory &getFactory()
    {
        static MatrixFactory singletonInstance;
        return singletonInstance;
    }

    //! Create the matrix from the restart file
    static std::shared_ptr<Matrix> create( int64_t fid, AMP::IO::RestartManager *manager );

    //! Register a matrix with the factory
    template<class MATRIX>
    static void registerMatrix( const std::string &name )
    {
        FactoryStrategy<Matrix, int64_t, AMP::IO::RestartManager *>::registerFactory(
            name, createMatrixFromRestart<MATRIX> );
    }
};


} // namespace AMP::LinearAlgebra

#endif
