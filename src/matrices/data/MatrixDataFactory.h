#ifndef included_AMP_Matrix_MatrixDataFactory
#define included_AMP_Matrix_MatrixDataFactory

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {

class MatrixData;

template<class MATRIXDATA>
std::unique_ptr<MATRIXDATA> createMatrixDataFromRestart( int64_t fid,
                                                         AMP::IO::RestartManager *manager )
{
    return std::make_unique<MATRIXDATA>( fid, manager );
}


//! MatrixData factory class
class MatrixDataFactory
{
public:
    //! get a singleton instance of the factory
    static MatrixDataFactory &getFactory()
    {
        static MatrixDataFactory singletonInstance;
        return singletonInstance;
    }

    //! Create the matrix data from the restart file
    static std::shared_ptr<MatrixData> create( int64_t fid, AMP::IO::RestartManager *manager );

    //! Register matrix data with the factory
    template<class MATRIXDATA>
    static void registerMatrixData( const std::string &name )
    {
        FactoryStrategy<MatrixData, int64_t, AMP::IO::RestartManager *>::registerFactory(
            name, createMatrixDataFromRestart<MATRIXDATA> );
    }
};


} // namespace AMP::LinearAlgebra

#endif
