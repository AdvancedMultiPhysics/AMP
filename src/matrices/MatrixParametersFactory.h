#ifndef included_AMP_MatrixParameters_MatrixParametersFactory
#define included_AMP_MatrixParameters_MatrixParametersFactory

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {

class MatrixParametersBase;

template<class MATRIXPARAMETERS>
std::unique_ptr<MATRIXPARAMETERS>
createMatrixParametersFromRestart( int64_t fid, AMP::IO::RestartManager *manager )
{
    return std::make_unique<MATRIXPARAMETERS>( fid, manager );
}


//! MatrixParameters factory class
class MatrixParametersFactory
{
public:
    //! get a singleton instance of the factory
    static MatrixParametersFactory &getFactory()
    {
        static MatrixParametersFactory singletonInstance;
        return singletonInstance;
    }

    //! Create the matrix parameters from the restart file
    static std::shared_ptr<MatrixParametersBase> create( int64_t fid,
                                                         AMP::IO::RestartManager *manager );

    //! Register a matrix parameters with the factory
    template<class MATRIXPARAMETERS>
    static void registerMatrixParameters( const std::string &name )
    {
        FactoryStrategy<MatrixParametersBase, int64_t, AMP::IO::RestartManager *>::registerFactory(
            name, createMatrixParametersFromRestart<MATRIXPARAMETERS> );
    }
};


} // namespace AMP::LinearAlgebra

#endif
