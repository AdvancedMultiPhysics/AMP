#ifndef included_AMP_Matrix_MatrixOperationsFactory
#define included_AMP_Matrix_MatrixOperationsFactory

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {

class MatrixOperations;

template<class VECTOROPERATIONS>
std::unique_ptr<VECTOROPERATIONS>
createMatrixOperationsFromRestart( int64_t fid, AMP::IO::RestartManager *manager )
{
    return std::make_unique<VECTOROPERATIONS>( fid, manager );
}


//! MatrixOperations factory class
class MatrixOperationsFactory
{
public:
    //! get a singleton instance of the factory
    static MatrixOperationsFactory &getFactory()
    {
        static MatrixOperationsFactory singletonInstance;
        return singletonInstance;
    }

    //! Create the vector from the restart file
    static std::shared_ptr<MatrixOperations> create( int64_t fid,
                                                     AMP::IO::RestartManager *manager );

    //! Register a vector with the factory
    template<class VECTOROPERATIONS>
    static void registerMatrixOperations( const std::string &name )
    {
        FactoryStrategy<MatrixOperations, int64_t, AMP::IO::RestartManager *>::registerFactory(
            name, createMatrixOperationsFromRestart<VECTOROPERATIONS> );
    }
};


} // namespace AMP::LinearAlgebra

#endif
