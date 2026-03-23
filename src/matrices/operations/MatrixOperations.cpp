#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/matrices/operations/MatrixOperationsFactory.h"
#include "AMP/vectors/Vector.h"


namespace AMP::LinearAlgebra {

/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
MatrixOperations::MatrixOperations() : d_hash( reinterpret_cast<uint64_t>( this ) ) {}


/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t MatrixOperations::getID() const { return d_hash; }


/****************************************************************
 * Copy cast data                                                *
 ****************************************************************/
void MatrixOperations::copyCast( const MatrixData &, MatrixData & )
{
    AMP_ERROR( "NOT IMPLEMENTED" );
}


/****************************************************************
 * Default implementation of getRowSums                          *
 ****************************************************************/
void MatrixOperations::getRowSums( MatrixData const &A, std::shared_ptr<Vector> sum )
{
    AMP_ASSERT( sum );
    sum->zero();
    std::vector<size_t> cols;
    std::vector<double> values;
    for ( size_t row = A.beginRow(); row != A.endRow(); row++ ) {
        A.getRowByGlobalID( row, cols, values );
        double s = 0.0;
        for ( auto &value : values )
            s += value;
        sum->setValueByGlobalID( row, s );
    }
}
void MatrixOperations::getRowSumsAbsolute( MatrixData const &A,
                                           std::shared_ptr<Vector> sum,
                                           const bool removeZeros )
{
    AMP_ASSERT( !removeZeros );
    AMP_ASSERT( sum );
    sum->zero();
    std::vector<size_t> cols;
    std::vector<double> values;
    for ( size_t row = A.beginRow(); row != A.endRow(); row++ ) {
        A.getRowByGlobalID( row, cols, values );
        double s = 0.0;
        for ( auto &value : values )
            s += fabs( value );
        sum->setValueByGlobalID( row, s );
    }
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MatrixOperations::registerChildObjects( AMP::IO::RestartManager * ) const {}
void MatrixOperations::writeRestart( int64_t ) const {}


} // namespace AMP::LinearAlgebra

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::MatrixOperations> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "ClassType", d_data->type() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::MatrixOperations>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    auto ops  = AMP::LinearAlgebra::MatrixOperationsFactory::create( gid, manager );
    closeGroup( gid );
    return ops;
}
