#include "AMP/operators/trilinos/ml/TrilinosMatrixShellOperator.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Operator {


TrilinosMatrixShellOperator::TrilinosMatrixShellOperator(
    std::shared_ptr<const OperatorParameters> params )
    : LinearOperator( params ), d_getRow( nullptr )
{
}


void TrilinosMatrixShellOperator::setGetRow(
    std::function<void( void *, int, std::vector<size_t> &, std::vector<double> & )> func )
{
    d_getRow = func;
}


void TrilinosMatrixShellOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                         AMP::LinearAlgebra::Vector::shared_ptr f )
{
    d_operator->apply( u, f );
}

void TrilinosMatrixShellOperator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                                            AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                            AMP::LinearAlgebra::Vector::shared_ptr r )
{
    d_operator->residual( f, u, r );
}

void TrilinosMatrixShellOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    d_operator->reset( params );
}


std::shared_ptr<AMP::LinearAlgebra::Variable> TrilinosMatrixShellOperator::getOutputVariable() const
{
    return d_operator->getOutputVariable();
}


std::shared_ptr<AMP::LinearAlgebra::Variable> TrilinosMatrixShellOperator::getInputVariable() const
{
    return d_operator->getInputVariable();
}


void TrilinosMatrixShellOperator::setOperator( std::shared_ptr<Operator> op ) { d_operator = op; }


void TrilinosMatrixShellOperator::setNodalDofMap(
    std::shared_ptr<AMP::Discretization::DOFManager> dofMap )
{
    d_nodalDofMap = dofMap;
}


size_t TrilinosMatrixShellOperator::getMatrixSize() { return ( d_nodalDofMap->numGlobalDOF() ); }


int TrilinosMatrixShellOperator::matVec(
    ML_Operator *data, int in_length, double in[], int out_length, double out[] )
{
    auto *op = reinterpret_cast<TrilinosMatrixShellOperator *>( ML_Get_MyMatvecData( data ) );

    AMP_ASSERT( in_length == out_length );
    AMP_ASSERT( (int) op->d_nodalDofMap->numLocalDOF() == out_length );

    auto inVec =
        AMP::LinearAlgebra::createVector( op->d_nodalDofMap, op->getInputVariable(), true );
    auto outVec =
        AMP::LinearAlgebra::createVector( op->d_nodalDofMap, op->getOutputVariable(), true );

    inVec->putRawData( in );

    op->d_operator->apply( inVec, outVec );

    outVec->getRawData( out );

    return 0;
}


int TrilinosMatrixShellOperator::getRow( ML_Operator *data,
                                         int N_requested_rows,
                                         int requested_rows[],
                                         int allocated_space,
                                         int columns[],
                                         double values[],
                                         int row_lengths[] )
{
    auto *op = reinterpret_cast<TrilinosMatrixShellOperator *>( ML_Get_MyGetrowData( data ) );
    auto fun = op->d_getRow;

    int spaceRequired = 0;
    int cnt           = 0;
    std::vector<size_t> cols;
    std::vector<double> vals;
    for ( int i = 0; i < N_requested_rows; i++ ) {
        int row = requested_rows[i];
        fun( op->d_operator.get(), row, cols, vals );
        spaceRequired += cols.size();
        if ( allocated_space >= spaceRequired ) {
            for ( size_t j = 0; j < cols.size(); j++ ) {
                columns[cnt] = cols[j];
                values[cnt]  = vals[j];
                cnt++;
            }
            row_lengths[i] = cols.size();
        } else {
            return 0;
        }
    }

    return 1;
}


void TrilinosMatrixShellOperator::getColumn( int column,
                                             std::vector<size_t> &rows,
                                             std::vector<double> &values )
{
    auto inVec  = AMP::LinearAlgebra::createVector( d_nodalDofMap, getInputVariable(), false );
    auto outVec = AMP::LinearAlgebra::createVector( d_nodalDofMap, getOutputVariable(), false );

    inVec->zero();
    auto idx         = size_t( column );
    const double val = 1.0;
    inVec->setValuesByGlobalID( 1, &idx, &val );
    inVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_operator->apply( inVec, outVec );

    size_t outLength = outVec->getLocalSize();
    auto outPtr      = new double[outLength];
    outVec->getRawData( outPtr );

    rows.clear();
    values.clear();
    for ( size_t i = 0; i < outLength; i++ ) {
        if ( outPtr[i] ) {
            rows.push_back( i );
            values.push_back( outPtr[i] );
        }
    }
    delete[] outPtr;
}


} // namespace AMP::Operator
