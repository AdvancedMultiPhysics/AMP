
#include "TrilinosMatrixShellOperator.h"
#include "vectors/VectorBuilder.h"

namespace AMP {
namespace Operator {


TrilinosMatrixShellOperator :: TrilinosMatrixShellOperator(const boost::shared_ptr<OperatorParameters>& params) :
    LinearOperator (params) 
{ 
}


void TrilinosMatrixShellOperator :: setGetRow(void (*func)(void* object, int row, std::vector<unsigned int> &cols, std::vector<double> &values)) 
{
    d_getRow = func;
}


void TrilinosMatrixShellOperator :: apply(AMP::LinearAlgebra::Vector::const_shared_ptr f,
    AMP::LinearAlgebra::Vector::const_shared_ptr u, AMP::LinearAlgebra::Vector::shared_ptr r,
    const double a, const double b) 
{
    d_operator->apply(f, u, r, a, b);
}


void TrilinosMatrixShellOperator :: reset(const boost::shared_ptr<OperatorParameters>& params) 
{
    d_operator->reset(params);
}


AMP::LinearAlgebra::Variable::shared_ptr TrilinosMatrixShellOperator :: getOutputVariable() 
{
    return d_operator->getOutputVariable();
}


AMP::LinearAlgebra::Variable::shared_ptr TrilinosMatrixShellOperator :: getInputVariable() 
{
    return d_operator->getInputVariable();
}


void TrilinosMatrixShellOperator :: setOperator(boost::shared_ptr<Operator> op) 
{
    d_operator = op;
}


void TrilinosMatrixShellOperator :: setNodalDofMap(boost::shared_ptr<AMP::Discretization::DOFManager> dofMap) 
{
    d_nodalDofMap = dofMap;
}


size_t TrilinosMatrixShellOperator :: getMatrixSize() {
    return (d_nodalDofMap->numGlobalDOF());
}


int TrilinosMatrixShellOperator :: matVec(ML_Operator *data, int in_length, double in[], 
    int out_length, double out[]) 
{
    TrilinosMatrixShellOperator* op = reinterpret_cast<TrilinosMatrixShellOperator *>(ML_Get_MyMatvecData(data));

    AMP::LinearAlgebra::Vector::shared_ptr inVec = AMP::LinearAlgebra::createVector( (op->d_nodalDofMap), (op->getInputVariable()), false );
    AMP::LinearAlgebra::Vector::shared_ptr outVec = AMP::LinearAlgebra::createVector( (op->d_nodalDofMap), (op->getOutputVariable()), false );

    inVec->putRawData(in);

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    (op->d_operator)->apply(nullVec, inVec, outVec, 1.0, 0.0);

    double* outPtr = outVec->getRawDataBlock<double>();
    for(int i = 0; i < out_length; i++) {
    out[i] = outPtr[i];
    }
    return 0;
}


int TrilinosMatrixShellOperator :: getRow(ML_Operator *data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]) 
{
    TrilinosMatrixShellOperator* op = reinterpret_cast<TrilinosMatrixShellOperator *>(ML_Get_MyGetrowData(data));

    int spaceRequired = 0;
    int cnt = 0;
    for(int i = 0; i < N_requested_rows; i++) {
        int row = requested_rows[i];
        std::vector<unsigned int> cols;
        std::vector<double> vals;

        (*(op->d_getRow))(op->d_operator.get(), row, cols, vals);
        spaceRequired += cols.size();

        if(allocated_space >= spaceRequired) {
            for(size_t j = 0; j < cols.size(); j++) {
                columns[cnt] = cols[j];
                values[cnt] = vals[j];
                cnt++;
            }
            row_lengths[i] = cols.size();
        } else {
            return 0;
        }
    }

    return 1;
}


void TrilinosMatrixShellOperator :: getColumn(int column, std::vector<unsigned int> &rows, std::vector<double> &values) 
{
    AMP::LinearAlgebra::Vector::shared_ptr inVec = AMP::LinearAlgebra::createVector( d_nodalDofMap, getInputVariable(), false );
    AMP::LinearAlgebra::Vector::shared_ptr outVec = AMP::LinearAlgebra::createVector( d_nodalDofMap, getOutputVariable(), false );

    inVec->zero();
    inVec->setValueByGlobalID(column, 1.0);
    inVec->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    d_operator->apply(nullVec, inVec, outVec, 1.0, 0.0);

    size_t outLength = outVec->getGlobalSize();
    double* outPtr = outVec->getRawDataBlock<double>();

    rows.clear();
    values.clear();
    for(size_t i = 0; i < outLength; i++) {
        if(outPtr[i]) {
            rows.push_back(i);
            values.push_back(outPtr[i]);
        }
    }
}


}  // Operator namespace
}  // AMP namespace

