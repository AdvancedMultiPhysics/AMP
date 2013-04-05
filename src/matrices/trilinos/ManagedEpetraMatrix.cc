#include "ManagedEpetraMatrix.h"
#include "vectors/trilinos/EpetraVector.h"
#include "vectors/trilinos/EpetraVectorEngine.h"
#include "vectors/trilinos/ManagedEpetraVector.h"
#include <algorithm>
#include "utils/Utilities.h"
#include "utils/ProfilerApp.h"

#include "utils/AMP_MPI.h"

#include "EpetraExt_MatrixMatrix.h"
#ifdef USE_EXT_MPI
    #include <Epetra_MpiComm.h>
#else
    #include <Epetra_SerialComm.h>
#endif

namespace AMP {
namespace LinearAlgebra {

void ManagedEpetraMatrix::multiply ( shared_ptr other_op , shared_ptr &result )
{
    if ( this->numGlobalColumns() != other_op->numGlobalRows() )
        AMP_ERROR( "Inner matrix dimensions must agree" );
    if ( !other_op->isA<ManagedEpetraMatrix>() )
        AMP_ERROR( "Incompatible matrix types" );

    #ifdef USE_EXT_MPI
        MPI_Comm epetraComm = (dynamic_cast<const Epetra_MpiComm *> (&d_epetraMatrix->RowMap().Comm()))->Comm();
    #else
        MPI_Comm epetraComm = AMP_COMM_SELF;
    #endif
    Vector::shared_ptr leftVec = this->getRightVector();
    Vector::shared_ptr rightVec = other_op->getRightVector();
    boost::shared_ptr<ManagedEpetraMatrixParameters> memp( new ManagedEpetraMatrixParameters( 
        leftVec->getDOFManager(), rightVec->getDOFManager(), AMP_MPI(epetraComm) ) );
    memp->d_CommListLeft = leftVec->getCommunicationList();
    memp->d_CommListRight = rightVec->getCommunicationList();
    ManagedEpetraMatrix *res = new ManagedEpetraMatrix( memp );
    PROFILE_START("Epetra::MatrixMultiply");
    int ierr = EpetraExt::MatrixMatrix::Multiply( *d_epetraMatrix, false, 
        *(other_op->castTo<ManagedEpetraMatrix> ().d_epetraMatrix), false, *(res->d_epetraMatrix), true );
    AMP_ASSERT(ierr==0);
    PROFILE_STOP("Epetra::MatrixMultiply");
    result = Matrix::shared_ptr ( res );
}


Vector::shared_ptr ManagedEpetraMatrix::getRightVector ()
{
    boost::shared_ptr<ManagedEpetraMatrixParameters> memp = boost::dynamic_pointer_cast<ManagedEpetraMatrixParameters>( d_pParameters );
    int  localSize = memp->getLocalNumberOfColumns();
    int  globalSize = memp->getGlobalNumberOfColumns();
    EpetraVectorEngineParameters *evep = new EpetraVectorEngineParameters( localSize, globalSize, memp->getEpetraComm() );
    VectorEngineParameters::shared_ptr p_eng ( evep );
    boost::shared_ptr<ManagedVectorParameters> p_params( new ManagedVectorParameters );
    p_params->d_Buffer = VectorEngine::BufferPtr ( new std::vector<double> ( localSize ) );
    p_params->d_Engine = VectorEngine::shared_ptr ( new EpetraVectorEngine ( p_eng , p_params->d_Buffer ) );
    p_params->d_CommList = memp->d_CommListRight;
    p_params->d_DOFManager = memp->getRightDOFManager();
    Vector::shared_ptr rtn = Vector::shared_ptr( new ManagedEpetraVector( p_params ) );
    rtn->setVariable( memp->d_VariableRight );
    //rtn->setVariable( Variable::shared_ptr( new Variable("right") ) );
    return rtn;
}


Vector::shared_ptr ManagedEpetraMatrix::getLeftVector ()
{
    boost::shared_ptr<ManagedEpetraMatrixParameters> memp = boost::dynamic_pointer_cast<ManagedEpetraMatrixParameters>( d_pParameters );
    int  localSize = memp->getLocalNumberOfRows();
    int  globalSize = memp->getGlobalNumberOfRows();
    EpetraVectorEngineParameters *evep = new EpetraVectorEngineParameters( localSize, globalSize, memp->getEpetraComm() );
    VectorEngineParameters::shared_ptr p_eng ( evep );
    boost::shared_ptr<ManagedVectorParameters> p_params( new ManagedVectorParameters );
    p_params->d_Buffer = VectorEngine::BufferPtr ( new std::vector<double> ( localSize ) );
    p_params->d_Engine = VectorEngine::shared_ptr ( new EpetraVectorEngine ( p_eng , p_params->d_Buffer ) );
    p_params->d_CommList = memp->d_CommListLeft;
    p_params->d_DOFManager = memp->getLeftDOFManager();
    Vector::shared_ptr rtn = Vector::shared_ptr( new ManagedEpetraVector( p_params ) );
    rtn->setVariable( memp->d_VariableLeft );
    //rtn->setVariable( Variable::shared_ptr( new Variable("left") ) );
    return rtn;
}


Discretization::DOFManager::shared_ptr ManagedEpetraMatrix::getRightDOFManager ()
{
    return d_pParameters->getRightDOFManager();
}


Discretization::DOFManager::shared_ptr ManagedEpetraMatrix::getLeftDOFManager ()
{
    return d_pParameters->getLeftDOFManager();
}


void ManagedEpetraMatrix::setOtherData ()
{
    AMP_MPI myComm = d_pParameters->getComm();
    int ndxLen = d_OtherData.size();
    int totNdxLen = myComm.sumReduce(ndxLen);
    if ( totNdxLen == 0 ) {
        return;
    }
    int  dataLen = 0;
    std::map<int,std::map<int , double> >::iterator cur_row = d_OtherData.begin();
    while ( cur_row != d_OtherData.end() ) {
        dataLen += cur_row->second.size();
        ++cur_row;
    }
    int *rows = new int [ dataLen+1 ];   //Add one to have the new work
    int *cols = new int [ dataLen+1 ];
    double *data = new double [ dataLen+1 ];
    int cur_ndx = 0;
    int cur_ptr = 0;
    cur_row = d_OtherData.begin();
    while ( cur_row != d_OtherData.end() )
    {
      cur_ndx++;
      std::map<int,double>::iterator cur_elem = cur_row->second.begin();
      while ( cur_elem != cur_row->second.end() )
      {
        rows[cur_ptr] = cur_row->first;
        cols[cur_ptr] = cur_elem->first;
        data[cur_ptr] = cur_elem->second;
        ++cur_ptr;
        ++cur_elem;
      }
      ++cur_row;
    }

    int totDataLen = myComm.sumReduce(dataLen);

    int *aggregateRows = new int [ totDataLen ];
    int *aggregateCols = new int [ totDataLen ];
    double *aggregateData = new double [ totDataLen ];

    myComm.allGather( rows, dataLen, aggregateRows );
    myComm.allGather( cols, dataLen, aggregateCols );
    myComm.allGather( data, dataLen, aggregateData );

    int MyFirstRow = d_pParameters->getLeftDOFManager()->beginDOF();
    int MyEndRow = d_pParameters->getLeftDOFManager()->endDOF();
    for ( int i = 0 ; i != totDataLen ; i++ )
    {
      if ( ( aggregateRows[i] >= MyFirstRow ) && ( aggregateRows[i] < MyEndRow ) )
      {
        setValueByGlobalID ( aggregateRows[i] , aggregateCols[i] , aggregateData[i] );
      }
    }

    d_OtherData.clear();
    delete [] rows;
    delete [] cols;
    delete [] data;
    delete [] aggregateRows;
    delete [] aggregateCols;
    delete [] aggregateData;
}



ManagedEpetraMatrix::ManagedEpetraMatrix ( boost::shared_ptr<ManagedEpetraMatrixParameters> params )
        : EpetraMatrix ( params->getEpetraRowMap(), 
                         params->getEpetraColMap(),
                         params->entryList() ),
          ManagedMatrix( params )
{
    d_pParameters = params;
}


ManagedEpetraMatrix::ManagedEpetraMatrix ( const ManagedEpetraMatrix &rhs )
        : Matrix () ,
          EpetraMatrix ( rhs.d_pParameters->getEpetraRowMap() , 
                           rhs.d_pParameters->getEpetraColMap()  ,
                           rhs.d_pParameters->entryList() ) ,
          ManagedMatrix ( rhs.d_pParameters ) ,
          d_pParameters ( rhs.d_pParameters )
{
    for ( size_t i = d_pParameters->getLeftDOFManager()->beginDOF(); i!=d_pParameters->getLeftDOFManager()->endDOF(); i++ ) {
      std::vector<unsigned int>  cols;
      std::vector<double>        vals;
      rhs.getRowByGlobalID ( (int)i , cols , vals );
      for ( size_t j = 0 ; j != cols.size() ; j++ )
      {
        vals[j] = 0;
      }
      if ( !cols.empty() )
        createValuesByGlobalID ( 1 , (int)cols.size() , (int *)&i , (int *)&(cols[0]) , &(vals[0]) );
    }
    d_RangeMap = rhs.d_RangeMap;
    d_DomainMap = rhs.d_DomainMap;
    makeConsistent();
}


/*
Matrix::shared_ptr  ManagedEpetraMatrix::cloneMatrix () const
{
    return Vector::shared_ptr ( new ManagedEpetraMatrix ( d_pParameters ) );
}
*/



void ManagedEpetraMatrix::fillComplete ()
{
    EpetraMatrix::fillComplete();
}


void ManagedEpetraMatrix::mult ( Vector::const_shared_ptr in , Vector::shared_ptr out )
{
    AMP_ASSERT ( in->getGlobalSize() == numGlobalRows() );
    AMP_ASSERT ( out->getGlobalSize() == numGlobalColumns() );
    boost::shared_ptr<const EpetraVector>  in_view = boost::dynamic_pointer_cast<const EpetraVector>( EpetraVector::constView( in ) );
    boost::shared_ptr<EpetraVector>  out_view = boost::dynamic_pointer_cast<EpetraVector>( EpetraVector::view( out ) );
    const Epetra_Vector &in_vec = in_view->getEpetra_Vector();
    Epetra_Vector &out_vec = out_view->getEpetra_Vector();
    int err = d_epetraMatrix->Multiply( false, in_vec, out_vec );
    VerifyEpetraReturn( err, "mult" );
}


void ManagedEpetraMatrix::multTranspose ( Vector::const_shared_ptr in , Vector::shared_ptr out )
{
    AMP_ASSERT ( in->getGlobalSize() == numGlobalColumns() );
    AMP_ASSERT ( out->getGlobalSize() == numGlobalRows() );
    boost::shared_ptr<const EpetraVector>  in_view = boost::dynamic_pointer_cast<const EpetraVector>( EpetraVector::constView( in ) );
    boost::shared_ptr<EpetraVector>  out_view = boost::dynamic_pointer_cast<EpetraVector>( EpetraVector::view( out ) );
    int err = d_epetraMatrix->Multiply( true, in_view->getEpetra_Vector(), out_view->getEpetra_Vector() );
    VerifyEpetraReturn( err, "mult" );
}


Vector::shared_ptr  ManagedEpetraMatrix::extractDiagonal ( Vector::shared_ptr v )
{
    Vector::shared_ptr  retVal;
    if ( v )
    {
      retVal = EpetraVector::view ( v );
    }
    else
    {
      retVal = getRightVector();
    }
    VerifyEpetraReturn ( d_epetraMatrix->ExtractDiagonalCopy ( retVal->castTo<EpetraVector>().getEpetra_Vector() ) , 
                         "extractDiagonal" );
    return retVal;
}
  

void  ManagedEpetraMatrix::scale ( double alpha ) 
{
    VerifyEpetraReturn ( d_epetraMatrix->Scale ( alpha ) , "scale" );
}


void  ManagedEpetraMatrix::axpy ( double alpha , const Matrix &rhs )
{
    int    *a , *b;
    double *values1;
    double *values2;
    d_epetraMatrix->ExtractCrsDataPointers ( a , b , values1 );
    rhs.castTo<ManagedEpetraMatrix>().d_epetraMatrix->ExtractCrsDataPointers ( a , b , values2 );
    for ( int i = 0 ; i != d_epetraMatrix->NumMyNonzeros() ; i++ )
      values1[i] += alpha*values2[i];
}


void  ManagedEpetraMatrix::addValuesByGlobalID ( int  num_rows , int num_cols , int *rows , int *cols , double *values )
{
    for ( int i = 0 ; i != num_rows ; i++ )
      VerifyEpetraReturn ( d_epetraMatrix->SumIntoGlobalValues ( rows[i] , num_cols , values+num_cols*i , cols ) , "addValuesByGlobalId" );
}


void  ManagedEpetraMatrix::createValuesByGlobalID ( int  num_rows , int num_cols , int *rows , int *cols , double *values )
{
    for ( int i = 0 ; i != num_rows ; i++ )
      VerifyEpetraReturn (d_epetraMatrix->InsertGlobalValues ( rows[i] , num_cols , values+num_cols*i , cols ) , "setValuesByGlobalID" );
}


void  ManagedEpetraMatrix::setValuesByGlobalID ( int  num_rows , int num_cols , int *rows , int *cols , double *values )
{

    int MyFirstRow = d_pParameters->getLeftDOFManager()->beginDOF();
    int MyEndRow = d_pParameters->getLeftDOFManager()->endDOF();
    for ( int i = 0 ; i != num_rows ; i++ ) {
        VerifyEpetraReturn (d_epetraMatrix->ReplaceGlobalValues ( rows[i] , num_cols , values+num_cols*i , cols ) , "setValuesByGlobalID" );
        if ( rows[i]<MyFirstRow || rows[i]>=MyEndRow ) {
            for ( int j = 0 ; j != num_cols ; j++ ) {
                d_OtherData[rows[i]][cols[j]] = values[num_cols*i + j];
            }
        }
    }
}


void  ManagedEpetraMatrix::setScalar ( double ans )
{
    VerifyEpetraReturn ( d_epetraMatrix->PutScalar ( ans ) , "setScalar" );
}


void  ManagedEpetraMatrix::makeConsistent ()
{
    Epetra_FECrsMatrix  *mat = dynamic_cast<Epetra_FECrsMatrix *> ( d_epetraMatrix );
    if ( mat )
    {
      VerifyEpetraReturn ( mat->GlobalAssemble ( false ) , "makeParallelConsistent" );
      fillComplete ();
    }
    setOtherData ();
}


void ManagedEpetraMatrix::setDiagonal ( const Vector::shared_ptr &in )
{
    VerifyEpetraReturn ( d_epetraMatrix->ReplaceDiagonalValues ( EpetraVector::view ( in )->castTo<EpetraVector>().getEpetra_Vector() ) , "setDiagonal" );
}


void ManagedEpetraMatrix::getRowByGlobalID ( int row , std::vector<unsigned int> &cols , std::vector<double> &values ) const
{   
    int firstRow = d_pParameters->getLeftDOFManager()->beginDOF();
    int numRows = d_pParameters->getLeftDOFManager()->endDOF();
    AMP_ASSERT ( row >= firstRow );
    AMP_ASSERT ( row < firstRow + numRows );

    int localRow = row - firstRow;
    int numCols = d_pParameters->entriesInRow ( localRow );
    cols.resize ( numCols );
    values.resize ( numCols );

    if ( numCols )
      VerifyEpetraReturn ( d_epetraMatrix->ExtractGlobalRowCopy ( row , numCols , numCols , &(values[0]) , (int *)&(cols[0]) ) , "getRowByGlobalID" );
}


}
}

