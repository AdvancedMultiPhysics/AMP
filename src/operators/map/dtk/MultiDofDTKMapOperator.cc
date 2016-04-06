#include <operators/map/dtk/MultiDofDTKMapOperator.h>

namespace AMP {
namespace Operator {

//---------------------------------------------------------------------------//
// Constructor
MultiDofDTKMapOperator::MultiDofDTKMapOperator( const AMP::shared_ptr<OperatorParameters> &params )
{
    // Get the operator parameters.
    AMP::shared_ptr<MultiDofDTKMapOperatorParameters> multiDofDTKMapOpParams =
        AMP::dynamic_pointer_cast<MultiDofDTKMapOperatorParameters>( params );
    AMP_ASSERT( multiDofDTKMapOpParams );
    d_multiDofDTKMapOpParams =
        AMP::dynamic_pointer_cast<MultiDofDTKMapOperatorParameters>( params );

    ; 
    AMP::Mesh::Mesh::shared_ptr mesh1 = multiDofDTKMapOpParams->d_Mesh1;
    AMP::Mesh::Mesh::shared_ptr mesh2 = multiDofDTKMapOpParams->d_Mesh2;
    int boundaryID1                   = multiDofDTKMapOpParams->d_BoundaryID1;
    int boundaryID2                   = multiDofDTKMapOpParams->d_BoundaryID2;
    std::string variable1             = multiDofDTKMapOpParams->d_Variable1;
    std::string variable2             = multiDofDTKMapOpParams->d_Variable2;
    std::size_t strideOffset1         = multiDofDTKMapOpParams->d_StrideOffset1;
    std::size_t strideOffset2         = multiDofDTKMapOpParams->d_StrideOffset2;
    std::size_t strideLength1         = multiDofDTKMapOpParams->d_StrideLength1;
    std::size_t strideLength2         = multiDofDTKMapOpParams->d_StrideLength2;
    AMP::LinearAlgebra::Vector::const_shared_ptr sourceVector =
        multiDofDTKMapOpParams->d_SourceVector;
    AMP::LinearAlgebra::Vector::shared_ptr targetVector = multiDofDTKMapOpParams->d_TargetVector;

    AMP::Mesh::Mesh::shared_ptr boundaryMesh1;
    AMP::Mesh::Mesh::shared_ptr boundaryMesh2;
    AMP::shared_ptr<AMP::Discretization::DOFManager> sourceDofManager12, sourceDofManager21;
    AMP::shared_ptr<AMP::Discretization::DOFManager> targetDofManager12, targetDofManager21;
    AMP::shared_ptr<AMP::Database> nullDatabase;

    if(mesh1.get()!=NULL){
      boundaryMesh1 = mesh1->Subset( mesh1->getBoundaryIDIterator( AMP::Mesh::Volume, boundaryID1 ) );
    // Build map 1 -> 2
      d_SourceVectorMap12 = sourceVector->constSelect( AMP::LinearAlgebra::VS_Mesh( boundaryMesh1 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_ByVariableName( variable1 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_Stride( strideOffset1, strideLength1 ), "var" );
      sourceDofManager12 = d_SourceVectorMap12->getDOFManager(); 
    }
    if(mesh2.get()!=NULL){
      boundaryMesh2 = mesh2->Subset( mesh2->getBoundaryIDIterator( AMP::Mesh::Volume, boundaryID2 ) );
      d_TargetVectorMap12 = targetVector->select( AMP::LinearAlgebra::VS_Mesh( boundaryMesh2 ), "var" )
        ->select( AMP::LinearAlgebra::VS_ByVariableName( variable2 ), "var" )
        ->select( AMP::LinearAlgebra::VS_Stride( strideOffset2, strideLength2 ), "var" );
      targetDofManager12 = d_TargetVectorMap12->getDOFManager(); 
    }

    AMP::shared_ptr<AMP::Operator::DTKMapOperatorParameters> map12Params(
        new AMP::Operator::DTKMapOperatorParameters( nullDatabase ) );
    map12Params->d_domain_mesh = boundaryMesh1;
    map12Params->d_range_mesh  = boundaryMesh2;
    map12Params->d_domain_dofs = sourceDofManager12 ;
    map12Params->d_range_dofs  = targetDofManager12 ;
    map12Params->d_globalComm  = multiDofDTKMapOpParams->d_globalComm;                     
    d_Map12                    = AMP::shared_ptr<AMP::Operator::DTKMapOperator>(
        new AMP::Operator::DTKMapOperator( map12Params ) );

    if(mesh2.get()!=NULL){
    // Build map 2 -> 1
      d_SourceVectorMap21 = sourceVector->constSelect( AMP::LinearAlgebra::VS_Mesh( boundaryMesh2 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_ByVariableName( variable2 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_Stride( strideOffset2, strideLength2 ), "var" );
      sourceDofManager21 = d_SourceVectorMap21->getDOFManager(); 
    }
    if(mesh1.get()!=NULL){
      d_TargetVectorMap21 = targetVector->select( AMP::LinearAlgebra::VS_Mesh( boundaryMesh1 ), "var" )
        ->select( AMP::LinearAlgebra::VS_ByVariableName( variable1 ), "var" )
        ->select( AMP::LinearAlgebra::VS_Stride( strideOffset1, strideLength1 ), "var" );
      targetDofManager21 = d_TargetVectorMap21->getDOFManager(); 
    }

    AMP::shared_ptr<AMP::Operator::DTKMapOperatorParameters> map21Params(
        new AMP::Operator::DTKMapOperatorParameters( nullDatabase ) );
    map21Params->d_domain_mesh = boundaryMesh2;
    map21Params->d_range_mesh  = boundaryMesh1;
    map21Params->d_domain_dofs = sourceDofManager21;
    map21Params->d_range_dofs  = targetDofManager21;
    map21Params->d_globalComm  = multiDofDTKMapOpParams->d_globalComm;                     
    d_Map21                    = AMP::shared_ptr<AMP::Operator::DTKMapOperator>(
        new AMP::Operator::DTKMapOperator( map21Params ) );
}

void MultiDofDTKMapOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                    AMP::LinearAlgebra::Vector::shared_ptr r)
{

    AMP::Mesh::Mesh::shared_ptr mesh1 = d_multiDofDTKMapOpParams->d_Mesh1;
    AMP::Mesh::Mesh::shared_ptr mesh2 = d_multiDofDTKMapOpParams->d_Mesh2;
    int boundaryID1                   = d_multiDofDTKMapOpParams->d_BoundaryID1;
    int boundaryID2                   = d_multiDofDTKMapOpParams->d_BoundaryID2;
    std::string variable1             = d_multiDofDTKMapOpParams->d_Variable1;
    std::string variable2             = d_multiDofDTKMapOpParams->d_Variable2;
    std::size_t strideOffset1         = d_multiDofDTKMapOpParams->d_StrideOffset1;
    std::size_t strideOffset2         = d_multiDofDTKMapOpParams->d_StrideOffset2;
    std::size_t strideLength1         = d_multiDofDTKMapOpParams->d_StrideLength1;
    std::size_t strideLength2         = d_multiDofDTKMapOpParams->d_StrideLength2;

    AMP::Mesh::Mesh::shared_ptr boundaryMesh1;
    AMP::Mesh::Mesh::shared_ptr boundaryMesh2;
    AMP::shared_ptr<AMP::Database> nullDatabase;
    if(mesh1.get()!=NULL){
      boundaryMesh1 = mesh1->Subset( mesh1->getBoundaryIDIterator( AMP::Mesh::Volume, boundaryID1 ) );
    // Build map 1 -> 2
      d_SourceVectorMap12 = u->constSelect( AMP::LinearAlgebra::VS_Mesh( boundaryMesh1 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_ByVariableName( variable1 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_Stride( strideOffset1, strideLength1 ), "var" );
    }
    if(mesh2.get()!=NULL){
      boundaryMesh2 = mesh2->Subset( mesh2->getBoundaryIDIterator( AMP::Mesh::Volume, boundaryID2 ) );
    // Build map 2 -> 1
      d_SourceVectorMap21 = u->constSelect( AMP::LinearAlgebra::VS_Mesh( boundaryMesh2 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_ByVariableName( variable2 ), "var" )
        ->constSelect( AMP::LinearAlgebra::VS_Stride( strideOffset2, strideLength2 ), "var" );
    }

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    // QUESTION:  should we apply on u rather than on d_SourceVectorMapXY ?
    //            in that case we would have to perform select again
    d_Map12->apply( d_SourceVectorMap12, d_TargetVectorMap12 );
    d_Map21->apply( d_SourceVectorMap21, d_TargetVectorMap21 );
}
}
}
