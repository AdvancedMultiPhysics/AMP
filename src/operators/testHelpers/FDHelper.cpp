#include "FDHelper.h"

std::array<double, 3> getDiscreteNorms( const std::vector<double> &h,
                                        std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{

    double vol = 1.0;
    for ( auto hk : h ) {
        vol *= hk;
    }

    // Compute norms
    double uL1Norm  = static_cast<double>( u->L1Norm() ) * vol;
    double uL2Norm  = static_cast<double>( u->L2Norm() ) * std::pow( vol, 0.5 );
    double uMaxNorm = static_cast<double>( u->maxNorm() );

    std::array<double, 3> unorms = { uL1Norm, uL2Norm, uMaxNorm };
    return unorms;
}

void fillMultiVectorWithFunction(
    std::shared_ptr<const AMP::Mesh::Mesh> Mesh,
    AMP::Mesh::GeomType geom,
    std::shared_ptr<const AMP::Discretization::DOFManager> scalarDOFMan,
    std::shared_ptr<AMP::LinearAlgebra::Vector> vec_,
    const std::function<double( size_t component, AMP::Mesh::Point &point )> &fun )
{

    // Unpack multiVector
    auto vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec_ );
    AMP_INSIST( vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto vec0 = vec->getVector( 0 );
    auto vec1 = vec->getVector( 1 );

    double u0, u1;
    auto it = Mesh->getIterator( geom ); // Mesh iterator
    for ( auto elem = it.begin(); elem != it.end(); elem++ ) {
        auto point = elem->centroid();
        u0         = fun( 0, point );
        u1         = fun( 1, point );
        std::vector<size_t> i;
        scalarDOFMan->getDOFs( elem->globalID(), i );
        vec0->setValueByGlobalID( i[0], u0 );
        vec1->setValueByGlobalID( i[0], u1 );
    }
    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}