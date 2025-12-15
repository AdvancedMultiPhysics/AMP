#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"

#include "ProfilerApp.h"

#include <map>
#include <vector>


namespace AMP::Operator {


void DiffusionNonlinearElement::initTransportModel()
{
    d_fe->reinit( d_elem );

    d_transportModel->preNonlinearInitElementOperation();

    for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_transportModel->nonlinearInitGaussPointOperation();
    } // end for qp

    d_transportModel->postNonlinearInitElementOperation();
}


void DiffusionNonlinearElement::apply()
{
    if ( d_transportModel->isaTensor() )
        applyTensor();
    else
        applyScalar();
}
void DiffusionNonlinearElement::applyScalar()
{
    PROFILE( "applyScalar", 5 );

    const auto &JxW  = *d_JxW;
    const auto &phi  = *d_phi;
    const auto &dphi = *d_dphi;
    size_t num_nodes = d_elem->n_nodes();

    // Reinitialize the element
    {
        PROFILE( "applyScalar-reinit", 6 );
        d_fe->reinit( d_elem );
    }

    // compute transport coefficients
    std::vector<double> transportCoeff( d_qrule->n_points() );
    std::map<std::string, std::shared_ptr<std::vector<double>>> transport_args;
    if ( d_transportAtGauss ) {
        // construct material evalv arguments
        PROFILE( "applyScalar-transport (guass)", 6 );
        const auto &q_point = d_fe->get_xyz();
        for ( auto &[name, vec] : d_elementInputVectors ) {
            auto values = std::make_shared<std::vector<double>>( d_qrule->n_points() );
            for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
                double x = 0;
                for ( size_t j = 0; j < num_nodes; j++ )
                    x += vec[j] * phi[j][qp];
                ( *values )[qp] = x;
            }
            transport_args[name] = values;
        }
        d_transportModel->getTransport( transportCoeff, transport_args, q_point );
    } else {
        // get element nodes
        PROFILE( "applyScalar-transport (nodes)", 6 );
        std::vector<libMesh::Point> elem_nodes( num_nodes );
        for ( size_t i = 0; i < num_nodes; i++ ) {
            elem_nodes[i] = d_elem->point( i );
        }

        // construct material evalv arguments
        for ( auto &[name, vec] : d_elementInputVectors ) {
            transport_args[name] = std::shared_ptr<std::vector<double>>( &vec, []( auto ) {} );
        }

        // evaluate
        std::vector<double> nodalTransportCoeff( num_nodes );
        d_transportModel->getTransport( nodalTransportCoeff, transport_args, elem_nodes );

        // interpolate to gauss points
        for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
            double coeff = 0.0;
            for ( size_t n = 0; n < num_nodes; n++ )
                coeff += nodalTransportCoeff[n] * phi[n][qp];
            transportCoeff[qp] = coeff;
        }
    }

    PROFILE( "applyScalar-apply", 5 );
    auto &primaryInputVec = d_elementInputVectors[d_PrincipalVariable];
    for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
        libMesh::RealGradient grad_phi = 0.0;

        for ( size_t n = 0; n < num_nodes; n++ )
            grad_phi += dphi[n][qp] * primaryInputVec[n];

        d_transportModel->preNonlinearAssemblyGaussPointOperation();

        for ( size_t n = 0; n < num_nodes; n++ ) {
            ( *d_elementOutputVector )[n] += JxW[qp] * transportCoeff[qp] * dphi[n][qp] * grad_phi;
        } // end for n

        d_transportModel->postNonlinearAssemblyGaussPointOperation();
    }

    PROFILE( "applyScalar-5", 5 );
    d_transportModel->postNonlinearAssemblyElementOperation();
}
void DiffusionNonlinearElement::applyTensor()
{
    PROFILE( "applyTensor", 5 );

    const auto &JxW  = *d_JxW;
    const auto &phi  = *d_phi;
    const auto &dphi = *d_dphi;

    d_fe->reinit( d_elem );

    const size_t num_nodes = d_elem->n_nodes();

    // create transport coefficient storage
    AMP::Array<std::shared_ptr<std::vector<double>>> transportCoeffTensor( 3, 3 );
    d_transportTensorModel =
        std::dynamic_pointer_cast<DiffusionTransportTensorModel>( d_transportModel );
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ )
            transportCoeffTensor( i, j ) =
                std::make_shared<std::vector<double>>( d_qrule->n_points() );
    }

    // compute transport coefficients
    std::map<std::string, std::shared_ptr<std::vector<double>>> transport_args;

    // at gauss points
    if ( d_transportAtGauss ) {
        // construct material evalv arguments
        const auto &q_point = d_fe->get_xyz();
        for ( auto &[name, vec] : d_elementInputVectors ) {
            auto values = std::make_shared<std::vector<double>>( d_qrule->n_points() );
            for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
                ( *values )[qp] = 0.0;
                for ( size_t j = 0; j < num_nodes; j++ )
                    ( *values )[qp] += vec[j] * phi[j][qp];
            }
            transport_args[name] = values;
        }
        d_transportTensorModel->getTensorTransport( transportCoeffTensor, transport_args, q_point );
    } else {
        // get element nodes
        std::vector<libMesh::Point> elem_nodes( num_nodes );
        for ( size_t i = 0; i < num_nodes; i++ ) {
            elem_nodes[i] = d_elem->point( i );
        }

        // set up storage for transport coefficients
        AMP::Array<std::shared_ptr<std::vector<double>>> nodalTransportCoeffTensor( 3, 3 );

        // construct material evalv arguments
        for ( auto &[name, vec] : d_elementInputVectors ) {
            transport_args[name] = std::shared_ptr<std::vector<double>>( &vec, []( auto ) {} );
        }

        // evaluate
        for ( int i = 0; i < 3; i++ )
            for ( int j = 0; j < 3; j++ ) {
                nodalTransportCoeffTensor( i, j ) =
                    std::make_shared<std::vector<double>>( num_nodes );
            }
        d_transportTensorModel->getTensorTransport(
            nodalTransportCoeffTensor, transport_args, elem_nodes );

        // interpolate to gauss points
        for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
            for ( int i = 0; i < 3; i++ )
                for ( int j = 0; j < 3; j++ )
                    ( *transportCoeffTensor( i, j ) )[qp] = 0.0;
            for ( size_t n = 0; n < num_nodes; n++ ) {
                for ( int i = 0; i < 3; i++ )
                    for ( int j = 0; j < 3; j++ ) {
                        ( *transportCoeffTensor( i, j ) )[qp] +=
                            ( *nodalTransportCoeffTensor( i, j ) )[n] * phi[n][qp];
                    }
            } // end for j
        }     // end for qp
    }

    auto &primaryInputVec = d_elementInputVectors[d_PrincipalVariable];
    for ( size_t qp = 0; qp < d_qrule->n_points(); qp++ ) {
        libMesh::RealGradient grad_phi = 0.0;

        for ( size_t n = 0; n < num_nodes; n++ )
            grad_phi += dphi[n][qp] * primaryInputVec[n];

        d_transportModel->preNonlinearAssemblyGaussPointOperation();

        for ( size_t n = 0; n < num_nodes; n++ ) {
            for ( int i = 0; i < 3; i++ )
                for ( int j = 0; j < 3; j++ ) {
                    ( *d_elementOutputVector )[n] += JxW[qp] *
                                                     ( *transportCoeffTensor( i, j ) )[qp] *
                                                     dphi[n][qp]( i ) * grad_phi( j );
                }
        } // end for n

        d_transportModel->postNonlinearAssemblyGaussPointOperation();
    } // end for qp

    d_transportModel->postNonlinearAssemblyElementOperation();
}


} // namespace AMP::Operator
