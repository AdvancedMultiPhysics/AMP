#include "AMP/operators/diffusion/DiffusionLinearElement.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


void DiffusionLinearElement::apply()
{
    PROFILE( "apply", 5 );

    const auto &JxW  = ( *d_JxW );
    const auto &phi  = ( *d_phi );
    const auto &dphi = ( *d_dphi );

    auto &elementStiffnessMatrix = ( *d_elementStiffnessMatrix );

    d_fe->reinit( d_elem );

    const auto &q_point = d_fe->get_xyz();

    d_transportModel->preLinearElementOperation();

    size_t N_dofs = d_elem->n_nodes();

    std::vector<double> conductivity( d_qrule->n_points() );
    AMP::Array<std::shared_ptr<std::vector<double>>> conductivityTensor( 3, 3 );
    if ( d_transportModel->isaTensor() ) {
        d_transportTensorModel =
            std::dynamic_pointer_cast<DiffusionTransportTensorModel>( d_transportModel );
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ )
                conductivityTensor( i, j ) =
                    std::make_shared<std::vector<double>>( d_qrule->n_points() );
        }
    }

    if ( d_transportAtGauss ) {
        std::map<std::string, std::shared_ptr<std::vector<double>>> args;
        for ( const auto &[name, vec] : d_localVecs ) {
            auto vec2 = std::make_shared<std::vector<double>>( d_qrule->n_points(), 0.0 );
            for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
                ( *vec2 )[qp] = 0.0;
                for ( unsigned int j = 0; j < N_dofs; j++ ) {
                    ( *vec2 )[qp] += vec[j] * phi[j][qp];
                }
            }
            args[name] = vec2;
        }
        if ( !d_transportModel->isaTensor() ) {
            d_transportModel->getTransport( conductivity, args, q_point );
        } else {
            d_transportTensorModel->getTensorTransport( conductivityTensor, args, q_point );
        }
    } else {
        std::map<std::string, std::shared_ptr<std::vector<double>>> args;
        for ( const auto &[name, vec] : d_localVecs ) {
            args[name] = std::shared_ptr<std::vector<double>>(
                const_cast<std::vector<double> *>( &vec ), []( auto ) {} );
        }

        std::vector<double> nodalConductivity( N_dofs );
        AMP::Array<std::shared_ptr<std::vector<double>>> nodalConductivityTensor( 3, 3 );
        if ( !d_transportModel->isaTensor() ) {
            d_transportModel->getTransport( nodalConductivity, args, q_point );
        } else {
            for ( int i = 0; i < 3; i++ )
                for ( int j = 0; j < 3; j++ ) {
                    nodalConductivityTensor( i, j ) =
                        std::make_shared<std::vector<double>>( N_dofs );
                }
            d_transportTensorModel->getTensorTransport( nodalConductivityTensor, args, q_point );
        }

        for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
            conductivity[qp] = 0.0;
            if ( !d_transportModel->isaTensor() ) {
                for ( unsigned int n = 0; n < N_dofs; n++ ) {
                    conductivity[qp] += nodalConductivity[n] * phi[n][qp];
                } // end for n
            } else {
                for ( unsigned int n = 0; n < N_dofs; n++ ) {
                    for ( int i = 0; i < 3; i++ )
                        for ( int j = 0; j < 3; j++ ) {
                            ( *conductivityTensor( i, j ) )[qp] +=
                                ( *nodalConductivityTensor( i, j ) )[n] * phi[n][qp];
                        }
                } // end for n
            }
        } // end for qp
    }

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
        d_transportModel->preLinearGaussPointOperation();

        if ( !d_transportModel->isaTensor() ) {
            for ( unsigned int n = 0; n < N_dofs; n++ ) {
                for ( unsigned int k = 0; k < N_dofs; k++ ) {
                    elementStiffnessMatrix[n][k] +=
                        ( JxW[qp] * conductivity[qp] * ( dphi[n][qp] * dphi[k][qp] ) );
                } // end for k
            }     // end for n
        } else {
            for ( unsigned int n = 0; n < N_dofs; n++ ) {
                for ( unsigned int k = 0; k < N_dofs; k++ ) {
                    for ( int i = 0; i < 3; i++ )
                        for ( int j = 0; j < 3; j++ ) {
                            elementStiffnessMatrix[n][k] +=
                                ( JxW[qp] * ( *conductivityTensor( i, j ) )[qp] *
                                  ( dphi[n][qp]( i ) * dphi[k][qp]( j ) ) );
                        }
                } // end for k
            }     // end for n
        }

        d_transportModel->postLinearGaussPointOperation();
    } // end for qp

    d_transportModel->postLinearElementOperation();
}
} // namespace AMP::Operator
