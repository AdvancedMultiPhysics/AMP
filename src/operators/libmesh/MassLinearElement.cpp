#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include "libmesh/point.h"


namespace AMP::Operator {

void MassLinearElement::apply()
{
    PROFILE( "apply", 5 );

    const std::vector<libMesh::Real> &JxW = ( *d_JxW );

    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );

    std::vector<std::vector<double>> &elementMassMatrix = ( *d_elementMassMatrix );

    d_fe->reinit( d_elem );

    d_densityModel->preLinearElementOperation();

    const unsigned int num_local_dofs = d_elem->n_nodes();

    std::vector<double> density( d_qrule->n_points() );

    d_densityModel->preLinearGaussPointOperation();

    d_equation = d_densityModel->getEquation();

    if ( d_densityAtGauss ) {
        std::vector<double> temperature( d_qrule->n_points() );
        std::vector<double> concentration( d_qrule->n_points() );
        std::vector<double> burnup( d_qrule->n_points() );
        for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
            temperature[qp]   = 0.0;
            concentration[qp] = 0.0;
            burnup[qp]        = 0.0;
            for ( unsigned int j = 0; j < num_local_dofs; j++ ) {
                temperature[qp] += d_LocalTemperature[j] * phi[j][qp];
                concentration[qp] += d_LocalConcentration[j] * phi[j][qp];
                burnup[qp] += d_LocalBurnup[j] * phi[j][qp];
            } // end for j
        }     // end for qp

        switch ( d_equation ) {
        case MassDensityModel::MassEquation::Mechanics:
            d_densityModel->getDensityMechanics( density, temperature, concentration, burnup );
            break;
        case MassDensityModel::MassEquation::Thermal:
            d_densityModel->getDensityThermal( density, temperature, concentration, burnup );
            break;
        case MassDensityModel::MassEquation::Chemical:
            d_densityModel->getDensityChemical( density, temperature, concentration, burnup );
            break;
        case MassDensityModel::MassEquation::Manufactured:
            d_densityModel->getDensityManufactured(
                density, temperature, concentration, burnup, d_fe->get_xyz() );
            break;
        default:
            AMP_ERROR( "Unknown enum for d_equation" );
        }
    } else {
        std::vector<double> nodalDensity( num_local_dofs );
        std::vector<libMesh::Point> elem_nodes;
        switch ( d_equation ) {
        case MassDensityModel::MassEquation::Mechanics:
            d_densityModel->getDensityMechanics(
                nodalDensity, d_LocalTemperature, d_LocalConcentration, d_LocalBurnup );
            break;
        case MassDensityModel::MassEquation::Thermal:
            d_densityModel->getDensityThermal(
                nodalDensity, d_LocalTemperature, d_LocalConcentration, d_LocalBurnup );
            break;
        case MassDensityModel::MassEquation::Chemical:
            d_densityModel->getDensityChemical(
                nodalDensity, d_LocalTemperature, d_LocalConcentration, d_LocalBurnup );
            break;
        case MassDensityModel::MassEquation::Manufactured:
            elem_nodes.resize( num_local_dofs );
            for ( size_t i = 0; i < num_local_dofs; i++ )
                elem_nodes[i] = d_elem->point( i );
            d_densityModel->getDensityManufactured(
                nodalDensity, d_LocalTemperature, d_LocalConcentration, d_LocalBurnup, elem_nodes );
            break;
        default:
            AMP_ERROR( "Unknown enum for d_equation" );
        }

        for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
            density[qp] = 0.0;
            for ( unsigned int j = 0; j < num_local_dofs; j++ ) {
                density[qp] += nodalDensity[j] * phi[j][qp];
            } // end for j
        }     // end for qp
    }

    for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {

        d_densityModel->preLinearGaussPointOperation();

        for ( unsigned int j = 0; j < num_local_dofs; j++ ) {
            for ( unsigned int k = 0; k < num_local_dofs; k++ ) {
                elementMassMatrix[j][k] += ( density[qp] * JxW[qp] * ( phi[j][qp] * phi[k][qp] ) );
            } // end for k
        }     // end for j

        d_densityModel->postLinearGaussPointOperation();

    } // end for qp

    d_densityModel->postLinearElementOperation();
}
} // namespace AMP::Operator
