#include "AMP/time_integrators/oxide/OxideModel.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_LAPACK_WRAPPERS
    #include "LapackWrappers.h"
#endif

#include <algorithm>
#include <cmath>


namespace AMP::TimeIntegrator {


/************************************************************************
 * Return the equiblibrium concentrations                                *
 ************************************************************************/
void OxideModel::get_equilibrium_concetration( double T, double *Ci )
{
    // The oxygen concentration in the oxide at the oxide-gas interface is
    // assumed to be 1.511 g/cm^3, derived from stoichiometric oxide
    Ci[0] = 1.511;

    // The oxygen concentration in the oxide at the oxide-alpha interface is
    // given in "Zirconium Metal-Water Oxidation Kinetics IV. Raction Rate
    // Studies", p. 36.
    Ci[1] = 1.517 - 7.5e-5 * T;

    // The oxygen concentration in the alpha layer at the alpha-oxide interface
    // is a fixed 29// (atomic density) or ~0.4537 g/cm^3
    // R.A. Perkins, "The Diffusion of Oxygen in Oxygen Stabilized alpha-Zirconium
    //    and Zircaloy-4", Jornal of Nuclear Materials, Vol. 73, pp 20-29 (1978).
    // "Zirconium Metal-Water Oxidation Kinetics IV. Raction Rate Studies", p. 36.
    Ci[2] = 0.4537;

    // The oxygen concentration in the alpha layer at the alpha-beta interface
    // is calculated using the equilibrium concentration
    int source = 2;
    if ( source == 1 ) {
        // "Zirconium Metal-Water Oxidation Kinetics IV. Reaction Rate Studies", p. 36.
        if ( T >= 1346 ) {
            Ci[3] = ( -0.2263 + std::sqrt( T / 63.385 - 16.877 ) ) *
                    0.0649; // Oxygen concentration in the alpha at the beta interface (g/cm^3)
        } else if ( T >= 1073 ) {
            // Just assume a fixed value for low temps for now
            Ci[3] = 0.1208; // Oxygen concentration in the alpha at the beta interface (g/cm^3)
        } else {
            AMP_ERROR( "Invalid temperature" );
        }
    } else if ( source == 2 ) {
        // R.A. Perkins, "The Diffusion of Oxygen in Oxygen Stabilized alpha-Zirconium
        //    and Zircaloy-4", Jornal of Nuclear Materials, Vol. 73, pp 20-29 (1978).
        if ( T <= 1123 ) {
            // The beta phase does not exist (or cannot have oxygen)
            Ci[3] = 0.0;
        } else {
            Ci[3] = 9.62e-3 * T - 8.17;
        }
        // Set the minimum concentration at the interface to a small (non-zero)
        // value for stability
        Ci[3] = std::max( Ci[3], 1e-3 );
    } else {
        AMP_ERROR( "Invalid source" );
    }

    // The oxygen concentration in the beta layer at the beta-alpha interface
    // is calculated using the equilibrium concentration
    if ( source == 1 ) {
        // "Zirconium Metal-Water Oxidation Kinetics IV. Raction Rate Studies", p. 36.
        if ( T >= 1646 ) {
            Ci[4] = ( ( T - 1081.7 ) / 491.159 ) *
                    0.0649; // Oxygen concentration in the alpha at the beta interface (g/cm^3)
        } else if ( T >= 1506 ) {
            Ci[4] = ( -0.00428 + std::sqrt( T / 392.46 - 3.1417 ) ) *
                    0.0649; // Oxygen concentration in the alpha at the beta interface (g/cm^3)
        } else {
            AMP_ERROR( "Invalid temperature" );
        }
    } else if ( source == 2 ) {
        // R.A. Perkins, "The Diffusion of Oxygen in Oxygen Stabilized alpha-Zirconium
        //    and Zircaloy-4", Jornal of Nuclear Materials, Vol. 73, pp 20-29 (1978).
        if ( T <= 1123 ) {
            // The beta phase does not exist (or cannot have oxygen)
            Ci[4] = 0;
        } else {
            AMP_ERROR( "Not programmed yet" );
        }
    } else {
        AMP_ERROR( "Invalid source" );
    }

    // The concentration at the Zirconium at the bulk layer is zero
    Ci[5] = 0.0;
}


/************************************************************************
 * Return the diffusion coefficients                                     *
 ************************************************************************/
void OxideModel::get_diffusion_coefficients( double T, double *D )
{
    const double R = 1.077 * 9 / 5;            // Convert Kelvin to Rankine
    D[0] = 0.1387 * exp( -34680 / ( R * T ) ); // Diffusivity of oxygen in the oxide layer (cm^2/s)
    D[1] = 3.9230 * exp( -51000 / ( R * T ) ); // Diffusivity of oxygen in the alpha layer (cm^2/s)
    D[2] = 0.0263 * exp( -28200 / ( R * T ) ); // Diffusivity of oxygen in the beta layer (cm^2/s)
}


/************************************************************************
 * Function to integrate the oxide over a fixed timestep                 *
 ************************************************************************/
int OxideModel::integrateOxide( double dT,
                                int N,
                                const int *N2,
                                const double *x0,
                                const double *Cb,
                                double **C0,
                                const double *D,
                                double **C1,
                                double *x1,
                                double *v1 )
{
    // Allocate some temporary memory
    double x2[2];
    int N2_max = 0;
    for ( int i = 0; i < N; i++ )
        N2_max = std::max( N2_max, N2[i] );
    auto C2 = new double[N2_max];
    // Copy the current solution
    for ( int i = 0; i < N + 1; i++ ) {
        x1[i] = x0[i];
    }
    for ( int i = 0; i < N; i++ ) {
        for ( int j = 0; j < N2[i]; j++ )
            C1[i][j] = C0[i][j];
    }
    // Compute the initial velocity
    computeVelocity( N, x1, Cb, N2, C1, D, v1 );
    // Iterate until we reach the final solution
    double t = 0;
    int N_it = 0;
    while ( t < dT ) {
        // Compute the current velocity
        computeVelocity( N, x1, Cb, N2, C1, D, v1 );
        // Compute the maximum safe timestep
        double dt = dT - t;
        for ( int i = 0; i < N; i++ ) {
            double h = ( x1[i + 1] - x1[i + 0] ) / N2[i];
            dt       = std::min( dt, fabs( 0.5 * h / v1[i + 0] ) );
            dt       = std::min( dt, fabs( 0.5 * h / v1[i + 1] ) );
            double dt_Diff =
                computeDiffustionTimestep( N2[i], &x1[i + 0], &Cb[2 * i], C1[i], D[i] );
            dt = std::min( dt, dt_Diff );
        }
        if ( dt <= 1e-15 )
            AMP_ERROR( "invalid dt found" );
        // Advance the solution by the timestep
        for ( int i = 0; i < N; i++ ) {
            solveLinearDiffusionLayer( N2[i], dt, &x1[i], &v1[i], &Cb[2 * i], C1[i], D[i], C2, x2 );
            for ( int j = 0; j < N2[i]; j++ )
                C1[i][j] = C2[j];
        }
        // Move the boundaries and update the timestep
        for ( int i = 0; i < N + 1; i++ )
            x1[i] += dt * v1[i];
        t += dt;
        if ( ( x1[N] - x1[N - 1] ) < 1e-6 * ( x1[N - 1] - x1[N - 2] ) ) {
            // The final layer has been depleted
            AMP_ERROR( "Final layer depleted" );
        }
        N_it++;
        if ( N_it > 1e6 ) {
            AMP_ERROR( "maximum number of iterations exceeded" );
        }
    }
    // Free temporary memory
    delete[] C2;
    return N_it;
}


/************************************************************************
 * Compute the velocity of the zone boundaries                           *
 ************************************************************************/
void OxideModel::computeVelocity( const int N,
                                  const double *x,
                                  const double *Cb,
                                  const int *Nl,
                                  double const *const *C,
                                  const double *D,
                                  double *v )
{
    // Compute the flux at each boundary layer
    double flux[100]; // Static memory of 50 layers
    for ( int i = 0; i < N; i++ ) {
        double h = 0.5 * ( x[i + 1] - x[i] ) /
                   ( (double) Nl[i] ); // Spacing between endpoints and nearest zone (0.5*h0)
        flux[2 * i + 0] = -D[i] * ( C[i][0] - Cb[2 * i + 0] ) / h;
        flux[2 * i + 1] = -D[i] * ( Cb[2 * i + 1] - C[i][Nl[i] - 1] ) / h;
    }
    // Compute the velocity of the interfaces
    v[0] = 0.0;
    for ( int i = 1; i < N; i++ )
        v[i] = ( flux[2 * i] - flux[2 * i - 1] ) / ( Cb[2 * i] - Cb[2 * i - 1] );
    v[N] = 0.0;
}


/************************************************************************
 * Compute the maximum timestep based on diffusion                       *
 ************************************************************************/
double OxideModel::computeDiffustionTimestep(
    const int N, const double x[2], const double Cb[2], const double *C, const double D )
{
    // Limit the timestep to a 20% change in C
    double tol = 0.20;
    double dt  = 1e100;
    double h   = ( x[1] - x[0] ) / N;
    for ( int i = 0; i < N; i++ ) {
        double tmp = 0.0;
        if ( i == 0 ) {
            tmp = D / ( h * h ) * ( 2.0 * Cb[0] - 3.0 * C[i] + C[i + 1] );
        } else if ( i == N - 1 ) {
            tmp = D / ( h * h ) * ( C[i - 1] - 3.0 * C[i] + 2.0 * Cb[1] );
        } else {
            tmp = D / ( h * h ) * ( C[i - 1] - 2.0 * C[i] + C[i + 1] );
        }
        dt = std::min( dt, tol / fabs( tmp ) );
    }
    return dt;
}


/************************************************************************
 * Perform a linear solve to get the future oxygen concentration         *
 ************************************************************************/
void OxideModel::solveLinearDiffusionLayer( const int N,
                                            const double dt,
                                            const double x0[2],
                                            const double v[2],
                                            const double Cb[2],
                                            const double *C0,
                                            const double D,
                                            double *C1,
                                            double *x1 )
{
    // Allocate memory for internal variables
    auto mem      = new double[4 * N + 1]; // We want a single block for cache access
    double *diag  = &mem[0];
    double *lower = &mem[N];
    double *upper = &mem[2 * N];
    double *Db    = &mem[3 * N];
    double *rhs   = C1;
    // Fill the diffusion coefficients at zone boundaries
    for ( int i = 0; i < N + 1; i++ )
        Db[i] = D;
    // Compute x1
    x1[0] = x0[0] + dt * v[0];
    x1[1] = x0[1] + dt * v[1];
    // Compute h0 and h1
    double h0 = ( x0[1] - x0[0] ) / N;
    double h1 = ( x1[1] - x1[0] ) / N;
    // Compute the rhs
    double Nd = N;
    double vi = v[0] + 0.5 / Nd * ( v[1] - v[0] );
    rhs[0] =
        C0[0] +
        0.5 * dt / ( h0 * h0 ) * ( Db[1] * ( C0[1] - C0[0] ) - Db[0] * 2 * ( C0[0] - Cb[0] ) ) +
        0.5 * dt / ( h1 * h1 ) * Db[0] * 2 * Cb[0];
    if ( vi >= 0 )
        rhs[0] += 0.5 * dt / h0 * vi * ( C0[1] - C0[0] );
    else
        rhs[0] += 0.5 * dt / h0 * vi * 2 * ( C0[0] - Cb[0] ) - 0.5 * dt / h1 * vi * 2 * Cb[0];
    for ( int i = 1; i < N - 1; i++ ) {
        vi     = v[0] + ( i + 0.5 ) / Nd * ( v[1] - v[0] );
        rhs[i] = C0[i] + 0.5 * dt / ( h0 * h0 ) *
                             ( Db[i + 1] * ( C0[i + 1] - C0[i] ) - Db[i] * ( C0[i] - C0[i - 1] ) );
        if ( vi >= 0 )
            rhs[i] += 0.5 * dt / h0 * vi * ( C0[i + 1] - C0[i] );
        else
            rhs[i] += 0.5 * dt / h0 * vi * ( C0[i] - C0[i - 1] );
    }
    vi         = v[0] + ( Nd - 0.5 ) / Nd * ( v[1] - v[0] );
    rhs[N - 1] = C0[N - 1] +
                 0.5 * dt / ( h0 * h0 ) *
                     ( Db[N] * 2 * ( Cb[1] - C0[N - 1] ) - Db[N] * ( C0[N - 1] - C0[N - 2] ) ) +
                 0.5 * dt / ( h1 * h1 ) * Db[N] * 2 * Cb[1];
    if ( vi >= 0 )
        rhs[N - 1] +=
            0.5 * dt / h0 * vi * 2 * ( Cb[1] - C0[N - 1] ) + 0.5 * dt / h1 * vi * 2 * Cb[1];
    else
        rhs[N - 1] += 0.5 * dt / h0 * vi * ( C0[N - 1] - C0[N - 2] );
    // Compute the matrix
    vi = v[0] + 0.5 / Nd * ( v[1] - v[0] );
    if ( vi >= 0 ) {
        diag[0]  = 1.0 + 0.5 * dt / ( h1 * h1 ) * ( Db[1] + 2.0 * Db[0] ) + 0.5 * dt / h1 * vi;
        upper[0] = -0.5 * dt / ( h1 * h1 ) * Db[1] - 0.5 * dt / h1 * vi;
    } else {
        diag[0]  = 1.0 + 0.5 * dt / ( h1 * h1 ) * ( Db[1] + 2.0 * Db[0] ) - 0.5 * dt / h1 * vi;
        upper[0] = -0.5 * dt / ( h1 * h1 ) * Db[1];
    }
    for ( int i = 1; i < N - 1; i++ ) {
        vi = v[0] + ( i + 0.5 ) / Nd * ( v[1] - v[0] );
        if ( vi >= 0 ) {
            diag[i] = 1.0 + 0.5 * dt / ( h1 * h1 ) * ( Db[i + 1] + Db[i] ) + 0.5 * dt / h1 * vi;
            lower[i - 1] = -0.5 * dt / ( h1 * h1 ) * Db[i];
            upper[i]     = -0.5 * dt / ( h1 * h1 ) * Db[i + 1] - 0.5 * dt / h1 * vi;
        } else {
            diag[i] = 1.0 + 0.5 * dt / ( h1 * h1 ) * ( Db[i + 1] + Db[i] ) - 0.5 * dt / h1 * vi;
            lower[i - 1] = -0.5 * dt / ( h1 * h1 ) * Db[i] + 0.5 * dt / h1 * vi;
            upper[i]     = -0.5 * dt / ( h1 * h1 ) * Db[i + 1];
        }
    }
    vi = v[0] + ( Nd - 0.5 ) / Nd * ( v[1] - v[0] );
    if ( vi >= 0 ) {
        diag[N - 1] =
            1.0 + 0.5 * dt / ( h1 * h1 ) * ( 2.0 * Db[N] + Db[N - 1] ) + 0.5 * dt / h1 * vi;
        lower[N - 2] = -0.5 * dt / ( h1 * h1 ) * Db[N - 1];
    } else {
        diag[N - 1] =
            1.0 + 0.5 * dt / ( h1 * h1 ) * ( 2.0 * Db[N] + Db[N - 1] ) - 0.5 * dt / h1 * vi;
        lower[N - 2] = -0.5 * dt / ( h1 * h1 ) * Db[N - 1] + 0.5 * dt / h1 * vi;
    }
    // Solve the system
#ifdef AMP_USE_LAPACK_WRAPPERS
    int error = 0;
    Lapack<double>::gtsv( N, 1, lower, diag, upper, rhs, N, error );
    if ( error != 0 ) {
        printf( "Error solving tridiagonal system\n" );
    }
    // Free memory
    delete[] mem;
#else
    AMP_ERROR( "Lapack required" );
#endif
}


} // namespace AMP::TimeIntegrator
