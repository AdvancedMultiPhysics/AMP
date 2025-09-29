import sympy as sym

"""This is a simple Python script for generating C++ code for exact solutions and corresponding source terms of heat equation problems in 1D, 2D, and 3D.

Specifically, this file provides C++ expressions for 'u' and 'Lu' corresponding to both the 'uzero' and 'ubnd' functions that are apart of the ManufacturedHeatEquationModel class in ../heatEquationFD/testImplicitIntegrationWithHeatEquationFD.cpp. 

To run this script from the command line:
    >> python3 ManufacturedHeatEquationModel-helper.py

Running this script will print the aforementioned C++ expressions for 1, 2, and 3 spatial dimensions to the console. For each spatial dimension, this console output is then copy pasted into the functions (in the dimension-appropriate places):
        ManufacturedHeatEquationModel::ubnd()
        ManufacturedHeatEquationModel::Lubnd()
        ManufacturedHeatEquationModel::uzero()
        ManufacturedHeatEquationModel::Luzero()

Dependencies: sympy (used for symbolic computation, including differentiation)
"""


def main():
    spatialdim = [1, 2, 3]
    for dim in spatialdim:
        heatEquation( dim )
    
def heatEquation( dim ):
    x, y, z = sym.symbols('x, y, z')
    cxx, cyy, cyx, czz, czx, czy = sym.symbols('d_cxx, d_cyy, d_cyx, d_czz, d_czx, d_czy')

    uzero_  = uzero( dim, x, y, z )
    Luzero_ = applyLu( dim, x, y, z, cxx, cyy, cyx, czz, czx, czy, uzero_ )
    ubnd_   = ubnd( dim, x, y, z )
    Lubnd_  = applyLu( dim, x, y, z, cxx, cyy, cyx, czz, czx, czy, ubnd_ )

    print("\n\n\n=================================")
    print("Heat equation spatial dimension={}".format( dim ))
    print("=================================\n")
    print("cxx expressions for 'uzero' and 'Luzero'")
    print("----------------------------------------")
    cxx_print( uzero_, Luzero_ )

    print("\ncxx expressions for 'ubnd' and 'Lubnd'")
    print("--------------------------------------")
    cxx_print( ubnd_, Lubnd_ )

def cxx_print(u_, Lu_):
    print( "u = ", sym.cxxcode( u_ ), ";" , sep = "")
    print( "Lu = ", sym.cxxcode( Lu_ ), ";", sep = "" )


def uzero( dim, x, y, z ):
    pi = sym.symbols('pi')
    d_Kx, d_Ky, d_Kz = sym.symbols('d_K[0], d_K[1], d_K[2]')
    d_PHIx, d_PHIy, d_PHIz = sym.symbols('d_PHI[0], d_PHI[1], d_PHI[2]')
    u = sym.sin( d_Kx * x + d_PHIx )
    if (dim >= 2):
        u *= sym.sin( d_Ky * y + d_PHIy )
        if ( dim >= 3 ):
            u *= sym.sin( d_Kz * z + d_PHIz )
    return u

def ubnd( dim, x, y, z ):
    u = sym.cos( x )
    if (dim >= 2):
        u *= sym.cos( y )
        if ( dim >= 3 ):
            u *= sym.cos( z )
    return u

def applyLu( dim, x, y, z, cxx, cyy, cyx, czz, czx, czy, u ):
    Lu = cxx*sym.diff( u, x, x )
    if ( dim >= 2 ):
        Lu += cyy*sym.diff( u, y, y ) + cyx*sym.diff( u, y, x )
        if ( dim >= 3 ):
            Lu += czz*sym.diff( u, z, z ) + czx*sym.diff( u, z, x ) + czy*sym.diff( u, z, y )
    return Lu


# Call the main method!
if __name__ == "__main__":
    main()