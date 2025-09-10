""" This is a simple python script to generating maunfactured solutions to a class of radiation diffusion problems and then translate those solutions (and related quantities) into C++ interpretable code. 

To run this script from the command line:
    >> python3 RadiationDiffusionModel-helper.py

This will run the "main" function, outputting to the console C++ expressions for
    1. the exact solution
    2. the gradient of the exact solutions
    3. the corresponding source term in the PDE
for the given problem dimension, and PDE model that's uncommented in the main function.

For dimension X, these C++ expressions are then copy pasted into the (model-appropriate places of the) following functions
    1. Manufactured_RadDifModel::exactSolutionXD
    2. Manufactured_RadDifModel::exactSolutionGradientXD
    3. Manufactured_RadDifModel::sourceTermXD

Dependencies: sympy (used for symbolic computation, including differentiation)
"""

import sympy as sym
import numpy as np

# Plotting stuff
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator

params = {'legend.fontsize': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18}
plt.rcParams.update(params)


def main():

    # Just uncomment whichever one you want to use
    manufacturedModel( 1,  "linear" )
    # manufacturedModel( 1,  "nonlinear" )

    # manufacturedModel( 2,  "linear" )
    # manufacturedModel( 2,  "nonlinear" )

    #manufacturedModel( 3,  "linear" )
    #manufacturedModel( 3,  "nonlinear" )

    #solveForGhostFromRobinBC()


# Generates manufactured solution for a problem in "dim" spatial dimensions and to a "model" PDE 
# that is either "linear" or "nonlinear"
#
# A pair of linear PDEs of the form
# dE/dt - nabla dot ( k11 * nabla E ) - k12 * ( T - E ) = s_E
# dT/dt - nabla dot ( k21 * nabla T ) + k22 * ( T - E ) = s_T
#
# A pair of nonlinear PDEs of the form
# dE/dt - nabla dot ( k11 * DE * nabla E ) - k12 * sigma * ( T^4 - E ) = s_E
# dT/dt - nabla dot ( k21 * DT * nabla T ) + k22 * sigma * ( T^4 - E ) = s_T
# 
# where:
#   k_ij are constants.
#   The source terms s_E and s_T are different for each system 
#
# Note we assume zatom is constant.
def manufacturedModel( dim, model ):
    
    print("-----------------------------")
    print("    {} model {}D    ".format(model, dim))
    print("-----------------------------", sep="\n")

    x, y, z, PI, zatom, k11, k12, k21, k22, t = sym.symbols('x y z PI zatom k11 k12 k21 k22 t')

    # Exact solutions
    if dim == 1:
        E, T = manufacturedSolutions1D(t, x) 
    elif dim == 2:
        E, T = manufacturedSolutions2D(t, x, y) 
    elif dim == 3:
        E, T = manufacturedSolutions3D(t, x, y, z) 

    # Get reaction term R
    if model == "nonlinear":
        sigma = (zatom/T)**3
        DE = 1/(3*sigma)
        DT = T**sym.Rational(5, 2)
        R = sigma*( T**4 - E )
    elif model == "linear":
        DE = 1
        DT = 1
        R = ( T - E )
    else:
        raise ValueError( "Invalid model" )

    # Compute the LHS of the equations, LE and LT
    # Reaction operator
    LE = - k12 * R
    LT = + k22 * R

    # Spatial differential operator applied to exact solution
    LE += -k11*( sym.diff( DE*sym.diff(E, x), x ) ) 
    LT += -k21*( sym.diff( DT*sym.diff(T, x), x ) )
    if dim >= 2:
        LE += -k11*( sym.diff( DE*sym.diff(E, y), y ) )
        LT += -k21*( sym.diff( DT*sym.diff(T, y), y ) )
    if dim >= 3:
        LE += -k11*( sym.diff( DE*sym.diff(E, z), z ) )
        LT += -k21*( sym.diff( DT*sym.diff(T, z), z ) )

    # Add time derivative of solutions
    LE += sym.diff( E, t )
    LT += sym.diff( T, t )

    # Translate into C++ code
    cxx_print(dim, x, y, z, E, T, LE, LT)

    

# Exact solutions
# The equilibirum condition is E = T^4, let's set the manufactured solution to be E = T^3, so that the reaction terms in the PDE don't just evaluate to zero.
def manufacturedSolutions1D(t, x):
    PI, kE0, kT, kX, kXPhi = sym.symbols('PI kE0 kT kX kXPhi')
    E = ( kE0 + 
         sym.cos( kT * PI * t ) *
         sym.sin( kX * PI * x + kXPhi )
        )
    T = E**sym.Rational(1, 3)
    return E, T

def manufacturedSolutions2D(t, x, y):
    PI, kE0, kT, kX, kXPhi, kY, kYPhi = sym.symbols('PI kE0 kT kX kXPhi kY kYPhi')
    E = ( kE0 + 
         sym.cos( kT * PI * t ) *
         sym.sin( kX * PI * x + kXPhi ) * 
         sym.cos( kY * PI * y + kYPhi )
        )
    T = E**sym.Rational(1, 3)
    return E, T

def manufacturedSolutions3D(t, x, y, z):
    PI, kE0, kT, kX, kXPhi, kY, kYPhi, kZ, kZPhi = sym.symbols('PI kE0 kT kX kXPhi kY kYPhi kZ kZPhi')
    E = ( kE0 + 
         sym.cos( kT * PI * t ) *
         sym.sin( kX * PI * x + kXPhi ) * 
         sym.cos( kY * PI * y + kYPhi ) * 
         sym.cos( kZ * PI * z + kZPhi )
        )
    T = E**sym.Rational(1, 3)
    return E, T


def cxx_print(dim, x, y, z, E, T, sE, sT):
    print("-----------------")
    print("exact solution E:")
    print("-----------------")
    print( "double E = ", sym.cxxcode( sym.simplify(E) ), ";" , sep = "")
    print("")

    print("-----------------")
    print("exact solution T:")
    print("-----------------")
    print( "double T = ", sym.cxxcode( sym.simplify(T) ), ";" , sep = "")
    print("")


    print("--------------------------")
    print("gradient exact solution E:")
    print("--------------------------")
    print( "double dEdx = ", sym.cxxcode( sym.diff(E, x) ), ";" , sep = "")
    if dim >= 2:
        print("")
        print( "double dEdy = ", sym.cxxcode( sym.diff(E, y) ), ";" , sep = "")
    if dim >= 3:
        print("")
        print( "double dEdz = ", sym.cxxcode( sym.diff(E, z) ), ";" , sep = "")
        

    print("--------------------------")
    print("gradient exact solution T:")
    print("--------------------------")
    print( "double dTdx = ", sym.cxxcode( sym.diff(T, x) ), ";" , sep = "")
    print("")
    if dim >= 2:
        print( "double dTdy = ", sym.cxxcode( sym.diff(T, y) ), ";" , sep = "")
        print("")
    if dim >= 3:
        print( "double dTdz = ", sym.cxxcode( sym.diff(T, z) ), ";" , sep = "")
        print("")
    print("")

    print("--------------")
    print("source term sE:")
    print("--------------")
    print( "double sE = ", sym.cxxcode( sE ), ";", sep = "" )
    print("")

    print("--------------")
    print("source term sT:")
    print("--------------")
    print( "double sT = ", sym.cxxcode( sT ), ";", sep = "" )
    print("")


def solveForGhostFromRobinBC():
    
    # First interior point on west, east, south, and north boundaries
    Eint, Eg = sym.symbols("Eint, Eg")
    # PDE coefficients
    h, c = sym.symbols("h, c")
    # Robin values
    a, b, r  = sym.symbols("a, b, r")   

    print("-------------")
    eqn = sym.Eq(a * (Eg + Eint)/2 + b * c * (Eg - Eint)/h, r )
    sol = sym.solve(eqn, Eg)[0]
    print(f"Solution: Eg = {sol}")


# Call the main method!
if __name__ == "__main__":
    main()