"""
This python script is used for generating maunfactured solutions to a class of radiation diffusion problems and translating them into cxx interpretable code. 

It also symbolically solves for ghost values from certain discretized boundary conditions
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

    

# Exact solutions
# The equilibirum condition is E = T^4, let's set the manufactured solution to be E = T^3, so that the reaction terms in the PDE don't just evaluate to zero.
def manufacturedSolutions1D(x, PI, t):
    kE0, kT, kX, kXPhi = sym.symbols('kE0 kT kX kXPhi')
    E = ( kE0 + 
         sym.sin( kX * PI * x + kXPhi ) *
         sym.cos( kT * PI * t ) 
        )
    T = E**sym.Rational(1, 3)
    return E, T

def manufacturedSolutions2D(x, y, PI, t):
    kE0, kT, kX, kXPhi, kY, kYPhi = sym.symbols('kE0 kT kX kXPhi kY kYPhi')
    E = ( kE0 + 
         sym.sin( kX * PI * x + kXPhi ) * 
         sym.cos( kY * PI * y + kYPhi ) * 
         sym.cos( kT * PI * t ) 
        )
    T = E**sym.Rational(1, 3)
    return E, T

def manufacturedSolutions3D(x, y, z, PI, t):
    kE0, kT, kX, kXPhi, kY, kYPhi, kZ, kZPhi = sym.symbols('kE0 kT kX kXPhi kY kYPhi kZ kZPhi')
    E = ( kE0 + 
         sym.sin( kX * PI * x + kXPhi ) * 
         sym.cos( kY * PI * y + kYPhi ) * 
         sym.cos( kZ * PI * z + kZPhi ) * 
         sym.cos( kT * PI * t ) 
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





# A pair of linear PDEs of the form
# dE/dt - nabla dot ( k11 * nabla E ) - k12 * simga * ( T - E ) = s_E
# dT/dt - nabla dot ( k21 * nabla T ) + k22 * simga * ( T - E ) = s_T
#
#
# A pair of nonlinear PDEs of the form
# dE/dt - nabla dot ( k11 * DE * nabla E ) - k12 * sigma * ( T^4 - E ) = s_E
# dT/dt - nabla dot ( k21 * DT * nabla T ) + k22 * sigma * ( T^4 - E ) = s_T
# for constants k_ij. 
#
# Note we assume zatom is constant, while in the paper it can vary.
def manufacturedModel( dim, model ):
    
    print("-----------------------------")
    print("    {} model {}D    ".format(model, dim))
    print("-----------------------------", sep="\n")

    x, y, z, PI, zatom, k11, k12, k21, k22, t = sym.symbols('x y z PI zatom k11 k12 k21 k22 t')

    # Exact solutions
    if dim == 1:
        E, T = manufacturedSolutions1D(x, PI, t) 
    elif dim == 2:
        E, T = manufacturedSolutions2D(x, y, PI, t) 
    elif dim == 3:
        E, T = manufacturedSolutions3D(x, y, z, PI, t) 

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






    # # Create lambda functions for sympy expressions
    # Epy = sym.lambdify((x, y, PI), E, modules=['numpy'])
    # Tpy = sym.lambdify((x, y, PI), T, modules=['numpy'])

    # LEpy = sym.lambdify((x, y, PI, zatom, k11, k12), LE, modules=['numpy'])
    # LTpy = sym.lambdify((x, y, PI, zatom, k21, k22), LT, modules=['numpy'])


    # xnum = np.linspace(0, 1)
    # ynum = np.linspace(0, 1)
    # Xnum, Ynum = np.meshgrid(xnum, ynum)

    # znum = 5 # If this is 1 then it really amplifies S_E
    # # k11num = 1
    # # k12num = 1
    # # k21num = 0.01
    # # k22num = 1

    # k11num = 1
    # k12num = 1
    # k21num = 0.01  
    # k22num = 1

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(Xnum, Ynum, Epy(Xnum, Ynum, np.pi).T, cmap=cm.coolwarm)
    # ax.set_title("$E(x, y)$: $(k_{{11}},k_{{12}},k_{{21}},k_{{22}})=({},{},{},{})$".format(k11num, k12num, k21num, k22num))
    # ax.set_xlabel( "$x$" )
    # ax.set_ylabel( "$y$" )

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(Xnum, Ynum, Tpy(Xnum, Ynum, np.pi).T, cmap=cm.coolwarm)
    # ax.set_title("$T(x, y)$: $(k_{{11}},k_{{12}},k_{{21}},k_{{22}})=({},{},{},{})$".format(k11num, k12num, k21num, k22num))
    # ax.set_xlabel( "$x$" )
    # ax.set_ylabel( "$y$" )
    
    

    # # ax.plot( xnum, LEpy(xnum, np.pi, znum, k11num, k12num), '--g', label = "$S_E(x)$" ) 
    # # ax.plot( xnum, LTpy(xnum, np.pi, znum, k21num, k22num), 'm', label = "$S_T(x)$" ) 
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(Xnum, Ynum, LEpy(Xnum, Ynum, np.pi, znum, k11num, k12num).T, cmap=cm.coolwarm)
    # ax.set_title("$s_E(x, y)$: $(k_{{11}},k_{{12}},k_{{21}},k_{{22}})=({},{},{},{})$".format(k11num, k12num, k21num, k22num))
    # ax.set_xlabel( "$x$" )
    # ax.set_ylabel( "$y$" )

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(Xnum, Ynum, LTpy(Xnum, Ynum, np.pi, znum, k21num, k22num).T, cmap=cm.coolwarm)
    # ax.set_title("$s_T(x, y)$: $(k_{{11}},k_{{12}},k_{{21}},k_{{22}})=({},{},{},{})$".format(k11num, k12num, k21num, k22num))
    # ax.set_xlabel( "$x$" )
    # ax.set_ylabel( "$y$" )

    # plt.show()


# Call the main method!
if __name__ == "__main__":
    main()