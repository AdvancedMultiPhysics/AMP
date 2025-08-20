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


# For radiation diffusion
def main():

    #manufacturedLinearModel1D()
    #manufacturedNonlinearModel1D()
    
    #manufacturedNonlinearModel2D()
    manufacturedLinearModel2D()

    #solveForGhostsFromRobinBC2D()


def solveForGhostsFromRobinBC2D():
    """The continuous BCs are:
        a1 * u - b1 * c * du/dx = r1 @ x = 0
        a2 * u + b2 * c * du/dx = r2 @ x = 1
        a3 * u - b3 * c * du/dy = r3 @ y = 0
        a4 * u + b4 * c * du/dy = r4 @ y = 1

    Otherwise, this is the same as the 1D case
    """

    print("2D boundaries")
    print("-------------")
    
    # First interior point on west, east, south, and north boundaries
    E1j,   E2j,   Ei3,   Ei4   = sym.symbols("E1j,   E2j,   Ei3,   Ei4")
    # Ghost point on west, east, south, and north boundaries
    Eg_1j, Eg_2j, Eg_i3, Eg_i4 = sym.symbols("Eg_1j, Eg_2j, Eg_i3, Eg_i4")
    # PDE coefficients
    h, c = sym.symbols("h, c")
    # Robin values
    a1, b1, r1  = sym.symbols("d_a1, d_b1, r1")
    a2, b2, r2  = sym.symbols("d_a2, d_b2, r2")
    a3, b3, r3  = sym.symbols("d_a3, d_b3, r3")
    a4, b4, r4  = sym.symbols("d_a4, d_b4, r4")   

    # 1
    print("\n-------------")
    print("Boundary 1")
    print("-------------")
    eqn1 = sym.Eq(a1 * (Eg_1j + E1j)/2 - b1 * c * (E1j - Eg_1j)/h, r1 )
    sol1 = sym.solve(eqn1, Eg_1j)[0]
    print(f"Solution: Eg = {sol1}")

    # 2
    print("\n-------------")
    print("Boundary 2")
    print("-------------")
    eqn2 = sym.Eq(a2 * (E2j + Eg_2j)/2 + b2 * c * (Eg_2j - E2j)/h, r2 )
    sol2 = sym.solve(eqn2, Eg_2j)[0]
    print(f"Solution: Eg = {sol2}")

    # 3
    print("\n-------------")
    print("Boundary 3")
    print("-------------")
    eqn3 = sym.Eq(a3 * (Eg_i3 + Ei3)/2 - b3 * c * (Ei3 - Eg_i3)/h, r3 )
    sol3 = sym.solve(eqn3, Eg_i3)[0]
    print(f"Solution: Eg = {sol3}")

    # 4
    print("\n-------------")
    print("Boundary 4")
    print("-------------")
    eqn4 = sym.Eq(a4 * (Ei4 + Eg_i4)/2 + b4 * c * (Eg_i4 - Ei4)/h, r4 )
    sol4 = sym.solve(eqn4, Eg_i4)[0]
    print(f"Solution: Eg = {sol4}")
    

# Exact solutions
def manufacturedSolutions1D(x, M_PI, t):
    E = ( 2 + sym.sin(1.5 * M_PI * x) * sym.cos( 2 * M_PI * t ) )
    T = E**(1/4)
    return E, T

def manufacturedSolutions2D(x, y, M_PI, t):
    E = ( 2 + sym.sin(1.5 * M_PI * x) * sym.cos( 3.5 * M_PI * y ) * sym.cos( 2 * M_PI * t ) )
    T = E**(1/4)
    return E, T


def cxx_print(E, T, sE, sT):
    print("-----------------")
    print("exact solution E:")
    print("-----------------")
    print( "E = ", sym.cxxcode( sym.simplify(E) ), ";" , sep = "")
    print("")

    print("-----------------")
    print("exact solution T:")
    print("-----------------")
    print( "T = ", sym.cxxcode( sym.simplify(T) ), ";" , sep = "")
    print("")
    
    print("--------------")
    print("source term sE:")
    print("--------------")
    print( "sE = ", sym.cxxcode( sym.simplify(sE) ), ";", sep = "" )
    print("")

    print("--------------")
    print("source term sT:")
    print("--------------")
    print( "sT = ", sym.cxxcode( sym.simplify(sT) ), ";", sep = "" )
    print("")



# A pair of linear PDEs of the form
# dE/dt - d/dx ( k11 * dE/dx ) - k12 * ( T - E ) = s_E
# dT/dt - d/dx ( k21 * dT/dx ) + k22 * ( T - E ) = s_T
# for constants k_ij
#
def manufacturedLinearModel1D():
    print("--------------------------------------")
    print("    Linear model 1D time dependent    ")
    print("--------------------------------------", sep="\n")

    x, M_PI, k11, k12, k21, k22, t = sym.symbols('x M_PI k11 k12 k21 k22 t')

    # Exact solutions
    E, T = manufacturedSolutions1D(x, M_PI, t)

    # Differential operator applied to exact solution
    LE = -sym.diff( k11 * sym.diff( E, x), x) - k12 * (T - E) 
    LT = -sym.diff( k21 * sym.diff( T, x), x) + k22 * (T - E) 

    # Add time derivative of solutions
    LE += sym.diff( E, t )
    LT += sym.diff( T, t )

    # Translate into C++ code
    cxx_print(E, T, LE, LT)

    print("--------------------------")
    print("gradient exact solution E:")
    print("--------------------------")
    print( "dEdx = ", sym.cxxcode( sym.simplify(sym.diff(E, x)) ), ";" , sep = "")
    print("")

    print("--------------------------")
    print("gradient exact solution T:")
    print("--------------------------")
    print( "dTdx = ", sym.cxxcode( sym.simplify(sym.diff(T, x)) ), ";" , sep = "")
    print("")



# A pair of nonlinear PDEs of the form
# dE/dt - d/dx ( k11 * DE * dE/dx ) - k12 * simga * ( T^4 - E ) = s_E
# dE/dt - d/dx ( k21 * DT * dT/dx ) + k22 * simga * ( T^4 - E ) = s_T
# for constants k_ij. 
# PDEs in the paper correspond to:
#   k11 = k12 = k22 = 1, k21 = k.
# Also, the k in the paper corresponds to k21, the diffusion of T
# Note we assume z is constant, while in the paper it can vary.
def manufacturedNonlinearModel1D():
    print("-----------------------------------------")
    print("    Nonlinear model 1D time-dependent    ")
    print("-----------------------------------------", sep="\n")

    x, M_PI, z, k11, k12, k21, k22, t = sym.symbols('x M_PI z k11 k12 k21 k22 t')

    # Exact solutions
    E, T = manufacturedSolutions1D(x, M_PI, t)

    sigma = (z/T)**3.0
    DE = 1/(3.0*sigma)
    DT = T**(2.5)

    R = sigma*( T**4 - E )

    # Spatial differential operator applied to exact solution
    LE = -k11 * sym.diff( DE * sym.diff(E, x), x ) - k12 * R
    LT = -k21 * sym.diff( DT * sym.diff(T, x), x ) + k22 * R

    # Add time derivative of solutions
    LE += sym.diff( E, t )
    LT += sym.diff( T, t )

    # Translate into C++ code
    cxx_print(E, T, LE, LT)

    print("--------------------------")
    print("gradient exact solution E:")
    print("--------------------------")
    print( "dEdx = ", sym.cxxcode( sym.simplify(sym.diff(E, x)) ), ";" , sep = "")
    print("")

    print("--------------------------")
    print("gradient exact solution T:")
    print("--------------------------")
    print( "dTdx = ", sym.cxxcode( sym.simplify(sym.diff(T, x)) ), ";" , sep = "")
    print("")

# A pair of linear PDEs of the form
# dE/dt - nabla dot ( k11 * nabla E ) - k12 * simga * ( T - E ) = s_E
# dT/dt - nabla dot ( k21 * nabla T ) + k22 * simga * ( T - E ) = s_T
# 
def manufacturedLinearModel2D():
    
    print("-----------------------------------------")
    print("      Linear model 2D time-dependent     ")
    print("-----------------------------------------", sep="\n")

    x, y, M_PI, z, k11, k12, k21, k22, t = sym.symbols('x y M_PI z k11 k12 k21 k22 t')

    # Exact solutions
    E, T = manufacturedSolutions2D(x, y, M_PI, t)    

    R = ( T - E )

    # Spatial differential operator applied to exact solution
    LE = -k11 * ( sym.diff( sym.diff(E, x), x ) + sym.diff( sym.diff(E, y), y ) ) - k12 * R
    LT = -k21 * ( sym.diff( sym.diff(T, x), x ) + sym.diff( sym.diff(T, y), y ) ) + k22 * R

    # Add time derivative of solutions
    LE += sym.diff( E, t )
    LT += sym.diff( T, t )

    # Translate into C++ code
    cxx_print(E, T, LE, LT)

    # Translate into C++ code

    print("--------------------------")
    print("gradient exact solution E:")
    print("--------------------------")
    print( "dEdx = ", sym.cxxcode( sym.simplify(sym.diff(E, x)) ), ";" , sep = "")
    print("")
    print( "dEdy = ", sym.cxxcode( sym.simplify(sym.diff(E, y)) ), ";" , sep = "")
    print("")

    print("--------------------------")
    print("gradient exact solution T:")
    print("--------------------------")
    print( "dTdx = ", sym.cxxcode( sym.simplify(sym.diff(T, x)) ), ";" , sep = "")
    print("")
    print( "dTdy = ", sym.cxxcode( sym.simplify(sym.diff(T, y)) ), ";" , sep = "")
    print("")



# A pair of nonlinear PDEs of the form
# dE/dt - nabla dot ( k11 * DE * nabla E ) - k12 * simga * ( T^4 - E ) = s_E
# dT/dt - nabla dot ( k21 * DT * nabla T ) + k22 * simga * ( T^4 - E ) = s_T
# for constants k_ij. 
# PDEs in the paper correspond to:
#   k11 = k12 = k22 = 1, k21 = k.
# Also, the k in the paper corresponds to k21, the diffusion of T
# Note we assume z is constant, while in the paper it can vary.
def manufacturedNonlinearModel2D():
    
    print("-----------------------------------------")
    print("    Nonlinear model 2D time-dependent    ")
    print("-----------------------------------------", sep="\n")

    x, y, M_PI, z, k11, k12, k21, k22, t = sym.symbols('x y M_PI z k11 k12 k21 k22 t')

    # Exact solutions
    E, T = manufacturedSolutions2D(x, y, M_PI, t)    

    sigma = (z/T)**3.0
    DE = 1/(3.0*sigma)
    DT = T**(2.5)

    R = sigma*( T**4 - E )

    # Spatial differential operator applied to exact solution
    LE = -k11 * ( sym.diff( DE * sym.diff(E, x), x ) + sym.diff( DE * sym.diff(E, y), y ) ) - k12 * R
    LT = -k21 * ( sym.diff( DT * sym.diff(T, x), x ) + sym.diff( DT * sym.diff(T, y), y ) ) + k22 * R

    # Add time derivative of solutions
    LE += sym.diff( E, t )
    LT += sym.diff( T, t )

    # Translate into C++ code
    cxx_print(E, T, LE, LT)

    # Translate into C++ code

    print("--------------------------")
    print("gradient exact solution E:")
    print("--------------------------")
    print( "dEdx = ", sym.cxxcode( sym.simplify(sym.diff(E, x)) ), ";" , sep = "")
    print("")
    print( "dEdy = ", sym.cxxcode( sym.simplify(sym.diff(E, y)) ), ";" , sep = "")
    print("")

    print("--------------------------")
    print("gradient exact solution T:")
    print("--------------------------")
    print( "dTdx = ", sym.cxxcode( sym.simplify(sym.diff(T, x)) ), ";" , sep = "")
    print("")
    print( "dTdy = ", sym.cxxcode( sym.simplify(sym.diff(T, y)) ), ";" , sep = "")
    print("")


    # # Create lambda functions for sympy expressions
    # Epy = sym.lambdify((x, y, M_PI), E, modules=['numpy'])
    # Tpy = sym.lambdify((x, y, M_PI), T, modules=['numpy'])

    # LEpy = sym.lambdify((x, y, M_PI, z, k11, k12), LE, modules=['numpy'])
    # LTpy = sym.lambdify((x, y, M_PI, z, k21, k22), LT, modules=['numpy'])


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