import sympy as sym

"""
Some simple code to get C++ code for exact solutions and corresponding source term of Poisson problems in 1D, 2D, and 3D. 

The 3D problem is based on that in pyamg/gallery/diffusion.py but uses slightly different conventions, so as to be consistent with the Euler angles defined at https://en.wikipedia.org/wiki/Euler_angles#Definition_by_extrinsic_rotations

In particular, PyAMG uses angles phi, theta, and psi, which I believe are related to the Euler angles gamma, beta, and alpha on the wiki page (https://en.wikipedia.org/wiki/Euler_angles#Definition_by_extrinsic_rotations) as:
    phi   == -gamma,
    theta == -beta,
    psi   == -alpha

I think there is this - sign difference because the 3D roration matrices being used in PyAMG were clockwise about an axis, rather than counter clockwise, which is the default, and convention used in the wiki page.
"""


def main():

    # Just uncomment whichever one you want to use.
    poisson_1D()
    poisson_2D()
    poisson_3D()
    
    #_symbolic_rotation_helper2D()
    #_diffusionTensor_2D()

    #_symbolic_rotation_helper3D()
    #_diffusionTensor_3D()
    

def cxx_print_sol_and_rhs(u, f):
    print("-----------------")
    print("exact solution u:")
    print("-----------------")
    print( "u = ", sym.cxxcode( sym.simplify(u) ), ";" , sep = "")
    print("")
    
    print("--------------")
    print("source term f:")
    print("--------------")
    print( "f = ", sym.cxxcode( sym.simplify(f) ), ";", sep = "" )



# eqn in 1D
def poisson_1D():
    x, cxx, d_X_SHIFT = sym.symbols('x cxx d_X_SHIFT')

    # Exact solution
    u = sym.sin(2 * sym.pi * x + d_X_SHIFT) 

    # Differential operator applied to exact solution
    LU = -cxx*sym.diff( sym.diff(u, x), x) 
    
    print("\n\n1D Poisson problem:")
    cxx_print_sol_and_rhs(u, LU)


# Rotated anisotropic eqn in 2D
def poisson_2D():
    x, y, cxx, cyy, cyx, d_X_SHIFT, d_Y_SHIFT = sym.symbols('x y cxx cyy cyx d_X_SHIFT d_Y_SHIFT')

    # Exact solution
    u = ( sym.sin(2 * sym.pi * x + d_X_SHIFT) * sym.sin(4 * sym.pi * y + d_Y_SHIFT) )

    # Differential operator applied to exact solution
    LU = -cxx*sym.diff( sym.diff(u, x), x) \
         -cyy*sym.diff( sym.diff(u, y), y) \
         -cyx*sym.diff( sym.diff(u, x), y)
    
    print("\n\n2D Poisson problem:")
    cxx_print_sol_and_rhs(u, LU)

# Rotated anisotropic eqn in 3D
def poisson_3D():

    x, y, z, cxx, cyy, czz, cyx, czx, czy, d_X_SHIFT, d_Y_SHIFT, d_Z_SHIFT = sym.symbols('x y z cxx cyy czz cyx czx czy d_X_SHIFT d_Y_SHIFT d_Z_SHIFT')

    # Exact solution
    u = ( sym.sin(2 * sym.pi * x + d_X_SHIFT) * sym.sin(4 * sym.pi * y + d_Y_SHIFT) * sym.sin(6 * sym.pi * z + d_Z_SHIFT) )

    # Differential operator applied to exact solution
    LU = -cxx*sym.diff( sym.diff(u, x), x) \
         -cyy*sym.diff( sym.diff(u, y), y) \
         -czz*sym.diff( sym.diff(u, z), z) \
         -cyx*sym.diff( sym.diff(u, x), y) \
         -czx*sym.diff( sym.diff(u, x), z) \
         -czy*sym.diff( sym.diff(u, y), z) \
    
    print("\n\n3D Poisson problem:")
    cxx_print_sol_and_rhs(u, LU)




# 2D equation with anisotropy eps y. The PDE is rotated through an angle of theta radians counter clockwise from the positive x-axis 
# These 4 coefficients come from the function _symbolic_rotation_helper2D
# Here we just convert them from symbolic python into cxx interpratable code
# Note that the actually diffusion tensor has coefficients d_{ij}, 1<=i,j<=2, but that we only need the 3 coefficients that define the upper triangular component since it's symmetric.
def _diffusionTensor_2D():

    cth, sth, eps = sym.symbols('cth, sth, eps')
    
    d11 = cth**2 + eps*sth**2
    d12 = -cth*eps*sth + cth*sth
    d21 = -cth*eps*sth + cth*sth
    d22 = cth**2*eps + sth**2

    # Just print the upper trinagular components
    print( "double d11 = ", sym.cxxcode( sym.simplify(d11) ), ";", sep = "" )
    print( "double d22 = ", sym.cxxcode( sym.simplify(d22) ), ";", sep = "" )
    print( "double d12 = ", sym.cxxcode( sym.simplify(d12) ), ";", sep = "" )
    

# 3D equation with anisotropy epsy and epsz in the y and z directions, respectively. The angles here gamma, beta, and alpha are the so-called extrinsic Euler angles used to define rotation in 3D. 
# These 9 coefficients come from the function _symbolic_rotation_helper3D
# Here we just convert them from symbolic python into cxx interpratable code
# Note that the actually diffusion tensor has coefficients d_{ij}, 1<=i,j<=3, but that we only need the 6 coefficients that define the upper triangular component since it's symmetric.
def _diffusionTensor_3D():

    cg, sg, cb, sb, ca, sa, epsy, epsz = sym.symbols('cg, sg, cb, sb, ca, sa, epsy, epsz')
    
    d11 = epsy*(-ca*sg - cb*cg*sa)**2 + epsz*sa**2*sb**2 + (ca*cg - cb*sa*sg)**2
    d12 = -ca*epsz*sa*sb**2 + epsy*(-ca*sg - cb*cg*sa)*(ca*cb*cg - sa*sg) + (ca*cg - cb*sa*sg)*(ca*cb*sg + cg*sa)
    d13 = cb*epsz*sa*sb + cg*epsy*sb*(-ca*sg - cb*cg*sa) + sb*sg*(ca*cg - cb*sa*sg)
    d21 = -ca*epsz*sa*sb**2 + epsy*(-ca*sg - cb*cg*sa)*(ca*cb*cg - sa*sg) + (ca*cg - cb*sa*sg)*(ca*cb*sg + cg*sa)
    d22 = ca**2*epsz*sb**2 + epsy*(ca*cb*cg - sa*sg)**2 + (ca*cb*sg + cg*sa)**2
    d23 = -ca*cb*epsz*sb + cg*epsy*sb*(ca*cb*cg - sa*sg) + sb*sg*(ca*cb*sg + cg*sa)
    d31 = cb*epsz*sa*sb + cg*epsy*sb*(-ca*sg - cb*cg*sa) + sb*sg*(ca*cg - cb*sa*sg)
    d32 = -ca*cb*epsz*sb + cg*epsy*sb*(ca*cb*cg - sa*sg) + sb*sg*(ca*cb*sg + cg*sa)
    d33 = cb**2*epsz + cg**2*epsy*sb**2 + sb**2*sg**2

    # Just print the upper trinagular components
    print( "double d11 = ", sym.cxxcode( sym.simplify(d11) ), ";", sep = "" )
    print( "double d22 = ", sym.cxxcode( sym.simplify(d22) ), ";", sep = "" )
    print( "double d33 = ", sym.cxxcode( sym.simplify(d33) ), ";", sep = "" )
    print( "double d12 = ", sym.cxxcode( sym.simplify(d12) ), ";", sep = "" )
    print( "double d13 = ", sym.cxxcode( sym.simplify(d13) ), ";", sep = "" )
    print( "double d23 = ", sym.cxxcode( sym.simplify(d23) ), ";", sep = "" )



def _symbolic_rotation_helper2D():
    """Use SymPy to generate the diffusion tensor for the 2D problem."""
    import sympy
    from sympy import symbols, Matrix  


    cth, sth = symbols('cth, sth')    
    # Counter-clockwise rotation of theta radians in the xy-plane
    Rg = Matrix([[cth, -sth], [sth, cth]]) 
    
    Q = Rg
    print("Diffusion 2D, Underlying rotation matrix:")
    for i in range(2):
        for j in range(2):
            print(f'Q{i+1}{j+1} = {Q[i, j]}')
    print("")

    eps = symbols('eps')
    A = Matrix([[1, 0], [0, eps]])

    D = Q * A * Q.T
    print("Diffusion 2D, Diffusion tensor Q * A * Q^T:")
    for i in range(2):
        for j in range(2):
            print(f'd{i+1}{j+1} = {D[i, j]}')
    print("")


def _symbolic_rotation_helper3D():
    """Use SymPy to generate the diffusion tensor for the 3D problem."""
    import sympy
    from sympy import symbols, Matrix  

    # From wiki's "Definition by extrinsice rotations," we rotate the XYZ system, which is initially aligned with the xyz system by the following composition 
    # 1. gamma about the z-axis
    # 2. beta  about the x-axis
    # 3. alpha about the z-axis 
    # This sequence is denoted as 3-1-3 or z-x-z

    ca, sa = symbols('ca, sa')
    cb, sb = symbols('cb, sb')
    cg, sg = symbols('cg, sg')

    # Counter-clockwise rotation of gamma radians in the xy-plane, i.e., the z-axis
    Rg = Matrix([[ca, -sa, 0], [sa, ca,   0], [0,  0,   1]]) 
    # Counter-clockwise rotation of beta radians in the yz-plane, i.e., the x-axis
    Rb = Matrix([[1,    0, 0], [0,  cb, -sb], [0,  sb, cb]]) 
    # Counter-clockwise rotation of alpha radians in the xy-plane, i.e., the z-axis
    Ra = Matrix([[cg, -sg, 0], [sg, cg,   0], [0,  0,   1]]) 

    # This rotation matrix is the same as the Z_alpha * X_beta * Z_gamma one in the bottom row of the "Proper Euler Angles" 
    Q = Rg * Rb * Ra
    print("Diffusion 3D, Underlying rotation matrix:")
    for i in range(3):
        for j in range(3):
            print(f'Q{i+1}{j+1} = {Q[i, j]}')
    print("")

    epsy, epsz = symbols('epsy, epsz')
    A = Matrix([[1, 0, 0], [0, epsy, 0], [0, 0, epsz]])

    D = Q * A * Q.T
    print("Diffusion 3D, Diffusion tensor Q * A * Q^T:")
    for i in range(3):
        for j in range(3):
            print(f'd{i+1}{j+1} = {D[i, j]}')
    print("")


# Call the main method!
if __name__ == "__main__":
    main()