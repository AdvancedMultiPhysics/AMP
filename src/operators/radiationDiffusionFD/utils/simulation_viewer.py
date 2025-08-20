import sympy as sym
import numpy as np
import os
import re

# Plotting stuff
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator
#matplotlib.use('QT5Agg') # Or 'QT4Agg'

params = {'legend.fontsize': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18}
plt.rcParams.update(params)
#plt.rcParams['figure.figsize'] = [5, 2.5] # for square canvas

import copy

def main():

    mousseau_compare_1D()

    #main1D()
    #main2D()

def main2D():
    #qoi = "E"
    qoi = "T"

    show_man = False # Show manufactured solution

    Eexp = 0.25 # "Radiation energy"
    #Eexp = 1.0  # Regular energy

    freeze_axes = not True # Force vertical axis to be time independent

    n = 64 # n DOFs for each variable
    n = 128
    #n = 256
    
    h = 1/n
    x = np.linspace( h/2, 1-h/2, n )
    y = np.linspace( h/2, 1-h/2, n )
    X, Y = np.meshgrid(x, y)

    ax_vert_lim  = []
    ax3_vert_lim = []
    
    N = n**2 # Total # of DOFs
    data_dir = "../build/out/n" + str(n) + "_2D/"

    # Get the number of time steps
    nTimeSteps = get_max_steps( lambda step : data_dir + "ETnum_" + str(step) + ".ascii" )
    print( "Number of time steps=", nTimeSteps )
    plot_time_step_data( data_dir, nTimeSteps )

    stepPlotSkip = int(0.1*nTimeSteps)
    steps = list(range(0, nTimeSteps, stepPlotSkip))
    steps.append(nTimeSteps)
    for step in steps:

        ETnum_file = data_dir + "ETnum_" + str(step) + ".ascii"        
        with open(ETnum_file) as f:
            header = f.readline()
        time = re.findall(r'"(.*?)"', header)[0]

        ETnum_data = np.genfromtxt(ETnum_file, skip_header=1)
        Enum_data = (ETnum_data[0:N].reshape(n, n)) ** Eexp
        Tnum_data =  ETnum_data[N::].reshape(n, n)

        if qoi == "E":
            qoi_data = Enum_data
        elif qoi == "T":
            qoi_data = Tnum_data

        # Numerical solution: Surface plot
        fig, ax  = plt.subplots(subplot_kw={"projection": "3d"})
        fig.canvas.manager.window.move(1000, 200)
        col_map = copy.copy(cm.coolwarm)
        surf = ax.plot_surface(X, Y, qoi_data, cmap=col_map)
        ax.view_init(azim = 225)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid()
        ax.set_title( "numerical: time = {}, step = {}".format( time, step ) )
        # Force axis to be time-independent
        if freeze_axes:
            if step == 0:
                ax_vert_lim = ax.get_zlim()
            else:
                ax.set_zlim( ax_vert_lim )

        # Numerical solution: Contour plot
        # --------------------------------
        fig2, ax2 = plt.subplots()
        fig2.canvas.manager.window.move(200, 200)
        col_map = copy.copy(cm.coolwarm)
        levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        con1 = ax2.contourf(X, Y, qoi_data, levels = levels, cmap=col_map) 
        con2 = ax2.contour(X, Y, qoi_data, levels = con1.levels, colors = "k")
        ax2.clabel(con2, inline=True, fontsize=10) # Add labels to contour levels
        cbar = fig2.colorbar(con1)
        cbar.add_lines(con2)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        ax2.grid()
        ax2.set_title( "numerical: time = {}, step = {}".format( time, step ) )
        plt.tight_layout()
        

        # Manufactured
        if show_man:
            ETman_file = data_dir + "ETman_" + str(step) + ".ascii"
            ETman_data = np.genfromtxt(ETman_file, stepPlotSkip_header=1)
            Eman_data = (ETman_data[0:N].reshape(n, n)) **Eexp
            Tman_data =  ETman_data[N::].reshape(n, n)
            fig3, ax3  = plt.subplots(subplot_kw={"projection": "3d"})
            fig3.canvas.manager.window.move(200, 200)
            col_map = copy.copy(cm.coolwarm)
            surf = ax3.plot_surface(X, Y, Eman_data - Enum_data, cmap=col_map)
            ax3.set_title( "error: time = {}, step = {}".format( time, step ) )
            #surf = ax2.plot_surface(X, Y, Eman_data, cmap=col_map)
            #ax2.set_title( "manufactured: time = {}, step = {}".format( time, step ) )
            ax3.view_init(azim = 225)
            ax3.set_xlabel("$x$")
            ax3.set_ylabel("$y$")
            ax3.grid()
            plt.tight_layout()

            # Force axis to be time-independent
            if freeze_axes:
                if step == 0:
                    ax3_vert_lim = ax3.get_zlim()
                else:
                    ax3.set_zlim( ax3_vert_lim )
        
        plt.show()




def main1D():

    show_man = False # Show manufactured solution

    Eexp = 0.25 # "Radiation energy"
    #Eexp = 1.0  # Regular energy

    freeze_axes = not True # Force vertical axis to be time independent
    
    #n = 64 # n DOFs for each variable
    #n = 128
    n = 256
    
    h = 1/n
    x = np.linspace( h/2, 1-h/2, n )

    ax_vert_lim = []
    #ax_vert_lim = [0, 1.5]

    data_dir = "../build/out/n" + str(n) + "_1D" + "/"
    
    # Get the number of time steps
    nTimeSteps = get_max_steps( lambda step : data_dir + "ETnum_" + str(step) + ".ascii" )
    plot_time_step_data( data_dir, nTimeSteps )

    stepPlotSkip = int(0.1*nTimeSteps)
    steps = list(range(0, nTimeSteps, stepPlotSkip))
    steps.append(nTimeSteps)
    for step in steps:

        ETnum_file = data_dir + "ETnum_" + str(step) + ".ascii"        
        with open(ETnum_file) as f:
            header = f.readline()
            time = re.findall(r'"(.*?)"', header)[0]
        
        ETnum_data = np.genfromtxt(ETnum_file, skip_header=1)
        Enum_data = ETnum_data[0:n] ** Eexp
        Tnum_data = ETnum_data[n::]

        fig, ax = plt.subplots()

    # Example: Move the window to a specific position (e.g., 100 pixels from top, 200 pixels from left)
        fig.canvas.manager.window.move(200, 200)

        # Overlay manufactured solution
        if show_man:
            ETman_file = data_dir + "ETman_" + str(step) + ".ascii"
            ETman_data = np.genfromtxt(ETman_file, stepPlotSkip_header=1)
            Eman_data = ETman_data[0:n] **Eexp
            Tman_data = ETman_data[n::]
            ax.plot(x, Eman_data, 'ro', markersize = 7, markerfacecolor = "white", label = "$E_{{man}}$")
            ax.plot(x, Tman_data, 'bo', markersize = 7, markerfacecolor = "white", label = "$T_{{man}}$")

        
        ax.plot(x, Enum_data, '-r',  markersize = 7, label = "$E_{{num}}^{{{}}}$".format(Eexp))
        ax.plot(x, Tnum_data, '--b', markersize = 7, label = "$T_{{num}}$")
        ax.set_xlabel("$x$")
        ax.legend()
        ax.grid()
        ax.set_title( "time={}, step={}/{}".format( time, step, nTimeSteps ) )
        plt.tight_layout()

        # Force axis to be time independent
        if freeze_axes:
            if step == 0:
                ax_vert_lim = ax.get_ylim()
            else:
                ax.set_ylim( ax_vert_lim )
        #ax.set_ylim( ax_vert_lim )

        plt.show()
 



# Get number of steps taken by finding the file with the maximum "step". This is dumb, but it'll do. 
# Where step_dependent_file_path is a handle taking in "step"
def get_max_steps( step_dependent_file_path ):
    current_step = -1
    while True:
        if os.path.exists( step_dependent_file_path( current_step+1 ) ):
            current_step += 1
        else:
            break
    if current_step == -1:
        raise ValueError( "The file '{}' does not even exist".format(step_dependent_file_path(0)) )
    return current_step


# Plot dt as a function of the step number
def plot_time_step_data( data_dir, max_steps ):

    times = []
    steps = range( max_steps )

    # Strip out time at every step
    for step in steps:
        ETnum_file = data_dir + "ETnum_" + str(step) + ".ascii"        
        with open(ETnum_file) as f:
            header = f.readline()
            times.append(re.findall(r'"(.*?)"', header)[0])

    steps = np.array( steps )
    times = np.array( times, dtype=float )
    dts   = times[1:] - times[:-1]

    fig, ax = plt.subplots()
    ax.semilogy( steps[:-1], dts, '-bo' )
    ax.set_xlabel( "step" )
    ax.set_title( "$\\delta t$" )


# Plot some data I have saved in a tmp directory
def mousseau_compare_1D():
    Eexp = 0.25
    n = 256
    h = 1/n
    x = np.linspace( h/2, 1-h/2, n )

    kvals = ["1e-5", "1e-1"]
    lw    = [1, 2.5]
    cols  = ['k', 'r', 'g', 'm']

    fig, ax = plt.subplots()
    for count, kval in enumerate(kvals):
        filename = "../build/out/tmp/k{}.ascii".format( kval )
        with open(filename) as f:
            header = f.readline()
            time = re.findall(r'"(.*?)"', header)[0]
    
        ETnum_data = np.genfromtxt(filename, skip_header=1)
        Enum_data = ETnum_data[0:n] ** Eexp
        Tnum_data = ETnum_data[n::]

        ax.plot(x, Enum_data, '-', color=cols[2*count], linewidth=lw[count], label = "$k={{{}}}: E_{{num}}^{{{}}}$".format(kval, Eexp))
        ax.plot(x, Tnum_data, '--', color=cols[2*count+1], linewidth=lw[count], label = "$k={{{}}}: T_{{num}}$".format(kval))

        if kval == "1e-5":
            contour_levels = np.linspace(0, 1.2, 12)
            for cl in contour_levels:
                print( cl )
                ci = np.argmin(np.abs(Tnum_data-cl))
                ax.plot(x[ci], Tnum_data[ci], "*", markersize = 10, color=cols[2*count+1])

    ax.set_xlabel("$x$")
    ax.legend()
    ax.grid( "minor", "major" )
    ax.set_title( "time={}".format( time ) )
    ax.set_ylim( [0, 1.5] )
    ax.set_xlim( [0, 1] )
    # Set major tick locators
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    plt.tight_layout()
    plt.show()


# Call the main method!
if __name__ == "__main__":
    main()