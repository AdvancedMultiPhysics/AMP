import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18}
plt.rcParams.update(params)

filename_pref = "out/disc_err_test_2D"
filename_pref = "out/disc_err_test_2D_adaptive_dt"


# Define the range of n values
n_values = [16, 32, 64, 128] #, 256]

# Create a figure and axis
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

# Define line styles and markers
line_styles = ['--', ':', '--', ':', '--']
markers     = ['x', 'o', '>', 'd', '*']
colors      = ['r', 'b', 'g', 'm', 'k']

h_values = []
errmaxT  = []

# Loop over each n value
for i, n in enumerate(n_values):
    filename = filename_pref + f"/n{n}.csv"
    
    # Read first two rows
    df_header = pd.read_csv(filename, nrows=1)
    h_values.append( df_header["h"] )

    print(df_header)

    if (df_header["n"].iloc[0] != n):
        raise ValueError("Something is wrong {}!={}".format(df_header["n"].iloc[0], n))

    # Read all remaning rows
    df = pd.read_csv(filename, header=2)

    # Extract time, nonlinear_iters, and linear_iters
    dt              = df['dt']
    time            = df['time']
    nonlinear_iters = df['nonlinear_iters']
    linear_iters    = df['linear_iters']
    errmax          = df['errmax']
    
    
    # Plot nonlinear_iters with dashed line and crosses
    ax1.plot(time, nonlinear_iters, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=f'$n={n}$')
    
    # Plot linear_iters with dotted line and circles
    ax2.plot(time, linear_iters, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=f'$n={n}$')

    # Plot linear_iters with dotted line and circles
    ax3.semilogy(time, errmax, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=f'$n={n}$')

    # Plot linear_iters with dotted line and circles
    ax5.semilogy(np.arange(0, time.shape[0]), dt, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=f'$n={n}$')

    errmaxT.append(errmax.iloc[-1])

# Plot max error at final time
h_values = np.array( h_values )
ax4.loglog(h_values, errmaxT, "-ko", label = "error")
c = 0.7*errmaxT[-1]/(h_values[-1]**2)
ax4.loglog(h_values, c*h_values**2, "--b>", label = "$c h^2$")
# Set the x-axis scale to base 2
ax4.set_xscale('log', base=2)


# Set labels
ax1.set_xlabel('$t$')
ax2.set_xlabel('$t$')
ax3.set_xlabel('$t$')
ax4.set_xlabel('$h$')
ax5.set_xlabel('time step')

#ax1.set_ylabel('Nonlinear iterations')

#ax2.set_ylabel('Linear iterations')

#ax2.set_ylabel('Max discretization error')

# Set title
ax1.set_title('Nonlinear iterations vs. time')
ax2.set_title('Linear iterations vs. time')
ax3.set_title('Max disc. error vs. time')
ax4.set_title('Max disc. error at final time vs. $h$')
ax5.set_title('$\\delta t$')

# Show legend
ax1.legend(loc = "best", ncol = 2)
ax2.legend(loc = "best", ncol = 2)
ax3.legend(loc = "best", ncol = 2)
ax4.legend(loc = "best", ncol = 2)
ax5.legend(loc = "best", ncol = 2)

# Show legend
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

# Show the plot
plt.tight_layout()
plt.show()
