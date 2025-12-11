//---------------------------------------------------------------------------//
// Front page to the AMP autodoc system.
//---------------------------------------------------------------------------//

/*!

\mainpage The AMP (Advanced Multi-Physics) Package Documentation System

\par Authors
<a href="mailto:philipb@ornl.gov">Bobby Philip (ORNL)</a>.<BR>
<a href="mailto:berrillma@ornl.gov">Mark Berrill (ORNL)</a>.<BR>
<a href="mailto:clarnokt@ornl.gov">Kevin Clarno (ORNL)</a>.<BR>
<a href="mailto:sampathrs@ornl.gov">Rahul Sampath (ORNL)</a>.<BR>
<a href="mailto:hamiltonsp@ornl.gov">Steven Hamilton (ORNL)</a>.<BR>

\par Executive Summary
Simulations of multiphysics systems are becoming increasingly important in many
science application areas.  The Advanced Multi-Physics (AMP) package designed
with multi-domain multi-physics applications in mind. AMP currently builds
powerful and flexible deterministic multiphysics simulation algorithms from
lightweight operator, solver, linear algebra, material database, discretization,
and meshing interfaces. AMP's agnostic framework provides at least two important benefits.

Firstly, it enables domain scientists with considerable investment in physics codes
to experiment with multi-physics couplings without having to adopt dramatically
different data structures for multiphysics problems. Physics components are
represented as discrete operators that map vectors between domain and range
spaces and encode the action of physics operators on vectors. A minimal interface
is required of each operator, consisting of the ability to specify: the action of
the operator on a vector, operator (re)initialization, and approximate linearization
(optional). Domain scientists can thus preserve their investments in existing physics
by encapsulating them as operators while at the same time preserving the ability to
create new physics operators with potentially different data structures.
In future releases, operator adaptor interfaces will further minimize code modifications
required. An operator can now represent a single physics or potentially multiple physics
either as a composition of single physics operators or as a single tightly coupled
multiphysics operator. The minimal operator interface allows easy import of higher
fidelity models without significant code rewrite. This approach has also naturally
allowed for multi-domain applications. Finally, it allows domain scientists and
mathematicians to study and analyze the strength of the couplings between various
physics components, numerical error propagation, and the sensitivities of output
quantities of interest to input and model parameters, data, boundary and initial conditions.

Secondly, considerable research investment by DOE, through the SciDAC TOPS, Advanced Simulation
and Computing (ASC) projects, and others has led to the development of highly sophisticated
software for solving individual physics efficiently at the petascale (e.g. PETSc, Trilinos,
SUNDIALS, Hypre). Moreover, further investments in software for solution methods for nonlinear
and linear multiphysics problems will have to be made as applications become more complex
and computer architectures evolve. By carefully specified solution interfaces, AMP is able
to leverage these existing investments, while keeping the door open to future solution
methodologies. Within AMP, physics solvers are considered as inverse or approximate inverse
operators with a minimal interface. The concept of operator composition now again enables
AMP users to harness existing software to efficiently realize different multiphysics solution
algorithms and experiment with new physics solvers without extensive code rewrites.
Through the AMP solver interfaces a subset of the PETSc, Trilinos, and SUNDIALS solvers and time
integrators are already available to users with the ability to add new solvers as required.
This allows users to balance needs for accuracy, robustness, and efficiency while respecting
the nature of couplings between different physics components, and choosing appropriate
solvers for the individual physics. Using Jacobian Free Newton Krylov (JFNK) and nonlinear
Krylov methods AMP provides the ability to solve the multiphysics systems composed from
individual physics in a nonlinearly consistent manner, with the flexibility to choose the
ordering of the individual physics in the solution process, as well as consider a variety
of alternative multiplicative (i.e., the updated solution in a module are immediately passed
on to the next coupled module for its solution processing), additive (i.e., components of
weakly coupled subsystems are solution processed asynchronously), and hybrid Schwarz solution
algorithms. The same concepts for operator and solver composition and decomposition are used
for time-dependent problems, which will be required for the solution of multiphysics modules
with time-scales that differ by orders of magnitudes. AMP users have the flexibility to
choose different time integrators for the different physics modules, tailored according
to the optimal strategy of each particular model using composition of individual time
integrators.

\par Components
AMP consists of the following components.  Click on the links to
find documentation about each component.<BR>
\ref AMP::LinearAlgebra "Linear Algebra"<BR>
\ref AMP::Discretization "Discretization"<BR>
\ref AMP::Geometry "Geometry"<BR>
\ref AMP::IO "IO"<BR>
\ref AMP::Materials "Materials"<BR>
\ref AMP::Mesh "Mesh"<BR>
\ref AMP::Operator "Operators"<BR>
\ref AMP::Solver "Solvers"<BR>
\ref AMP::TimeIntegrator "Time Integrators"<BR>
\ref AMP::Utilities "Utilities"<BR>
\ref Macros "Utility Macros"<BR>

\par Building:
Build options for AMP can be found \ref BuildOptions "here".

\par Examples
A limited set of examples are provided for illustrating the usage of the AMP components.
To build the examples use the command "make build-examples".
Additional cases may be demonstrated in the unit tests.<BR>

\par Documentation
To build the documentation use the command "make doc".  To build the basic documentation
without compiling AMP, the "-D ONLY_BUILD_DOCS=1" flag can be used: \verbatim
   /usr/bin/cmake                           \
      -D AMP_DATA:PATH=${PATH_TO_AMP_DATA}  \
      -D ONLY_BUILD_DOCS=1                  \
      ${PATH_TO_AMP_SOURCE}<BR>
\endverbatim
Note that building the documentation requires doxygen and dot.  Latex is recommended.

\par Packages


*/

// end of mainpage
