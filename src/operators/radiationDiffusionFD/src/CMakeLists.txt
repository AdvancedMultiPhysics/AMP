cmake_minimum_required(VERSION 3.10)

# Name of the project
project(FD)

# Add the file(s) we want to compile
add_executable(FD driver.cpp discretization.cpp solver.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # -g enables debugging symbols

# MPI
find_package(MPI REQUIRED)
target_link_libraries(FD PUBLIC MPI::MPI_CXX)

# AMP
#set(AMP_DIR /home/okrzysik/software/AdvancedMultiPhysics/AMP-github/build/debug)
set(AMP_DIR /home/okrzysik/software/AdvancedMultiPhysics/amp/build/debug)
target_link_libraries(FD PUBLIC "${AMP_DIR}/src/libamp.so")
target_include_directories(FD PUBLIC "${AMP_DIR}/include")

# External package DIR
set(PACKAGE_DIR /home/okrzysik/software/AdvancedMultiPhysics/packages/install/amp-tpls-openmpi-4.1.6-gcc-13.3.0-debug)
# Stacktrace
target_link_libraries(FD PUBLIC "${PACKAGE_DIR}/stacktrace/lib/libstacktrace.so")
target_include_directories(FD PUBLIC "${PACKAGE_DIR}/stacktrace/include")

# Timer
target_link_libraries(FD PUBLIC "${PACKAGE_DIR}/timer/lib/libtimerutility.so")
target_include_directories(FD PUBLIC "${PACKAGE_DIR}/timer/include")

# Petsc
target_link_libraries(FD PUBLIC "${PACKAGE_DIR}/petsc/lib/libpetsc.so")
target_include_directories(FD PUBLIC "${PACKAGE_DIR}/petsc/include")

# Hdf5
target_include_directories(FD PUBLIC "${PACKAGE_DIR}/hdf5/include")

# Hypre
target_link_libraries(FD PUBLIC "${PACKAGE_DIR}/hypre/lib/libHYPRE.so")
target_include_directories(FD PUBLIC "${PACKAGE_DIR}/hypre/include")





