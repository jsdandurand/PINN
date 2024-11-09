# Import libraries
import numpy 
from matplotlib import pyplot as plt
from scipy import interpolate
import os
def update_diff_2D(T, sigma_x, sigma_z, nx, nz):
    
    # store old temperature field
    Tn = T.copy()
        
    # loop over spatial grid 
    for i in range(1,nx-1):
        for j in range(1,nz-1):
                
            T[j, i] = (Tn[j, i] +
                        sigma_x * (Tn[j, i+1] - 2.0 * Tn[j, i] + Tn[j, i-1]) +
                        sigma_z * (Tn[j+1, i] - 2.0 * Tn[j, i] + Tn[j-1, i]))
            
    return T            

# Marker-in-cell code to solve the 2D advection equation
# ------------------------------------------------------
def Adv_diff_2D(T0, nt, dt, dx, dz, Vx, Vz, Lx1, Lx2, Lz1, Lz2, x, z, X, Z, alpha):
    """
    Computes and returns the temperature distribution
    after a given number of time steps for the 2D advection 
    problem. A marker-in-cell approach with Dirichlet conditions 
    on all boundaries is used in order to mitigate the effect of 
    numerical diffusion.
    
    Parameters
    ----------
    T0 : numpy.ndarray
        The initial temperature distribution as a 2D array of floats.
    nt : integer
        Maximum number of time steps to compute.
    dt : float
        Time-step size.
    dx : float
        Grid spacing in the x direction.
    dz : float
        Grid spacing in the z direction.
    Vx : float
        x-component of the velocity field.
    Vz : float
        z-component of the velocity field.        
    Lx1, Lx2 : float
        Model extension from Lx1 - Lx2.
    Lz1, Lz2 : float
        Model extension from Lz1 - Lz2.
    x, z : float
        Model coordinates as 1D arrays.
    X, Z : float
        Model coordinates as 2D arrays.    
        
    
    Returns
    -------
    T : numpy.ndarray
        The temperature distribution as a 2D array of floats.
    """

    # Integrate in time.
    T = T0.copy()
    
    # Estimate number of grid points in x- and z-direction
    nz, nx = T.shape
    
    # Define number of markers and initial marker positions
    nx_mark = 4 * nx  # number of markers in x-direction
    nz_mark = 4 * nz  # number of markers in z-direction    
    
    # Define some constants.
    sigma_x = alpha * dt / dx**2
    sigma_z = alpha * dt / dz**2
    
    # Time loop
    for n in range(nt):        
        
        actual_time = n * dt
        # initial marker positions
        x_mark = numpy.linspace(Lx1, Lx2, num=nx_mark)
        z_mark = numpy.linspace(Lz1, Lz2, num=nz_mark)
        X_mark, Z_mark = numpy.meshgrid(x_mark,z_mark)
        
        # Interpolate velocities from grid to marker position at timestep n        
        f = interpolate.interp2d(x, z, Vx, kind='linear')
        vx_mark_n = f(x_mark, z_mark)
        
        f = interpolate.interp2d(x, z, Vz, kind='linear')
        vz_mark_n = f(x_mark, z_mark)
        
        # Interpolate temperature from grid to marker position at timestep n
        f = interpolate.interp2d(x, z, T, kind='cubic')
        T_mark = f(x_mark, z_mark)
        
        # Save current marker positions
        X0 = X_mark
        Z0 = Z_mark
        
        # Update marker position
        X_mark = X_mark + vx_mark_n * dt
        Z_mark = Z_mark + vz_mark_n * dt
        
        # Interpolate velocities from grid to marker position at timestep n+1 
        vx_mark_n1 = interpolate.griddata((X.flatten(), Z.flatten()), Vx.flatten(), (X_mark, Z_mark), method='linear')
        vz_mark_n1 = interpolate.griddata((X.flatten(), Z.flatten()), Vz.flatten(), (X_mark, Z_mark), method='linear')
        
        # Replace Nan values 
        mask = numpy.where(numpy.isnan(vx_mark_n1))
        vx_mark_n1[mask] = 0
        
        mask = numpy.where(numpy.isnan(vz_mark_n1))
        vz_mark_n1[mask] = 0
        
        # Update marker position with midpoint velocity
        X_mark = X0 + dt * (vx_mark_n + vx_mark_n1) / 2.
        Z_mark = Z0 + dt * (vz_mark_n + vz_mark_n1) / 2.

        # Interpolate temperature field from marker to grid positions
        T = interpolate.griddata((X_mark.flatten(), Z_mark.flatten()), T_mark.flatten(), (X, Z), method='cubic')
        
        # Replace Nan-values by old temperature field 
        mask = numpy.where(numpy.isnan(T))
        T[mask] = T0[mask]
        
        # Update diffusion equation
        T = update_diff_2D(T, sigma_x, sigma_z, nx, nz)
        
    return T

def pre_estimate(X_0, C, ADE_params, args):

    # Definition of modelling parameters
    # ----------------------------------
    Lx = (args.ub[0] - args.lb[0]) / 2 # half length
    Lz = (args.ub[1] - args.lb[1]) / 2 # half width (y refers
    nx = numpy.sqrt(len(X_0)).astype(int)  # number of points in the x direction
    nz = nx  # number of points in the z direction
    dx = 2 * Lx / (nx - 1)  # grid spacing in the x direction
    dz = 2 * Lz / (nz - 1)  # grid spacing in the z direction

    u, v, K = ADE_params  # velocity field and diffusion coefficient
    alpha = K

    x = numpy.linspace(-Lx, Lx, nx)
    z = numpy.linspace(-Lz, Lz, nz)

    X, Z = X_0[:, 0].reshape(nx, nx), X_0[:, 1].reshape(nx, nx)

    X_1, Z_1 = numpy.meshgrid(x, z)

    assert numpy.allclose(X, X_1)
    assert numpy.allclose(Z, Z_1)

    # initial temperature distribution
    T0 = C.reshape(nx, nx)

    # Define velocity field (Vx,Vz)
    Vx = u(X, Z)
    Vz = v(X, Z)

    # # Plot the initial temperature distribution and streamlines
    # plt.figure(figsize=(10.0, 7))

    # extent = [numpy.min(X), numpy.max(X),numpy.min(Z), numpy.max(Z)]
    # cmap = 'gist_heat'
    # im = plt.imshow(numpy.flipud(T0), extent=extent, interpolation='spline36', cmap=cmap)
    # stream = plt.streamplot(X,Z,Vx,Vz,color='w')

    # plt.xlabel('x [m]')
    # plt.ylabel('z [m]')
    # cbar = plt.colorbar(im)
    # cbar.set_label('Initial Temperature [°C]')
    # plt.show()

    dt = args.time_step
    nt = args.n_steps

    print("==================================================================")
    print(f"Using Finite Difference Approximation to pre estimate {dt * nt} seconds")
    print("==================================================================")

    C = Adv_diff_2D(T0, nt, dt, dx, dz, Vx, Vz, -Lx, Lx, -Lz, Lz, x, z, X, Z,alpha)

    # Plot the final temperature distribution and streamlines
        # Plot the initial temperature distribution and streamlines
        # Plot initial data and velocity field
    plt.figure(figsize=(7, 7))
    plt.scatter(X.flatten(), Z.flatten(), c=C.flatten(), cmap='viridis')
    # Streamplot
    plt.streamplot(X, Z, Vx, Vz, color='k')
    cbar = plt.colorbar()
    cbar.set_label('Concentration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Initial data and velocity field')
    # Make plot square
    plt.gca().set_aspect('equal', adjustable='box')
    # Save plot
    output_dir = args.output_dir
    # Experiment name
    exp_name = args.exp_name
    # combine output directory and experiment name
    output_dir = os.path.join(output_dir, exp_name)
    plt.savefig(os.path.join(output_dir, 'initial_data_pre_estim.png'))

    # Change C to original shape
    C = C.flatten()
    return C


