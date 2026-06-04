import numpy as np

# Initialization of basic constants
def init_const():

    c_spec = 1010 #Specific heat capacity air 
    D = 1.6 * 10**-5 #Diffusion coefficient of CO2 in air
    alpha_ =  2.1 * 10**-5 #Thermal diffusivity of air
    R_s = 287.1 #Specific gas constant of air
    v =  1.51 * 10**-5 #Kinematic viscosity of air
    g = 9.81 #Gravitational acceleration

    return c_spec, D, alpha_, R_s, v, g


# Initialization of the grid
def init_grid(simulationTime, dx, dt):

    steps = int(round(simulationTime/dt,0))
    h, l, b = int(round(1/dx,0)+1), int(round(1/dx,0)+1), int(round(1/dx,0)+1) 

    return steps, h, l, b


# Initialization of the density, concentrationand velocity fields
def init_rho_c_v(rho_start, h, l, b):

    rho = np.ones((h, l, b)) * rho_start
    rho[0::h-1,:,:], rho[:,0::l-1,:], rho[:,:,0::b-1] = 0, 0, 0

    c = np.ones((h, l, b)) * 450 # in ppm
    c[0::h-1,:,:], c[:,0::l-1,:], c[:,:,0::b-1] = 0, 0, 0
    
    vx, vy, vz = np.zeros((h-1, l, b)), np.zeros((h, l-1, b)), np.zeros((h, l, b-1))

    return rho, c, vx, vy, vz
