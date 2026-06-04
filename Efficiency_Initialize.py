import numpy as np
import scipy.constants as cst

# Initialization of basic constants
def init_const(dx, h_size, l_size, b_size):

    c_spec = 1010 #Specific heat capacity air 
    D = 1.6 * 10**-5 #Diffusion coefficient of CO2 in air
    alpha_ =  2.1 * 10**-5 #Thermal diffusivity of air
    R_s = 287.1 #Specific gas constant of air
    v =  1.51 * 10**-5 #Kinematic viscosity of air
    g = cst.g

    #v *= 1000
    #D = 0

    V = dx**3 * 2*((h_size+1)*(l_size+1)+(l_size+1)*(b_size+1)+(b_size+1)*(h_size+1)) #Volume of Human surface in Simulation 
    return c_spec, D, alpha_, R_s, v, g, V


# Initialization of the temperature and the body
def init_temp(h_source, l_source, b_source, dx, D_0):

    h_size, l_size, b_size = round(0.2/dx,0), round(0.2/dx,0), round(0.15/dx,0)
    body_arr = np.array([int(h_source-h_size+1), int(h_source+1), int(l_source-l_size-1), int(l_source-1), int(b_source-0.5*b_size), int(b_source+0.5*b_size)])
    
    
    length = int(round(D_0 / dx))
    if  length == 0:
        print("ATTENTION! The nostril size is much smaller than the dx. This will cause errors in the simulation.")

    nostril = np.array([h_source+1-length, h_source, l_source-1, b_source+1-np.ceil(length/2), b_source+int(length/2)]) 
    print("The target nostril size is "+str(D_0)+"m.")
    print("The nostril has a x size of "+str((nostril[1]-nostril[0]+1)*dx)+"m and z size of "+str((nostril[4]-nostril[3]+1)*dx)+"m.")
    print(nostril)
    return body_arr, h_size, l_size, b_size, nostril


# Initialization of the grid
def init_grid(simulationTime, dx, dt):

    steps = int(round(simulationTime/dt,0))
    h, l, b = int(round(1/dx,0)+1), int(round(1/dx,0)+1), int(round(1/dx,0)+1)
    h_source, l_source, b_source = int(round(0.5/dx,0)), int(round(0.3/dx,0)), int(round((0.2)/dx,0))
 
    source = np.array([h_source, l_source, b_source])

    return steps, h, l, b, source, h_source, l_source, b_source


# Initialization of the density, concentrationand velocity fields
def init_rho_c_v(rho_start, h, l, b, body_arr, nostril, dx, D_0, V_air, H, T, T_start):

    rho = np.ones((h, l, b)) * rho_start

    for x in range(rho.shape[0]-2):
        rho[x+1, :, :] *= np.exp(-x * dx / H) * (1 - (T[x+1, int(l // 2), int(b // 2 )] / T_start - 1))

    rho[0::h-1,:,:], rho[:,0::l-1,:], rho[:,:,0::b-1] = 0, 0, 0
    rho[body_arr[0]:body_arr[1]+1,body_arr[2]:body_arr[3]+1,body_arr[4]:body_arr[5]+1] = 0

    rho_lung_norm = rho_start * (V_air * 0.001) / (dx * D_0**2)
    rho[int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1] = rho_lung_norm

    c = np.ones((h, l, b)) * 450
    c[0::h-1,:,:], c[:,0::l-1,:], c[:,:,0::b-1] = 0, 0, 0 
    c[body_arr[0]:body_arr[1]+1,body_arr[2]:body_arr[3]+1,body_arr[4]:body_arr[5]+1] = 0

    c_lung = 40000
    c_lung_norm = c_lung * (V_air * 0.001) / (dx*D_0**2)
    c[int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1] = c_lung_norm
    
    vx, vy, vz = np.zeros((h-1, l, b)), np.zeros((h, l-1, b)), np.zeros((h, l, b-1))

    return rho, c, vx, vy, vz

# Initialization of the body boundary conditions
def body(x_size, y_size, z_size, body):

    Is_body_long = np.zeros((x_size, y_size, z_size))
    Is_body_x = np.zeros((x_size, y_size, z_size), dtype=np.bool_)
    Is_body_y = np.zeros((x_size, y_size, z_size), dtype=np.bool_)
    Is_body_z = np.zeros((x_size, y_size, z_size), dtype=np.bool_)

    for x in range(1, x_size-1):
        for y in range(1, y_size-1):
            for z in range(1, z_size-1):

                Is_body = int((x >= body[0])) + int((x <= body[1])) + int((y >= body[2])) + int((y <= body[3])) + int((z >= body[4])) + int((z <= body[5]))

                if Is_body == 6:

                    Is_body_long[x,y,z] = -1 # grid point is within body

                Is_border = (x == 0) or (y == 0) or (z == 0) or (x == x_size-1) or (y == y_size-1) or (z == z_size-1)

                if Is_border == True:

                    Is_body_long[x,y,z] = -1 # grid point is on the border

                Is_edge = (x == 1) or (y == 1) or (z == 1) or (x == x_size-2) or (y == y_size-2) or (z == z_size-2)

                if Is_edge == True:

                    Is_body_long[x,y,z] = -2 #grid point is on the edge

                body_x = int(((x == body[0]-1) or (x == body[1]+1)) and (y >= body[2]) and (y <= body[3]) and (z >= body[4]) and (z <= body[5]))
                body_y = int(((y == body[2]-1) or (y == body[3]+1)) and (x >= body[0]) and (x <= body[1]) and (z >= body[4]) and (z <= body[5]))
                body_z = int(((z == body[4]-1) or (z == body[5]+1)) and (x >= body[0]) and (x <= body[1]) and (y >= body[2]) and (y <= body[3]))
                body_num = body_x + body_y + body_z  

                if body_num != 0:

                    Is_body_long[x,y,z] = 1 #grid point is on the surface of the body 

                # for velocity the three different directions are considered separately due to staggered grid
                
                Is_body_x[x, y, z] = (x >= body[0]-1) and (x <= body[1]) and (y >= body[2]) and (y <= body[3]) and (z >= body[4]) and (z <= body[5]) or (x == x_size-2)
                Is_body_y[x, y, z] = (x >= body[0]) and (x <= body[1]) and (y >= body[2]-1) and (y <= body[3]) and (z >= body[4]) and (z <= body[5]) or (y == y_size-2)
                Is_body_z[x, y, z] = (x >= body[0]) and (x <= body[1]) and (y >= body[2]) and (y <= body[3]) and (z >= body[4]-1) and (z <= body[5]) or (z == z_size-2)
           
    return Is_body_long, Is_body_x, Is_body_y, Is_body_z