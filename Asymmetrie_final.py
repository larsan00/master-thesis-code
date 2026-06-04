from Asymmetrie_animation import *
from Asymmetrie_equations import *
from Asymmetrie_Initialize import *

#simulation and animation parameter
dx = 0.01 # in meters
dt = 0.00001 # in seconds
simulationTime = 1000 # in seconds
animation_steps = 200 # in frames

F = np.array([0.0, 1.15*10**-10, 0.0]) 

#initialize the grid of the room
steps, h, l, b = init_grid(simulationTime, dx, dt)

T_start = 273.15 + 20 # Initial temperature in K
T = T_start * np.ones((h, l, b)) 

#initialize the constants
c_spec, D, alpha_, R_s, v = init_const()

#initialize the density
rho_start = 1.18 
H = 8559 # scale height

rho, c, vx, vy, vz = init_rho_c_v(rho_start, h, l, b)
dT_Wall, A, B, C, D, Alpha, dt_dx = dt / (c_spec * 1.18 * dx), 2 * R_s / dx, v / dx**2, (-v / 3) / (4 * dx**2), D / dx, alpha_/ dx, dt / dx


# Landau Jet 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ux, uy, uz, _ = landau_jet_grid(h, l, b, dx, F[1], 1.18, v)
v0 = np.max(uy)
print("max Landau velocity: ", np.max(uy))

f_name = '_landau_dx_'+str(dx)
#np.save('vy' + f_name, uy[:,:,uz.shape[2]//2])


# Non Oscillating Oseen velocity field
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ux2, uy2, uz2 = compute_oseen_velocity(F, v, rho_start, dx, vx, vy, vz)
print("max Oseen velocity: ", np.max(uy2))

f_name = '_oseen_dx_'+str(dx)
#np.save('vy' + f_name, uy2[:,:,uy2.shape[2]//2])


# Run the main simulation loop
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
c, rho, T, vx, vy, vz = main_gpu(h, l, b, dx, dt, steps, animation_steps, c, rho, T, dt_dx, A, B, C, vx, vy, vz, v0, F)

reynolds_number(F[1], rho_start, np.max(vy), dx)

# save the final velocity field for the animation
f_name = '_sim_dx_'+str(dx)+'.'
#np.save('vy' + f_name + "npy", vy[:,:,:,vy.shape[3]//2])

