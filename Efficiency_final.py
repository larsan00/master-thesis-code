from Efficiency_animation import *
from Efficiency_equations import *
from Efficiency_Initialize import *

#simulation and animation parameter
dx = 0.01 # in meters
dt = 0.00001 # in seconds
simulationTime = 5 # in seconds
animation_steps = 200 # in frames

#initialize the grid of the room
steps, h, l, b, source, h_source, l_source, b_source = init_grid(simulationTime, dx, dt)

#initialize the velocity 
V_air = 0.7 # Volume of breathing in dm^3 = liter
D_0 = 0.02 # Nostril size
theta = 0

t_ex = 2.5
t_in = 2.5

v_ex = V_air / (1000 * D_0**2 * t_ex)
v_in = V_air / (1000 * D_0**2 * t_in)
print(v_ex)
print(v_in)

#initialize the temperature
h_newton = 0 # Heat transfer coefficient of air 
P = 0 # Heat power of a human

body_arr, h_size, l_size, b_size, nostril = init_temp(h_source, l_source, b_source, dx, D_0)

T_start = 273.15 + 20 # Initial temperature in K
T = T_start * np.ones((h, l, b)) 
T[body_arr[0]:body_arr[1]+1,body_arr[2]:body_arr[3]+1,body_arr[4]:body_arr[5]+1] = 0

#initialize the constants
c_spec, D, alpha_, R_s, v, g, V = init_const(dx, h_size, l_size, b_size)

#initialize the density
rho_start = 1.18 
H = 8559 # scale height

rho, c, vx, vy, vz = init_rho_c_v(rho_start, h, l, b, body_arr, nostril, dx, D_0, V_air, H, T, T_start)

Is_body, Is_body_x, Is_body_y, Is_body_z = body(h, l, b, body_arr)
dT_Human, dT_Wall, dc, A, B, C, D, Alpha, dt_dx = dt / (c_spec * 1.18 * V), dt / (c_spec * 1.18 * dx), 0, 2 * R_s / dx, v / dx**2, (-v / 3) / (4 * dx**2), D / dx, alpha_/ dx, dt / dx

# Run the main simulation loop
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
c, rho, T, vx, vy, vz = main_gpu(h, l, b, dx, dt, steps, source, g, Is_body, Is_body_x, Is_body_y, Is_body_z, animation_steps, c, rho, T, P, h_newton, dT_Human, dT_Wall, T_start, dc, D, Alpha, dt_dx, A, B, C, vx, vy, vz, nostril, theta, v_ex, v_in, t_ex, t_in)


mean_first = c[0, int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1].mean()
mean_last = c[-1, int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1].mean()

print("\n=== Simulations results ===\n")
print("Mean concentration in nostril at first   time step: ", mean_first)
print("Mean concentration in nostril at last    time step: ", mean_last)    

print(f"Reynolds-Number: {max(v_in, v_ex)*D_0/v:.2f} \n")
print(f"Reynolds-Number: {min(v_in, v_ex)*D_0/v:.2f} \n")


# Save the final results
#np.save("c_v1_T20_D.npy", c)
#np.save("rho_v1_T20_D.npy", rho)
#np.save("vy_v1_T20_D.npy", vy)

