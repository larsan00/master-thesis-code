import numpy as np
from numba import cuda
from numba import float64

def main_gpu(x_size, y_size, z_size, dx, dt, steps, animation_steps, c, rho, T, dt_dx, A, B, C, vx, vy, vz, h_newton, dT_Wall, Alpha, g, D):
    
    # Initialize the final arrays on the CPU
    c_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    rho_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    T_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    vx_final = np.zeros((animation_steps+1, x_size-1, y_size, z_size))
    vy_final = np.zeros((animation_steps+1, x_size, y_size-1, z_size))
    vz_final = np.zeros((animation_steps+1, x_size, y_size, z_size-1))

    c_final[0], rho_final[0], T_final[0], vx_final[0], vy_final[0], vz_final[0] = c, rho, T, vx, vy, vz
    k, max = 1, np.zeros((2, x_size, y_size, z_size))

    # Copy initial arrays to the gpu
    c_gpu, rho_gpu, T_gpu = cuda.to_device(c), cuda.to_device(rho), cuda.to_device(T)
    vx_gpu, vy_gpu, vz_gpu = cuda.to_device(vx), cuda.to_device(vy), cuda.to_device(vz)
    max_gpu = cuda.to_device(max)

    # Set up CUDA grid and block sizes
    threads_per_block_v = (4, 8, 4)
    blocks_per_grid_v = ((x_size + threads_per_block_v[0] - 1) // threads_per_block_v[0],
                       (y_size + threads_per_block_v[1] - 1) // threads_per_block_v[1],
                       (z_size + threads_per_block_v[2] - 1) // threads_per_block_v[2])
    threads_per_block_rho = (8, 8, 8)
    blocks_per_grid_rho = ((x_size + threads_per_block_rho[0] - 1) // threads_per_block_rho[0],
                       (y_size + threads_per_block_rho[1] - 1) // threads_per_block_rho[1],
                       (z_size + threads_per_block_rho[2] - 1) // threads_per_block_rho[2])



    # Iteration for diffusion and Navier-Stokes
    for n in range(1, steps + 1):      
        
        update_velocity_gpu[blocks_per_grid_v, threads_per_block_v](
                rho_gpu, T_gpu, vx_gpu, vy_gpu, vz_gpu, dx, dt, A, B, g) 
        
        update_navier_stokes_gpu[blocks_per_grid_rho, threads_per_block_rho](
            c_gpu, rho_gpu, T_gpu, vx_gpu, vy_gpu, vz_gpu, dt_dx, h_newton, dT_Wall, Alpha, D) 
                                                                      

        # Every animation step, copy results back from GPU to CPU
        if n % (steps // animation_steps) == 0:
            print(round(n * dt, 4), "seconds")  
          
            c_final[k] = c_gpu.copy_to_host()
            rho_final[k] = rho_gpu.copy_to_host()
            T_final[k] = T_gpu.copy_to_host()
            vx_final[k] = vx_gpu.copy_to_host()
            vy_final[k] = vy_gpu.copy_to_host()
            vz_final[k] = vz_gpu.copy_to_host()

            k += 1

    return c_final, rho_final, T_final, vx_final, vy_final, vz_final
     

# CUDA kernel for updating velocity fields
@cuda.jit
def update_velocity_gpu(rho, T, vx, vy, vz, dx, dt, A, B, g):
  
    # Calculate global thread ID and local thread ID
    x, y, z = cuda.grid(3)
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bdx, bdy, bdz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z

    # Define shared memory for the block    
    s_vx = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vy = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vz = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_rho = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_T = cuda.shared.array(shape=(8, 8, 8), dtype=float64)

    in_bounds = (x < rho.shape[0]) and (y < rho.shape[1]) and (z < rho.shape[2])

    s_vx[tx, ty, tz] = vx[x, y, z] if in_bounds else s_vx[tx, ty, tz]
    s_vy[tx, ty, tz] = vy[x, y, z] if in_bounds else s_vy[tx, ty, tz]
    s_vz[tx, ty, tz] = vz[x, y, z] if in_bounds else s_vz[tx, ty, tz]
    s_rho[tx, ty, tz] = rho[x, y, z] if in_bounds else s_rho[tx, ty, tz]
    s_T[tx, ty, tz]   = T[x, y, z]   if in_bounds else s_T[tx, ty, tz]

    cuda.syncthreads()

    # Load the central values for the current thread
    vx_x, vy_x, vz_x, rho_x, T_x = s_vx[tx, ty, tz], s_vy[tx, ty, tz], s_vz[tx, ty, tz], s_rho[tx, ty, tz], s_T[tx, ty, tz]

    vx_xm1 = vx[x-1, y, z] if tx == 0 else s_vx[tx-1, ty, tz]
    vx_xp1 = vx[x+1, y, z] if tx == bdx-1 and x < rho.shape[0]-1 else s_vx[tx+1, ty, tz]
    vx_ym1 = vx[x, y-1, z] if ty == 0 else s_vx[tx, ty-1, tz]
    vx_yp1 = vx[x, y+1, z] if ty == bdy-1 and y < rho.shape[1]-1 else s_vx[tx, ty+1, tz]
    vx_zm1 = vx[x, y, z-1] if tz == 0 else s_vx[tx, ty, tz-1]
    vx_zp1 = vx[x, y, z+1] if tz == bdz-1 and z < rho.shape[2]-1 else s_vx[tx, ty, tz+1]

    vy_xm1 = vy[x-1, y, z] if tx == 0 else s_vy[tx-1, ty, tz]
    vy_xp1 = vy[x+1, y, z] if tx == bdx-1 and x < rho.shape[0]-1 else s_vy[tx+1, ty, tz]
    vy_ym1 = vy[x, y-1, z] if ty == 0 else s_vy[tx, ty-1, tz]
    vy_yp1 = vy[x, y+1, z] if ty == bdy-1 and y < rho.shape[1]-1 else s_vy[tx, ty+1, tz]
    vy_zm1 = vy[x, y, z-1] if tz == 0 else s_vy[tx, ty, tz-1]
    vy_zp1 = vy[x, y, z+1] if tz == bdz-1 and z < rho.shape[2]-1 else s_vy[tx, ty, tz+1]

    vz_xm1 = vz[x-1, y, z] if tx == 0 else s_vz[tx-1, ty, tz]
    vz_xp1 = vz[x+1, y, z] if tx == bdx-1 and x < rho.shape[0]-1 else s_vz[tx+1, ty, tz]
    vz_ym1 = vz[x, y-1, z] if ty == 0 else s_vz[tx, ty-1, tz]
    vz_yp1 = vz[x, y+1, z] if ty == bdy-1 and y < rho.shape[1]-1 else s_vz[tx, ty+1, tz]
    vz_zm1 = vz[x, y, z-1] if tz == 0 else s_vz[tx, ty, tz-1]
    vz_zp1 = vz[x, y, z+1] if tz == bdz-1 and z < rho.shape[2]-1 else s_vz[tx, ty, tz+1]

    rho_xp1 = rho[x+1, y, z] if tx == bdx-1 and x < rho.shape[0]-1 else s_rho[tx+1, ty, tz]
    rho_yp1 = rho[x, y+1, z] if ty == bdy-1 and y < rho.shape[1]-1 else s_rho[tx, ty+1, tz]
    rho_zp1 = rho[x, y, z+1] if tz == bdz-1 and z < rho.shape[2]-1 else s_rho[tx, ty, tz+1]

    T_xp1 = T[x+1, y, z] if tx == bdx-1 and x < rho.shape[0]-1 else s_T[tx+1, ty, tz]
    T_yp1 = T[x, y+1, z] if ty == bdy-1 and y < rho.shape[1]-1 else s_T[tx, ty+1, tz]
    T_zp1 = T[x, y, z+1] if tz == bdz-1 and z < rho.shape[2]-1 else s_T[tx, ty, tz+1]

    vy_xp1_ym1 = vy[x+1, y-1, z] if (tx == bdx-1 or ty == 0) and x < rho.shape[0]-1 and y > 0 else s_vy[tx+1, ty-1, tz]
    vz_xp1_zm1 = vz[x+1, y, z-1] if (tx == bdx-1 or tz == 0) and x < rho.shape[0]-1 and z > 0 else s_vz[tx+1, ty, tz-1]
    vx_xm1_yp1 = vx[x-1, y+1, z] if (tx == 0 or ty == bdy-1) and x > 0 and y < rho.shape[1]-1 else s_vx[tx-1, ty+1, tz]
    vz_yp1_zm1 = vz[x, y+1, z-1] if (ty == bdy-1 or tz == 0) and y < rho.shape[1]-1 and z > 0 else s_vz[tx, ty+1, tz-1]
    vx_xm1_zp1 = vx[x-1, y, z+1] if (tx == 0 or tz == bdz-1) and x > 0 and z < rho.shape[2]-1 else s_vx[tx-1, ty, tz+1]
    vy_ym1_zp1 = vy[x, y-1, z+1] if (ty == 0 or tz == bdz-1) and y > 0 and z < rho.shape[2]-1 else s_vy[tx, ty-1, tz+1]

    if 1 <= x < rho.shape[0]-1 and 1 <= y < rho.shape[1]-1 and 1 <= z < rho.shape[2]-1:

        vx_diff, vy_diff, vz_diff = 0.0, 0.0, 0.0
   
        # Calculation for x-velocity
        if x != vx.shape[0]-1:
            vy_bar = (vy_ym1 + vy_x + vy_xp1_ym1 + vy_xp1) / 4
            vz_bar = (vz_zm1 + vz_x + vz_xp1_zm1 + vz_xp1) / 4
            
            if vx_x > 0:
                vx_diff -= vx_x * (vx_x - vx_xm1)
            else:
                vx_diff -= vx_x * (vx_xp1 - vx_x) 

            if vy_bar > 0:
                vx_diff -= vy_bar * (vx_x - vx_ym1)
            else:
                vx_diff -= vy_bar * (vx_yp1 - vx_x)

            if vz_bar > 0:
                vx_diff -= vz_bar * (vx_x - vx_zm1)
            else:
                vx_diff -= vz_bar * (vx_zp1 - vx_x)
            vx_diff /= dx
            
            vx_diff -= A * (rho_xp1 * T_xp1 - rho_x * T_x) / (rho_x + rho_xp1)
            vx_diff += B * (vx_xp1 + vx_xm1 + vx_yp1 + vx_ym1 + vx_zp1 + vx_zm1 - 6 * vx_x)

            vx_diff -= g
            
            vx[x, y, z] += dt * vx_diff

        # Calculation for y-velocity
        if y != vy.shape[1]-1:
            vx_bar = (vx_xm1 + vx_x + vx_xm1_yp1 + vx_yp1) / 4
            vz_bar = (vz_zm1 + vz_x + vz_yp1_zm1 + vz_yp1) / 4

            if vy_x > 0:
                vy_diff -= vy_x * (vy_x - vy_ym1)
            else:
                vy_diff -= vy_x * (vy_yp1 - vy_x)

            if vx_bar > 0:
                vy_diff -= vx_bar * (vy_x - vy_xm1)
            else:
                vy_diff -= vx_bar * (vy_xp1 - vy_x)

            if vz_bar > 0:
                vy_diff -= vz_bar * (vy_x - vy_zm1)
            else:
                vy_diff -= vz_bar * (vy_zp1 - vy_x)
            vy_diff /= dx

            vy_diff -= A * (rho_yp1 * T_yp1 - rho_x * T_x) / (rho_x + rho_yp1)
            vy_diff += B * (vy_xp1 + vy_xm1 + vy_yp1 + vy_ym1 + vy_zp1 + vy_zm1 - 6 * vy_x)
            
            vy[x, y, z] +=  dt * vy_diff
            

        # Calculation for z-velocity
        if z != vz.shape[2]-1:
            vx_bar = (vx_xm1 + vx_x + vx_xm1_zp1 + vx_zp1) / 4
            vy_bar = (vy_ym1 + vy_x + vy_ym1_zp1 + vy_zp1) / 4

            if vz_x > 0:
                vz_diff -= vz_x * (vz_x - vz_zm1)
            else:
                vz_diff -= vz_x * (vz_zp1 - vz_x)

            if vx_bar > 0:
                vz_diff -= vx_bar * (vz_x - vz_xm1)
            else:
                vz_diff -= vx_bar * (vz_zp1 - vz_x)

            if vy_bar > 0:
                vz_diff -= vy_bar * (vz_x - vz_ym1)
            else:
                vz_diff -= vy_bar * (vz_yp1 - vz_x)
            vz_diff /= dx 

            vz_diff -= A * (rho_zp1 * T_zp1 - rho_x * T_x) / (rho_x + rho_zp1)
            vz_diff += B * (vz_xp1 + vz_xm1 + vz_yp1 + vz_ym1 + vz_zp1 + vz_zm1 - 6 * vz_x)
            
            vz[x, y, z] += dt * vz_diff
   
# CUDA kernel for updating concentration, density and temperature fields
@cuda.jit
def update_navier_stokes_gpu(c, rho, T, vx, vy, vz, dt_dx, h_newton, dT_Wall, Alpha, D):
   
    # Calculate global thread ID and local thread ID
    x, y, z = cuda.grid(3)
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bdx, bdy, bdz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    
    # Define shared memory for the block
    s_vx = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vy = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vz = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_rho = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_T = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_c = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    
    in_bounds = (x < rho.shape[0]) and (y < rho.shape[1]) and (z < rho.shape[2])

    s_vx[tx, ty, tz] = vx[x, y, z] if in_bounds else s_vx[tx, ty, tz]
    s_vy[tx, ty, tz] = vy[x, y, z] if in_bounds else s_vy[tx, ty, tz]
    s_vz[tx, ty, tz] = vz[x, y, z] if in_bounds else s_vz[tx, ty, tz]
    s_rho[tx, ty, tz] = rho[x, y, z] if in_bounds else s_rho[tx, ty, tz]
    s_T[tx, ty, tz]   = T[x, y, z]   if in_bounds else s_T[tx, ty, tz]
    s_c[tx, ty, tz]   = c[x, y, z]   if in_bounds else s_c[tx, ty, tz]
    
    cuda.syncthreads() 
    
    # Load the central values for the current thread
    vx_x, vy_x, vz_x, rho_x, T_x, c_x = s_vx[tx, ty, tz], s_vy[tx, ty, tz], s_vz[tx, ty, tz], s_rho[tx, ty, tz], s_T[tx, ty, tz], s_c[tx, ty, tz]
    
    vx_xm1 = vx[x-1, y, z] if (tx == 0 and x > 0) else s_vx[tx-1, ty, tz]
    rho_xm1 = rho[x-1, y, z] if (tx == 0 and x > 0) else s_rho[tx-1, ty, tz]
    T_xm1   = T[x-1, y, z]   if (tx == 0 and x > 0) else s_T[tx-1, ty, tz]
    c_xm1   = c[x-1, y, z]   if (tx == 0 and x > 0) else s_c[tx-1, ty, tz]
    
    rho_xp1 = rho[x+1, y, z] if (tx == bdx-1 and x < rho.shape[0]-1) else s_rho[tx+1, ty, tz]
    T_xp1   = T[x+1, y, z]   if (tx == bdx-1 and x < rho.shape[0]-1) else s_T[tx+1, ty, tz]
    c_xp1   = c[x+1, y, z]   if (tx == bdx-1 and x < rho.shape[0]-1) else s_c[tx+1, ty, tz]
    
    vy_ym1 = vy[x, y-1, z] if (ty == 0 and y > 0) else s_vy[tx, ty-1, tz]
    rho_ym1 = rho[x, y-1, z] if (ty == 0 and y > 0) else s_rho[tx, ty-1, tz]
    T_ym1   = T[x, y-1, z]   if (ty == 0 and y > 0) else s_T[tx, ty-1, tz]
    c_ym1   = c[x, y-1, z]   if (ty == 0 and y > 0) else s_c[tx, ty-1, tz]
    
    rho_yp1 = rho[x, y+1, z] if (ty == bdy-1 and y < rho.shape[1]-1) else s_rho[tx, ty+1, tz]
    T_yp1   = T[x, y+1, z]   if (ty == bdy-1 and y < rho.shape[1]-1) else s_T[tx, ty+1, tz]
    c_yp1   = c[x, y+1, z]   if (ty == bdy-1 and y < rho.shape[1]-1) else s_c[tx, ty+1, tz]
    
    vz_zm1 = vz[x, y, z-1] if (tz == 0 and z > 0) else s_vz[tx, ty, tz-1]
    rho_zm1 = rho[x, y, z-1] if (tz == 0 and z > 0) else s_rho[tx, ty, tz-1]
    T_zm1   = T[x, y, z-1]   if (tz == 0 and z > 0) else s_T[tx, ty, tz-1]
    c_zm1   = c[x, y, z-1]   if (tz == 0 and z > 0) else s_c[tx, ty, tz-1]
    
    rho_zp1 = rho[x, y, z+1] if (tz == bdz-1 and z < rho.shape[2]-1) else s_rho[tx, ty, tz+1]
    T_zp1   = T[x, y, z+1]   if (tz == bdz-1 and z < rho.shape[2]-1) else s_T[tx, ty, tz+1]
    c_zp1   = c[x, y, z+1]   if (tz == bdz-1 and z < rho.shape[2]-1) else s_c[tx, ty, tz+1]
    
    if x >= 1 and x < c.shape[0]-1 and y >= 1 and y < c.shape[1]-1 and z >= 1 and z < c.shape[2]-1:
        

        c_diff, rho_diff, T_diff = 0.0, 0.0, 0.0  

        T_diff += Alpha * (T_xp1 + T_xm1 + T_yp1 + T_ym1 + T_zp1 + T_zm1 - 6.0 * T_x) 
        T_diff += h_newton * dT_Wall / dt_dx * ((T_xm1 - T_x) * int((x == 1)) + (T_ym1 - T_x) * int((y == 1)) + (T_zm1 - T_x) * int((z == 1)) + (T_xp1 - T_x) * int((x == T.shape[0] - 2)) + (T_yp1 - T_x) * int((y == T.shape[1] - 2)) + (T_zp1 - T_x) * int((z == T.shape[2] - 2)))            
        
        edge = (x == 1) + (y == 1) + (z == 1) + (x == c.shape[0]-2) + (y == c.shape[1]-2) + (z == c.shape[2]-2)
        c_diff += D * (c_xp1 + c_xm1 + c_yp1 + c_ym1 + c_zp1 + c_zm1 - (6.0 - edge) * c_x)

        # Calculation of the advection terms in x-, y- and z-direction
        # x-direction
        if vx_x > 0:
            c_diff -= c_x * vx_x
            rho_diff -= rho_x * vx_x
            T_diff -= T_x * vx_x
        else:
            c_diff -= c_xp1 * vx_x
            rho_diff -= rho_xp1 * vx_x
            T_diff -= T_xp1 * vx_x

        if vx_xm1 > 0:
            c_diff += c_xm1 * vx_xm1
            rho_diff += rho_xm1 * vx_xm1
            T_diff += T_xm1 * vx_xm1
        else:
            c_diff += c_x * vx_xm1
            rho_diff += rho_x * vx_xm1
            T_diff += T_x * vx_xm1

        # y-direction
        if vy_x > 0:
            c_diff -= c_x * vy_x
            rho_diff -= rho_x * vy_x
            T_diff -= T_x * vy_x
        else:
            c_diff -= c_yp1 * vy_x
            rho_diff -= rho_yp1 * vy_x
            T_diff -= T_yp1 * vy_x

        if vy_ym1 > 0:
            c_diff += c_ym1 * vy_ym1
            rho_diff += rho_ym1 * vy_ym1
            T_diff += T_ym1 * vy_ym1
        else:
            c_diff += c_x * vy_ym1
            rho_diff += rho_x * vy_ym1
            T_diff += T_x * vy_ym1

        # z-direction
        if vz_x > 0:
            c_diff -= c_x * vz_x
            rho_diff -= rho_x * vz_x
            T_diff -= T_x * vz_x
        else:
            c_diff -= c_zp1 * vz_x
            rho_diff -= rho_zp1 * vz_x
            T_diff -= T_zp1 * vz_x

        if vz_zm1 > 0:
            c_diff += c_zm1 * vz_zm1
            rho_diff += rho_zm1 * vz_zm1
            T_diff += T_zm1 * vz_zm1
        else:
            c_diff += c_x * vz_zm1
            rho_diff += rho_x * vz_zm1
            T_diff += T_x * vz_zm1

        c_diff, rho_diff, T_diff = c_diff * dt_dx, rho_diff * dt_dx, T_diff * dt_dx
    
        # Update of the concentration, density and temperature fields
        c[x, y, z] = c_x + c_diff
        rho[x, y, z] = rho_x + rho_diff 
        T[x, y, z] = T_x + T_diff

