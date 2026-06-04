import numpy as np
from numba import cuda
from numba import float64

def main_gpu(x_size, y_size, z_size, dx, dt, steps, source, g, Is_body, Is_body_x, Is_body_y, Is_body_z, animation_steps, c, rho, T, P, h_newton, dT_Human, dT_Wall, T_start, dc, D, Alpha, dt_dx, A, B, C, vx, vy, vz, nostril, theta, v_ex, v_in, t_ex, t_in):
    
    # Initialize the final arrays on the CPU
    c_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    rho_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    T_final = np.zeros((animation_steps+1, x_size, y_size, z_size))
    vx_final = np.zeros((animation_steps+1, x_size-1, y_size, z_size))
    vy_final = np.zeros((animation_steps+1, x_size, y_size-1, z_size))
    vz_final = np.zeros((animation_steps+1, x_size, y_size, z_size-1))

    c_final[0], rho_final[0], T_final[0], vx_final[0], vy_final[0], vz_final[0] = c, rho, T, vx, vy, vz
    k, max = 1, np.zeros((2, x_size, y_size, z_size))
    max[0, int(nostril[0]):int(nostril[1]+1), source[1], int(nostril[3]):int(nostril[4])+1], max[1, int(nostril[0]):int(nostril[1]+1), source[1], int(nostril[3]):int(nostril[4])+1] = c[int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1], rho[int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1]

    # Copy initial arrays to the gpu
    c_gpu, rho_gpu, T_gpu = cuda.to_device(c), cuda.to_device(rho), cuda.to_device(T)
    vx_gpu, vy_gpu, vz_gpu = cuda.to_device(vx), cuda.to_device(vy), cuda.to_device(vz)
    source_gpu = cuda.to_device(source)
    Is_body_gpu = cuda.to_device(Is_body)
    Is_body_x_gpu, Is_body_y_gpu, Is_body_z_gpu = cuda.to_device(Is_body_x), cuda.to_device(Is_body_y), cuda.to_device(Is_body_z)
    max_gpu = cuda.to_device(max)
    nostril_gpu = cuda.to_device(nostril)

    # Set up CUDA grid and block sizes
    threads_per_block_v = (4, 8, 4)
    blocks_per_grid_v = ((x_size + threads_per_block_v[0] - 1) // threads_per_block_v[0],
                       (y_size + threads_per_block_v[1] - 1) // threads_per_block_v[1],
                       (z_size + threads_per_block_v[2] - 1) // threads_per_block_v[2])
    threads_per_block_rho = (8, 8, 8)
    blocks_per_grid_rho = ((x_size + threads_per_block_rho[0] - 1) // threads_per_block_rho[0],
                       (y_size + threads_per_block_rho[1] - 1) // threads_per_block_rho[1],
                       (z_size + threads_per_block_rho[2] - 1) // threads_per_block_rho[2])

    # Precompute sin and cos of breathing velocity
    vx_ex = v_ex * np.sin(np.radians(theta))
    vy_ex = v_ex * np.cos(np.radians(theta))

    # velocity decomposition for inhalation
    vx_in = v_in * np.sin(np.radians(theta))
    vy_in = v_in * np.cos(np.radians(theta))

    # Iteration for diffusion and Navier-Stokes
    for n in range(1, steps + 1):      

        # CPU Calculation of breathing velocity components
        if n * dt % (t_ex + t_in) < t_ex:
            vx_breath = -vx_ex
            vy_breath = vy_ex
        else:
            vx_breath = vx_in
            vy_breath = -vy_in
        
        update_velocity_gpu[blocks_per_grid_v, threads_per_block_v](
                rho_gpu, T_gpu, vx_gpu, vy_gpu, vz_gpu, dx, dt, A, B, C, g, Is_body_x_gpu, Is_body_y_gpu, Is_body_z_gpu, nostril_gpu, vx_breath, vy_breath) 
        
        update_navier_stokes_gpu[blocks_per_grid_rho, threads_per_block_rho](
            c_gpu, rho_gpu, T_gpu, vx_gpu, vy_gpu, vz_gpu, D, dc, dt_dx, Is_body_gpu, source_gpu, P, T_start, Alpha, h_newton, dT_Human, dT_Wall, max_gpu, dt, nostril_gpu, v_in, t_ex) 
                                                                      

        # Every animation step, copy results back from GPU to CPU
        if n % (steps // animation_steps) == 0:
            print(round(n * dt, 4), "seconds")  
          
            c_final[k] = c_gpu.copy_to_host()
            rho_final[k] = rho_gpu.copy_to_host()
            T_final[k] = T_gpu.copy_to_host()
            vx_final[k] = vx_gpu.copy_to_host()
            vy_final[k] = vy_gpu.copy_to_host()
            vz_final[k] = vz_gpu.copy_to_host()

            print("The total density is " + str(round(np.sum(rho_final[k-1]), 1)))
            print(np.sum(rho_final[k-1]))
            
            k += 1

    return c_final, rho_final, T_final, vx_final, vy_final, vz_final
     

@cuda.jit
def update_velocity_gpu(rho, T, vx, vy, vz, dx, dt, A, B, C, g, Is_body_x, Is_body_y, Is_body_z, nostril, vx_breath, vy_breath):
  
    # Calculate global thread ID and local thread ID
    x, y, z = cuda.grid(3)
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bdx, bdy, bdz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    
    sx, sy, sz = tx+1, ty+1, tz+1
    
    # Define shared memory for the block 
    s_vx  = cuda.shared.array(shape=(6, 10, 6), dtype=float64)
    s_vy  = cuda.shared.array(shape=(6, 10, 6), dtype=float64)
    s_vz  = cuda.shared.array(shape=(6, 10, 6), dtype=float64)
    s_rho = cuda.shared.array(shape=(4, 8, 4), dtype=float64)
    s_T   = cuda.shared.array(shape=(4, 8, 4), dtype=float64)

    in_rho_c_T = (x < rho.shape[0]) and (y < rho.shape[1]) and (z < rho.shape[2])
    in_vx  = (x < vx.shape[0]) and (y < vx.shape[1]) and (z < vx.shape[2])
    in_vy  = (x < vy.shape[0]) and (y < vy.shape[1]) and (z < vy.shape[2])
    in_vz  = (x < vz.shape[0]) and (y < vz.shape[1]) and (z < vz.shape[2])

    s_vx[sx, sy, sz]  = vx[x, y, z]  if in_vx else 0.0
    s_vy[sx, sy, sz]  = vy[x, y, z]  if in_vy else 0.0
    s_vz[sx, sy, sz]  = vz[x, y, z]  if in_vz else 0.0
    s_rho[tx, ty, tz] = rho[x, y, z] if in_rho_c_T else 0.0
    s_T[tx, ty, tz]   = T[x, y, z]   if in_rho_c_T else 0.0

    if tx == 0:
        if x > 0 and in_vx:
            s_vx[0, sy, sz] = vx[x-1, y, z]
        else:
            s_vx[0, sy, sz] = 0.0
        
        if x > 0 and in_vy:
            s_vy[0, sy, sz] = vy[x-1, y, z]
        else:
            s_vy[0, sy, sz] = 0.0
        
        if x > 0 and in_vz:
            s_vz[0, sy, sz] = vz[x-1, y, z]
        else:
            s_vz[0, sy, sz] = 0.0

    if tx == bdx - 1:
        if x + 1 < vx.shape[0] and in_vx:
            s_vx[5, sy, sz] = vx[x+1, y, z]
        else:
            s_vx[5, sy, sz] = 0.0

        if x + 1 < vy.shape[0] and in_vy:
            s_vy[5, sy, sz] = vy[x+1, y, z]
        else:
            s_vy[5, sy, sz] = 0.0

        if x + 1 < vz.shape[0] and in_vz:
            s_vz[5, sy, sz] = vz[x+1, y, z]
        else:
            s_vz[5, sy, sz] = 0.0

    if ty == 0:
        if y > 0 and in_vx:
            s_vx[sx, 0, sz] = vx[x, y-1, z]
        else:
            s_vx[sx, 0, sz] = 0.0

        if y > 0 and in_vy:
            s_vy[sx, 0, sz] = vy[x, y-1, z]
        else:
            s_vy[sx, 0, sz] = 0.0

        if y > 0 and in_vz:
            s_vz[sx, 0, sz] = vz[x, y-1, z]
        else:
            s_vz[sx, 0, sz] = 0.0

    if ty == bdy - 1:
        if y + 1 < vx.shape[1] and in_vx:
            s_vx[sx, 9, sz] = vx[x, y+1, z]
        else:
            s_vx[sx, 9, sz] = 0.0

        if y + 1 < vy.shape[1] and in_vy:
            s_vy[sx, 9, sz] = vy[x, y+1, z]
        else:
            s_vy[sx, 9, sz] = 0.0

        if y + 1 < vz.shape[1] and in_vz:
            s_vz[sx, 9, sz] = vz[x, y+1, z]
        else:
            s_vz[sx, 9, sz] = 0.0

    if tz == 0:
        if z > 0 and in_vx:
            s_vx[sx, sy, 0] = vx[x, y, z-1]
        else:
            s_vx[sx, sy, 0] = 0.0

        if z > 0 and in_vy:
            s_vy[sx, sy, 0] = vy[x, y, z-1]
        else:
            s_vy[sx, sy, 0] = 0.0

        if z > 0 and in_vz:
            s_vz[sx, sy, 0] = vz[x, y, z-1]
        else:
            s_vz[sx, sy, 0] = 0.0

    if tz == bdz - 1:
        if z + 1 < vx.shape[2] and in_vx:
            s_vx[sx, sy, 5] = vx[x, y, z+1]
        else:
            s_vx[sx, sy, 5] = 0.0

        if z + 1 < vy.shape[2] and in_vy:
            s_vy[sx, sy, 5] = vy[x, y, z+1]
        else:
            s_vy[sx, sy, 5] = 0.0

        if z + 1 < vz.shape[2] and in_vz:
            s_vz[sx, sy, 5] = vz[x, y, z+1]
        else:
            s_vz[sx, sy, 5] = 0.0

    cuda.syncthreads()

    if 1 <= x < rho.shape[0]-1 and 1 <= y < rho.shape[1]-1 and 1 <= z < rho.shape[2]-1:

        # Load the central values for the current thread
        vx_x, vy_x, vz_x, rho_x, T_x = s_vx[sx,sy,sz], s_vy[sx,sy,sz], s_vz[sx,sy,sz], s_rho[tx, ty, tz], s_T[tx, ty, tz]

        vx_xm1, vx_xp1, vx_ym1, vx_yp1, vx_zm1, vx_zp1 = s_vx[sx-1,sy,sz], s_vx[sx+1,sy,sz], s_vx[sx,sy-1,sz], s_vx[sx,sy+1,sz], s_vx[sx,sy,sz-1], s_vx[sx,sy,sz+1]
        vy_xm1, vy_xp1, vy_ym1, vy_yp1, vy_zm1, vy_zp1 = s_vy[sx-1,sy,sz], s_vy[sx+1,sy,sz], s_vy[sx,sy-1,sz], s_vy[sx,sy+1,sz], s_vy[sx,sy,sz-1], s_vy[sx,sy,sz+1]
        vz_xm1, vz_xp1, vz_ym1, vz_yp1, vz_zm1, vz_zp1 = s_vz[sx-1,sy,sz], s_vz[sx+1,sy,sz], s_vz[sx,sy-1,sz], s_vz[sx,sy+1,sz], s_vz[sx,sy,sz-1], s_vz[sx,sy,sz+1]


        rho_xp1 = rho[x+1, y, z] if tx == bdx-1 else s_rho[tx+1, ty, tz]
        rho_yp1 = rho[x, y+1, z] if ty == bdy-1 else s_rho[tx, ty+1, tz]
        rho_zp1 = rho[x, y, z+1] if tz == bdz-1 else s_rho[tx, ty, tz+1]

        T_xp1   = T[x+1, y, z] if tx == bdx-1 else s_T[tx+1, ty, tz]
        T_yp1   = T[x, y+1, z] if ty == bdy-1 else s_T[tx, ty+1, tz]
        T_zp1   = T[x, y, z+1] if tz == bdz-1 else s_T[tx, ty, tz+1]

        # Velocity neighbors for cross-terms
        vy_xp1_ym1 = vy[x+1, y-1, z] if (tx == bdx-1 and ty == 0)     else s_vy[sx+1, sy-1, sz]
        vz_xp1_zm1 = vz[x+1, y, z-1] if (tx == bdx-1 and tz == 0)     else s_vz[sx+1, sy,   sz-1]
        vx_xm1_yp1 = vx[x-1, y+1, z] if (tx == 0     and ty == bdy-1) else s_vx[sx-1, sy+1, sz]
        vz_yp1_zm1 = vz[x, y+1, z-1] if (ty == bdy-1 and tz == 0)     else s_vz[sx,   sy+1, sz-1]
        vx_xm1_zp1 = vx[x-1, y, z+1] if (tx == 0     and tz == bdz-1) else s_vx[sx-1, sy,   sz+1]
        vy_ym1_zp1 = vy[x, y-1, z+1] if (ty == 0     and tz == bdz-1) else s_vy[sx,   sy-1, sz+1]

        # Velocity neighbors for divergence terms (x-direction)
        vx_xp2 = (s_vx[sx+2, sy, sz] if (sx+2 <= 5) else (vx[x+2, y, z] if (x+2 < vx.shape[0]) else 0.0))
        vx_xm2 = (s_vx[sx-2, sy, sz] if (sx-2 >= 0) else (vx[x-2, y, z] if (x-2 >= 0)          else 0.0))
        vy_xp1_yp1 = vy[x+1, y+1, z] if (tx >= bdx-1 or ty >= bdy-1) else s_vy[sx+1, sy+1, sz]
        vy_xm1_yp1 = vy[x-1, y+1, z] if (tx == 0 or ty >= bdy-1)     else s_vy[sx-1, sy+1, sz]
        vy_xm1_ym1 = vy[x-1, y-1, z] if (tx == 0 or ty == 0)         else s_vy[sx-1, sy-1, sz]
        vz_xm1_zp1 = vz[x-1, y, z+1] if (tx == 0 or tz >= bdz-1)     else s_vz[sx-1, sy,   sz+1]
        vz_xm1_zm1 = vz[x-1, y, z-1] if (tx == 0 or tz == 0)         else s_vz[sx-1, sy,   sz-1]
        vz_xp1_zp1 = vz[x+1, y, z+1] if (tx >= bdx-1 or tz >= bdz-1) else s_vz[sx+1, sy,   sz+1]

        # Velocity neighbors for divergence terms (y-direction)
        vx_xp1_yp1 = vx[x+1, y+1, z] if (tx >= bdx-1 or ty >= bdy-1) else s_vx[sx+1, sy+1, sz]
        vx_xm1_ym1 = vx[x-1, y-1, z] if (tx == 0 or ty == 0)         else s_vx[sx-1, sy-1, sz]
        vx_xp1_ym1 = vx[x+1, y-1, z] if (tx >= bdx-1 or ty == 0)     else s_vx[sx+1, sy-1, sz]
        vy_yp2 = (s_vy[sx, sy+2, sz] if (sy+2 <= 9) else (vy[x, y+2, z] if (y+2 < vy.shape[1]) else 0.0))
        vy_ym2 = (s_vy[sx, sy-2, sz] if (sy-2 >= 0) else (vy[x, y-2, z] if (y-2 >= 0)          else 0.0))
        vz_yp1_zp1 = vz[x, y+1, z+1] if (ty >= bdy-1 or tz >= bdz-1) else s_vz[sx,   sy+1, sz+1]
        vz_ym1_zp1 = vz[x, y-1, z+1] if (ty == 0 or tz >= bdz-1)     else s_vz[sx,   sy-1, sz+1]
        vz_ym1_zm1 = vz[x, y-1, z-1] if (ty == 0 or tz == 0)         else s_vz[sx,   sy-1, sz-1]

        # Velocity neighbors for divergence terms (z-direction)
        vx_xp1_zp1 = vx[x+1, y, z+1] if (tx >= bdx-1 or tz >= bdz-1) else s_vx[sx+1, sy,   sz+1]
        vx_xp1_zm1 = vx[x+1, y, z-1] if (tx >= bdx-1 or tz == 0)     else s_vx[sx+1, sy,   sz-1]
        vx_xm1_zm1 = vx[x-1, y, z-1] if (tx == 0     or tz == 0)     else s_vx[sx-1, sy,   sz-1]
        vy_yp1_zp1 = vy[x, y+1, z+1] if (ty >= bdy-1 or tz >= bdz-1) else s_vy[sx,   sy+1, sz+1]
        vy_yp1_zm1 = vy[x, y+1, z-1] if (ty >= bdy-1 or tz == 0)     else s_vy[sx,   sy+1, sz-1]
        vy_ym1_zm1 = vy[x, y-1, z-1] if (ty == 0     or tz == 0)     else s_vy[sx,   sy-1, sz-1]
        vz_zp2 = (s_vz[sx, sy, sz+2] if (sz+2 <= 5) else (vz[x, y, z+2] if (z+2 < vz.shape[2]) else 0.0))
        vz_zm2 = (s_vz[sx, sy, sz-2] if (sz-2 >= 0) else (vz[x, y, z-2] if (z-2 >= 0)          else 0.0))

        vx_diff, vy_diff, vz_diff = 0.0, 0.0, 0.0

        # Calculation for x-velocity
        if not Is_body_x[x, y, z]:
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
            vx_diff += B * (vx_xp1 + vx_xm1 + vx_yp1 + vx_ym1 + vx_zp1 + vx_zm1 - 6.0 * vx_x)

            div_xp1 = ((vx_xp2 - vx_x) + (vy_xp1_yp1 - vy_xp1_ym1) + (vz_xp1_zp1 - vz_xp1_zm1)) 
            div_xm1 = ((vx_x - vx_xm2) + (vy_xm1_yp1 - vy_xm1_ym1) + (vz_xm1_zp1 - vz_xm1_zm1)) 
            vx_diff += C * (div_xp1 - div_xm1) 
            
            vx_diff -= g
            
            vx[x, y, z] += dt * vx_diff

  

        # Calculation for y-velocity
        if not Is_body_y[x, y, z]:
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
            vy_diff += B * (vy_xp1 + vy_xm1 + vy_yp1 + vy_ym1 + vy_zp1 + vy_zm1 - 6.0 * vy_x)

            div_yp1 = ((vx_xp1_yp1 - vx_xm1_yp1) + (vy_yp2 - vy_x) + (vz_yp1_zp1 - vz_yp1_zm1))
            div_ym1 = ((vx_xp1_ym1 - vx_xm1_ym1) + (vy_x - vy_ym2) + (vz_ym1_zp1 - vz_ym1_zm1))
            vy_diff += C * (div_yp1 - div_ym1)
            
            vy[x, y, z] +=  dt * vy_diff
        
 
            

        # Calculation for z-velocity
        if not Is_body_z[x, y, z]:
            vx_bar = (vx_xm1 + vx_x + vx_xm1_zp1 + vx_zp1) / 4
            vy_bar = (vy_ym1 + vy_x + vy_ym1_zp1 + vy_zp1) / 4

            if vz_x > 0:
                vz_diff -= vz_x * (vz_x - vz_zm1)
            else:
                vz_diff -= vz_x * (vz_zp1 - vz_x)

            if vx_bar > 0:
                vz_diff -= vx_bar * (vz_x - vz_xm1)
            else:
                vz_diff -= vx_bar * (vz_xp1 - vz_x)

            if vy_bar > 0:
                vz_diff -= vy_bar * (vz_x - vz_ym1)
            else:
                vz_diff -= vy_bar * (vz_yp1 - vz_x)
            vz_diff /= dx 

            vz_diff -= A * (rho_zp1 * T_zp1 - rho_x * T_x) / (rho_x + rho_zp1)
            vz_diff += B * (vz_xp1 + vz_xm1 + vz_yp1 + vz_ym1 + vz_zp1 + vz_zm1 - 6.0 * vz_x)

            div_zp1 = ((vx_xp1_zp1 - vx_xm1_zp1) + (vy_yp1_zp1 - vy_ym1_zp1) + (vz_zp2 - vz_x))
            div_zm1 = ((vx_xp1_zm1 - vx_xm1_zm1) + (vy_yp1_zm1 - vy_ym1_zm1) + (vz_x - vz_zm2))
            vz_diff += C * (div_zp1 - div_zm1)

            vz[x, y, z] += dt * vz_diff

        # Exhaling
        if nostril[0] <= x+1 <= nostril[1] and nostril[3] <= z <= nostril[4] and y-1 == nostril[2]:    
            vx[x, y, z] = vx_breath

        if nostril[0] <= x <= nostril[1] and nostril[3] <= z <= nostril[4] and y == nostril[2]:  
            vy[x, y, z] = vy_breath


@cuda.jit
def update_navier_stokes_gpu(c, rho, T, vx, vy, vz, D, dc, dt_dx, Is_body, source, P, T_start, Alpha, h_newton, dT_Human, dT_Wall, max, dt, nostril, v_in, T_ex):
   
    # Calculate global thread ID and local thread ID
    x, y, z = cuda.grid(3)
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bdx, bdy, bdz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z

    sx, sy, sz = tx+1, ty+1, tz+1
    
    # Define shared memory for the block
    s_vx = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vy = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_vz = cuda.shared.array(shape=(8, 8, 8), dtype=float64)
    s_rho = cuda.shared.array(shape=(10, 10, 10), dtype=float64)
    s_T = cuda.shared.array(shape=(10, 10, 10), dtype=float64)
    s_c = cuda.shared.array(shape=(10, 10, 10), dtype=float64)

    in_rho_c_T = (x < rho.shape[0]) and (y < rho.shape[1]) and (z < rho.shape[2])
    in_vx  = (x < vx.shape[0]) and (y < vx.shape[1]) and (z < vx.shape[2])
    in_vy  = (x < vy.shape[0]) and (y < vy.shape[1]) and (z < vy.shape[2])
    in_vz  = (x < vz.shape[0]) and (y < vz.shape[1]) and (z < vz.shape[2])

    s_vx[tx, ty, tz]  = vx[x, y, z]  if in_vx else 0.0
    s_vy[tx, ty, tz]  = vy[x, y, z]  if in_vy else 0.0
    s_vz[tx, ty, tz]  = vz[x, y, z]  if in_vz else 0.0
    s_rho[sx, sy, sz] = rho[x, y, z] if in_rho_c_T else 0.0
    s_T[sx, sy, sz]   = T[x, y, z]   if in_rho_c_T else 0.0
    s_c[sx, sy, sz]   = c[x, y, z]   if in_rho_c_T else 0.0

    if tx == 0:
        if x > 0 and in_rho_c_T:
            s_rho[0, sy, sz], s_T[0, sy, sz], s_c[0, sy, sz] = rho[x-1, y, z], T[x-1, y, z], c[x-1, y, z]
        else:
            s_rho[0, sy, sz], s_T[0, sy, sz], s_c[0, sy, sz] = 0.0, 0.0, 0.0

    if tx == bdx - 1:
        if x + 1 < rho.shape[0] and in_rho_c_T:
            s_rho[9, sy, sz], s_T[9, sy, sz], s_c[9, sy, sz] = rho[x+1, y, z], T[x+1, y, z], c[x+1, y, z]
        else:
            s_rho[9, sy, sz], s_T[9, sy, sz], s_c[9, sy, sz] = 0.0, 0.0, 0.0

    if ty == 0:
        if y > 0 and in_rho_c_T:
            s_rho[sx, 0, sz], s_T[sx, 0, sz], s_c[sx, 0, sz] = rho[x, y-1, z], T[x, y-1, z], c[x, y-1, z]
        else:
            s_rho[sx, 0, sz], s_T[sx, 0, sz], s_c[sx, 0, sz] = 0.0, 0.0, 0.0

    if ty == bdy - 1:
        if y + 1 < rho.shape[1] and in_rho_c_T:
            s_rho[sx, 9, sz], s_T[sx, 9, sz], s_c[sx, 9, sz] = rho[x, y+1, z], T[x, y+1, z], c[x, y+1, z]
        else:
            s_rho[sx, 9, sz], s_T[sx, 9, sz], s_c[sx, 9, sz] = 0.0, 0.0, 0.0

    if tz == 0:
        if z > 0 and in_rho_c_T:
            s_rho[sx, sy, 0], s_T[sx, sy, 0], s_c[sx, sy, 0] = rho[x, y, z-1], T[x, y, z-1], c[x, y, z-1]
        else:
            s_rho[sx, sy, 0], s_T[sx, sy, 0], s_c[sx, sy, 0] = 0.0, 0.0, 0.0

    if tz == bdz - 1:
        if z + 1 < rho.shape[2] and in_rho_c_T:
            s_rho[sx, sy, 9], s_T[sx, sy, 9], s_c[sx, sy, 9] = rho[x, y, z+1], T[x, y, z+1], c[x, y, z+1]
        else:
            s_rho[sx, sy, 9], s_T[sx, sy, 9], s_c[sx, sy, 9] = 0.0, 0.0, 0.0

    cuda.syncthreads() 
    
    if x >= 1 and x < c.shape[0]-1 and y >= 1 and y < c.shape[1]-1 and z >= 1 and z < c.shape[2]-1:
      
        # Load the central values for the current thread
        vx_x, vy_x, vz_x, rho_x, T_x, c_x = s_vx[tx, ty, tz], s_vy[tx, ty, tz], s_vz[tx, ty, tz], s_rho[sx, sy, sz], s_T[sx, sy, sz], s_c[sx, sy, sz]
        
        vx_xm1 = vx[x-1, y, z] if tx == 0 else s_vx[tx-1, ty, tz]
        vy_ym1 = vy[x, y-1, z] if ty == 0 else s_vy[tx, ty-1, tz]
        vz_zm1 = vz[x, y, z-1] if tz == 0 else s_vz[tx, ty, tz-1]
    
        rho_x, rho_xm1, rho_xp1, rho_ym1, rho_yp1, rho_zm1, rho_zp1 = s_rho[sx,sy,sz], s_rho[sx-1,sy,sz], s_rho[sx+1,sy,sz], s_rho[sx,sy-1,sz], s_rho[sx,sy+1,sz], s_rho[sx,sy,sz-1], s_rho[sx,sy,sz+1]

        T_x, T_xm1, T_xp1, T_ym1, T_yp1, T_zm1, T_zp1 = s_T[sx,sy,sz], s_T[sx-1,sy,sz], s_T[sx+1,sy,sz], s_T[sx,sy-1,sz], s_T[sx,sy+1,sz], s_T[sx,sy,sz-1], s_T[sx,sy,sz+1]

        c_x, c_xm1, c_xp1, c_ym1, c_yp1, c_zm1, c_zp1 = s_c[sx,sy,sz], s_c[sx-1,sy,sz], s_c[sx+1,sy,sz], s_c[sx,sy-1,sz], s_c[sx,sy+1,sz], s_c[sx,sy,sz-1], s_c[sx,sy,sz+1]
    
        # Code for concentration, density, and temperature updates
        body = Is_body[x, y, z]

        if body != -1:
            c_diff, rho_diff, T_diff = 0.0, 0.0, 0.0
            
            if body == 1:
                c_diff += D * (c_xp1 + c_xm1 + c_yp1 + c_ym1 + c_zp1 + c_zm1 - 5 * c_x)
                # Person temperature
                P_cur = P * ((303.15 - T_x) / (303.15 - T_start))
                T_diff += P_cur * dT_Human / dt_dx if P_cur >= 0 else 0 #dividing by dt_dx to adjust with later calculations
            elif body == -2:
                edge = (x == 1) + (y == 1) + (z == 1) + (x == c.shape[0]-2) + (y == c.shape[1]-2) + (z == c.shape[2]-2)
                c_diff += D * (c_xp1 + c_xm1 + c_yp1 + c_ym1 + c_zp1 + c_zm1 - (6 - edge) * c_x)
                # Wall temperature
                T_diff += h_newton * dT_Wall / dt_dx * ((T_xm1 - T_x) * int((x == 1)) + (T_ym1 - T_x) * int((y == 1)) + (T_zm1 - T_x) * int((z == 1)) + (T_xp1 - T_x) * int((x == T.shape[0] - 2)) + (T_yp1 - T_x) * int((y == T.shape[1] - 2)) + (T_zp1 - T_x) * int((z == T.shape[2] - 2))) #dividing by dt_dx to adjust with later calculations
            else:
                c_diff += D * (c_xp1 + c_xm1 + c_yp1 + c_ym1 + c_zp1 + c_zm1 - 6 * c_x)
                T_diff += Alpha * (T_xp1 + T_xm1 + T_yp1 + T_ym1 + T_zp1 + T_zm1 - (6 - body) * T_x)    
            

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
        
            if nostril[0] <= x <= nostril[1] and nostril[3] <= z <= nostril[4] and y == nostril[2] + 1:
              
                if vy[source[0], source[1]-1, source[2]] > 0:
                    # Exhaling
                    c_diff += max[0, x, source[1], z] / T_ex * dt
                    rho_temp = max[1, x, source[1], z] / T_ex * dt
                    rho_diff += rho_temp
                    
                    T_diff = ((rho_x * T_x + rho_temp * 306.15) / (rho_x + rho_temp)) - T_x
                    
                # CO2 production
                #grid_num = (nostril[1] - nostril[0] + 1) * (nostril[4] - nostril[3] + 1)
                #c_diff += dc / grid_num
            
            # Update of the concentration, density and temperature fields
            c[x, y, z] = c_x + c_diff
            rho[x, y, z] = rho_x + rho_diff 
            T[x, y, z] = T_x + T_diff
        
        elif nostril[0] <= x <= nostril[1] and nostril[3] <= z <= nostril[4] and y == nostril[2] - 1: 

            grid_num = (nostril[1] - nostril[0] + 1) * (nostril[4] - nostril[3] + 1)
    
                    
            if vy[source[0], source[1] - 1, source[2]] > 0:
                # Exhaling 
                c[x, y, z] = c_x - max[0, x, source[1], z] / T_ex * dt
                rho[x, y, z] = rho_x - max[1, x, source[1], z] / T_ex * dt

            elif vy[source[0], source[1] - 1, source[2]] < 0:
                # Inhaling
                c[x, y, z] = c_x - c[x, y + 2, z] * (-v_in) * dt_dx + 3 * dc / grid_num
                rho[x, y, z] = rho_x - rho[x, y + 2, z] * (-v_in) * dt_dx
                max[1, x, y + 2, z] = rho[x, y, z]
                max[0, x, y + 2, z] = c[x, y, z]
