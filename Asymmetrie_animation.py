import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Arc

# Set FFmpeg path in rcParams
mpl.rcParams["animation.ffmpeg_path"] = r"C:\Users\Lars\miniconda3\envs\Roli\Library\bin\ffmpeg.exe"
mpl.rcParams.dpi = 300

# Set a consistent scientific style for all plots
def set_scientific_style():
    plt.rcParams.update({

        "font.size": 18,                
        "axes.labelsize": 18,           
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "font.family": "serif",         
        "mathtext.fontset": "dejavuserif",

        "lines.linewidth": 3.0,
        "lines.markersize": 6,

        "axes.linewidth": 1.2,         
        "axes.spines.top": True,
        "axes.spines.right": True,

        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,

        "legend.frameon": False,     
        "legend.handlelength": 2.0,

        "axes.prop_cycle": plt.cycler(color=[
            "#1f77b4", 
            "#ff7f0e",  
            "#2ca02c",  
            "#d62728",  
            "#9467bd",  
            "#8c564b", 
        ]),
    })


# Nonlinear corretion of the Landau-Squire jet
def forces(rho, v):

    set_scientific_style()

    mu = rho*v
    theta = np.linspace(0, np.pi/2-1e-4, 90)

    fig, ax = plt.subplots(figsize=(7,5))

    for F in [1e-6, 1e-8, 1e-10]:
        plt.plot(
            theta*180/np.pi,
            (F/(16*np.pi*rho*mu))*(np.sin(theta)**2/np.cos(theta)),
            linewidth=2,
            label=fr"$F=10^{{{int(np.log10(F))}}}\,\mathrm{{N}}$"
        )

    ax.text(0.98, 0.02, r"($\mathbf{a}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)

    plt.xlabel(r"$\theta\,[^\circ]$")
    plt.ylabel(r"$|v_r^{(2)}/v_r^{(1)}|$")
    plt.xlim(0,90)
    plt.yscale('log')
    plt.legend(frameon=False, loc='upper left')
    plt.savefig('forces.pdf', bbox_inches='tight')
    plt.savefig('forces.svg', bbox_inches='tight')

# Nonlinear correction of the Landau-Squire jet
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#forces(1.18, 1.51e-5)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the velocity profiles along the centerline for the Oseen tensor, Landau-Squire jet and the simulation results
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_velocity_profiles(arrays, x_length=1.0, point_force_position=0.502, labels=None):
    
    set_scientific_style()

    max_val = max(np.max(np.abs(arr)) for arr in arrays)

    if max_val < 1e-3:
        factor = 1000
        ylabel = r'$v_x\ [mm/s]$'

    elif max_val < 1e-1:
        factor = 100
        ylabel = r'$v_x\ [cm/s]$'

    else:
        factor = 1
        ylabel = r'$v_x\ [m/s]$'

    fig, ax = plt.subplots(figsize=(7,5))

    for i, arr in enumerate(arrays):
        x = np.linspace(0, x_length, len(arr))
        if labels:
            plt.plot(x, arr * factor, label=labels[i])
        else:
            plt.plot(x, arr * factor)

    plt.axvline(point_force_position, color='red', linestyle='--', linewidth=2, label='Point Force')

    ax.text(0.98, 0.08, r"($\mathbf{b}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)

    plt.xlabel('Position along the centerline [m]')
    plt.ylabel(ylabel)
    plt.xlim(0, 1)

    if labels:
        plt.legend(loc='upper left')

    plt.savefig('F10.pdf', bbox_inches='tight')
    plt.savefig('F10.svg', bbox_inches='tight')

# Centerline velocity profiles for the Oseen tensor, Landau-Squire jet and the simulation results
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vy_sim = np.load('vy_sim_dx_0.01.npy')
#vy_lan = np.load('vy_landau_dx_0.01.npy')
#vy_oseen = np.load('vy_oseen_dx_0.01.npy')
#plot_velocity_profiles([vy_oseen[vy_oseen.shape[0]//2,:], vy_lan[vy_lan.shape[0]//2,:], vy_sim[-1,vy_sim.shape[1]//2,vy_sim.shape[2]//4:3*vy_sim.shape[2]//4]], labels=["Oseen-Tensor", "Landau-\nSquire Jet", "Simulation"])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the velocity field as a heatmap animation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def update_heatmap(i, heatmap, data, dt, text):
    heatmap.set_array(data[i])
    text.set_text(f"{round((i+1)*dt,1)}s")
    return heatmap, text


def create_animation_heatmap(data, path, animation_steps, dt, vmin, vmax):

    set_scientific_style()

    fig, ax = plt.subplots(figsize=(14,5))

    max_val = np.max(np.abs(data))

    if max_val < 1e-3:
        factor = 1e6
        ylabel = r'$v_x\ [\mu m/s]$'

    elif max_val < 1e-2:
        factor = 1e3
        ylabel = r'$v_x\ [mm/s]$'

    elif max_val < 1e0:
        factor = 1e2
        ylabel = r'$v_x\ [cm/s]$'

    else:
        factor = 1
        ylabel = r'$v_x\ [m/s]$'

    data = data * factor
    vmin = vmin * factor
    vmax = vmax * factor

    heatmap = ax.imshow(
        np.zeros_like(data[0]),
        cmap='coolwarm',
        norm=Normalize(vmin=vmin, vmax=vmax),
        animated=True,
        origin='lower'
    )

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Simulation")
    fig.tight_layout()


    cbar = fig.colorbar(heatmap, ax=ax, pad=0.01)
    cbar.set_label(ylabel, fontsize=16)

    text = ax.text(0.84, 1.03, '0s', transform=ax.transAxes)

    anim = FuncAnimation(
        fig,
        update_heatmap,
        frames=animation_steps,
        fargs=(heatmap, data, dt, text),
        interval=100
    )

    anim.save(path, writer=FFMpegWriter(fps=20))
    plt.close(fig)

# Plotting the velocity field as a heatmap animation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vy_sim = np.load('vy_sim_dx_0.01.npy')
#vy_lan = np.load('vy_landau_dx_0.01.npy')
#vy_oseen = np.load('vy_oseen_dx_0.01.npy')
#v0 = np.max(vy_lan)

#create_animation_heatmap(vy_sim[:,:, vy_sim.shape[2]//4:3*vy_sim.shape[2]//4], "sim.mp4", 200, 5, 0, v0/20)   

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting one frame of the velocity field as a heatmap
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_picture(data, vmin, vmax):

    set_scientific_style()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.set_aspect('equal')

    max_val = np.max(np.abs(data))

    if max_val < 1e-3:
        factor = 1e6
        ylabel = r'$v_x\ [\mu m/s]$'

    elif max_val < 1e-2:
        factor = 1e3
        ylabel = r'$v_x\ [mm/s]$'

    elif max_val < 1e0:
        factor = 1e2
        ylabel = r'$v_x\ [cm/s]$'

    else:
        factor = 1
        ylabel = r'$v_x\ [m/s]$'

    data = data * factor
    vmin = vmin * factor
    vmax = vmax * factor

    im = ax.imshow(
        data,
        cmap='coolwarm',
        norm=Normalize(vmin=vmin, vmax=vmax),
        origin='lower'
    )

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5), xycoords='axes fraction', arrowprops=dict(color='black', arrowstyle='-|>', lw=2.5, mutation_scale=20))

    ax.plot([0.0, 1.0], [0.5, 0.5], transform=ax.transAxes, color='red', linestyle='--', linewidth=2)
    ax.text(0.02, 0.52, 'Center line', transform=ax.transAxes, color='red', fontsize=14, va='bottom', ha='left')
    ax.plot([0.5, 0.5], [0.0, 1.0], transform=ax.transAxes, color='green', linestyle='--', linewidth=2)
    ax.text(0.52, 0.95, 'Symmetry axis', transform=ax.transAxes, color='green', fontsize=14, rotation=90, va='top', ha='left')

    theta_deg = 30
    x0, y0 = 0.5, 0.5
    x_end = 1.0
    y_end = y0 + np.tan(np.deg2rad(theta_deg)) * (x_end - x0)

    ax.plot([x0, x_end], [y0, y_end], transform=ax.transAxes, color='black', linestyle='--', linewidth=2)

    arc_radius = 0.5
    arc = Arc((x0, y0), width=arc_radius, height=arc_radius, angle=0, theta1=0, theta2=theta_deg, transform=ax.transAxes, color='black', linewidth=2)
    ax.add_patch(arc)

    theta_text_r = 0.18
    theta_x = x0 + theta_text_r * np.cos(np.deg2rad(theta_deg / 2))
    theta_y = y0 + theta_text_r * np.sin(np.deg2rad(theta_deg / 2))
    ax.text(theta_x, theta_y, r'$\theta$', transform=ax.transAxes, fontsize=18, color='black')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(ylabel, fontsize=16)

    ax.text(0.98, 0.05, r"($\mathbf{a}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=18)
    ax.set_title("Oseen Tensor")
    ax.set_title("Landau-Squire Jet")
    ax.set_title("Simulation")

    fig.savefig("oseen_F10.pdf", bbox_inches="tight")
    fig.savefig("oseen_F10.svg", bbox_inches="tight")
    plt.close(fig)

# Plotting one frame of the velocity field as a heatmap
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vy_sim = np.load('vy_sim_dx_0.01.npy')
#vy_lan = np.load('vy_landau_dx_0.01.npy')
#vy_oseen = np.load('vy_oseen_dx_0.01.npy')
#v0 = np.max(vy_lan)
#create_picture(vy_sim[-1,:,vy_sim.shape[2]//4:3*vy_sim.shape[2]//4], 0, v0/20)
#create_picture(vy_lan[:,:], 0, v0/20)
#create_picture(vy_oseen[:,:], 0, v0/20)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the asymmetry measure as a function of the Reynolds number
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_asymmetry(vy_list, Re_list):

    set_scientific_style()

    def asymmetry_measure(vy):
        vy_mirror = vy[:, ::-1]

        numerator = np.sum(np.abs(vy - vy_mirror))
        denominator = 2 * np.sum(np.abs(vy))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    epsilons = []

    for vy_sim in vy_list:
        vy_cut = vy_sim[-1, :, vy_sim.shape[2]//4 : 3*vy_sim.shape[2]//4]
        epsilon = asymmetry_measure(vy_cut)
        epsilons.append(epsilon)

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(Re_list, epsilons, marker='o')

    ax.axvspan(0, 1, color='green', alpha=0.2)
    ax.axvspan(1, max(Re_list)*1.2, color='red', alpha=0.2)

    ax.text(0.12, 0.60, "Stokes", fontsize=16, color="darkgreen", fontweight="bold")
    ax.text(0.12, 0.55, "regime", fontsize=16, color="darkgreen", fontweight="bold")
    ax.text(15, 0.60, "Inertia-dominated", fontsize=16, color="darkred", fontweight="bold")
    ax.text(15, 0.55, "regime", fontsize=16, color="darkred", fontweight="bold")
    ax.text(0.98, 0.05, r"($\mathbf{a}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)

    plt.xscale('log')

    plt.xlabel('Re Number')
    plt.ylabel(r'$\epsilon$')
    plt.xlim(6*1e-2, max(Re_list)*1.1)

    fig.savefig("epsilon.pdf", bbox_inches="tight")
    fig.savefig("epsilon.svg", bbox_inches="tight")
    plt.close(fig)

# Plotting the asymmetry measure as a function of the Reynolds number
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vy_sim_5 = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Asymmetrie/Sim Results/dx = 0.01m/F=4 10^-5N/vy_sim_dx_0.01.npy')
#vy_sim_6 = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Asymmetrie/Sim Results/dx = 0.01m/F=1.15 10^-6N/vy_sim_dx_0.01.npy')
#vy_sim_8 = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Asymmetrie/Sim Results/dx = 0.01m/F=1.15 10^-8N/vy_sim_dx_0.01.npy')
#vy_sim_9 = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Asymmetrie/Sim Results/dx = 0.01m/F=1.15 10^-9N/vy_sim_dx_0.01.npy')
#vy_sim_10 = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Asymmetrie/Sim Results/dx = 0.01m/F=1.15 10^-10N/vy_sim_dx_0.01.npy')

#plot_asymmetry([vy_sim_5, vy_sim_6, vy_sim_8, vy_sim_9, vy_sim_10], [927, 23.49, 7.83, 1.17, 0.07])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the relaxation of the velocity field towards the stationary state
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_relaxation(v_field, dt):

    set_scientific_style()

    t = np.arange(v_field.shape[0]) * dt
    v_sum = np.sum(np.abs(v_field), axis=tuple(range(1, v_field.ndim)))
    v_inf = v_sum[-1]
    v_sum = v_sum / v_inf
    v_inf = 1.0
    idx_relax = np.where(np.abs(v_sum - v_inf) / v_inf < 0.01)[0][0]
    t_relax = t[idx_relax]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t, v_sum, lw=2, label="Simulation")

    ax.axhline(y=v_inf, linestyle="--", linewidth=2, color="black", label=r"$v_\infty$")
    ax.axvline(x=t_relax, linestyle="--", linewidth=2, color="red")

    ax.text(t_relax, 0, fr"$t_{{\mathrm{{relax}}}}={t_relax:.1f}\,\mathrm{{s}}$", rotation=90, va="bottom", ha="right", color="red")
    ax.text(0.98, 0.05, r"($\mathbf{a}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$v^*$")
    ax.legend()
    plt.xlim(0, t[-1])
    plt.ylim(0, v_sum[-1]*1.1)

    plt.tight_layout()
    plt.savefig("Relaxation_F06.svg", bbox_inches="tight")
    plt.savefig("Relaxation_F06.pdf", bbox_inches="tight")
    plt.close()

# Plotting the relaxation of the velocity field towards the stationary state
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vy_sim = np.load('vy_sim_dx_0.01.npy')
#plot_relaxation(vy_sim, 1000/200)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------