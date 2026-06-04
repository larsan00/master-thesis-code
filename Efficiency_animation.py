import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter


# Set FFmpeg path in rcParams
mpl.rcParams["animation.ffmpeg_path"] = r"C:\Users\Lars\miniconda3\envs\Roli\Library\bin\ffmpeg.exe"
plt.rcParams["figure.dpi"] = 300

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

        "lines.linewidth": 1.5,
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


# Plotting the data as a heatmap animation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def update_heatmap(i, heatmap, data, dt, text, text_phase, T_ex, arrow, factor):
    heatmap.set_array(data[i])
    text.set_text(f"{round((i+1)*dt,1)}s")
    
    if (i * dt % 5) < T_ex:
        phase = "Exhalation →"
        arrow.set_positions((16*factor, 70*factor), (31*factor, 70*factor))
    else:
        phase = "← Inhalation"
        arrow.set_positions((31*factor, 70*factor), (16*factor, 70*factor))

    text_phase.set_text(phase)

    return heatmap, text, text_phase


def create_animation_heatmap(data, path, animation_steps, dt, dx, vmin, vmax, T_ex):

    set_scientific_style()

    vmax, vmin, data = vmax / 1000, vmin / 1000, data / 1000
    
    fig, ax = plt.subplots(figsize=(14,5))

    heatmap = ax.imshow(
        np.zeros_like(data[0]),
        cmap='coolwarm',
        norm=Normalize(vmin=vmin, vmax=vmax),
        animated=True,
        origin='lower'
    )

    img = plt.imread("Bild1.png")

    factor = 0.01 / dx
    ax.imshow(img, extent=[12*factor, 36*factor, 38*factor, 66*factor], origin='upper', zorder=3, alpha=1)

    arrow = FancyArrowPatch(
        (16*factor, 70*factor), (31*factor, 70*factor),
        arrowstyle='->',
        mutation_scale=30,
        color='red',
        linewidth=3
    )

    ax.add_patch(arrow)
    
    ax.set_xlim(0, data[0].shape[1])
    ax.set_ylim(0, data[0].shape[0])

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(heatmap, ax=ax, pad=0.01)

    text = ax.text(0.84, 1.03, '0s', transform=ax.transAxes)
    text_phase = ax.text(0.02, 1.03, 'Exhalation', transform=ax.transAxes, fontsize=14, color='red')

    anim = FuncAnimation(
        fig,
        update_heatmap,
        frames=animation_steps,
        fargs=(heatmap, data, dt, text, text_phase, T_ex, arrow, factor),
        interval=100
    )

    anim.save(path, writer=FFMpegWriter(fps=20), dpi=250)
    plt.close(fig)


# Plotting one frame of the data as a heatmap
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_picture(data, timestep, vmin, vmax):

    set_scientific_style()

    vmax, vmin, data = vmax / 1000, vmin / 1000, data / 1000

    fig, ax = plt.subplots(figsize=(14,5))

    heatmap = ax.imshow(data[timestep], cmap='coolwarm', norm=Normalize(vmin=vmin, vmax=vmax), origin='lower')

    img = plt.imread("Bild1.png")

    ax.imshow(img, extent=[3, 33, 31, 67], origin='upper', zorder=3, alpha=1)

    ax.set_xlim(0, data[0].shape[1])
    ax.set_ylim(0, data[0].shape[0])

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(heatmap, ax=ax, pad=0.01)
    cbar.set_label(r"$c\ [10^3\ \mathrm{ppm}]$", fontsize=16)
    ax.text(0.15, 0.05, r"($\mathbf{b}$)", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)
    ax.text(0.95, 0.9, r"$0.5\ s$", transform=ax.transAxes, ha="right", va="bottom", fontsize=20)

    fig.savefig("snap_Tex_05.pdf", bbox_inches="tight")
    fig.savefig("snap_Tex_05.svg", bbox_inches="tight")

    plt.close(fig)

# Plotting one frame of the concentration as a heatmap
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#dx = 0.01
#c = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/c_v1_T20_D.npy")
#create_picture(c[:,:,:, int(round((0.2)/dx,0))], 100, vmin=455, vmax=20000)

#c = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/Variable Atemperiode/c_v1_T20_D_Tex_05.npy")
#create_picture(c[:,:,:, c.shape[3]//2], 20, vmin=455, vmax=20000)

#c = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/Variable Atemperiode/c_v1_T20_D_Tex_45.npy")
#create_picture(c[:,:,:, c.shape[3]//2], 180, vmin=455, vmax=20000)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Creating the velocity snapshots
def create_velocity_snapshots(vx):

    set_scientific_style()

    Nt = vx.shape[0]
    times = [1, 2, 3, 4, 5]
    indices = [int((t / 5.0) * (Nt - 1)) for t in times]

    vx_plot = vx[:, :, :, vx.shape[3] // 2] * 100

    vmin = np.min(vx_plot) / 20
    vmax = np.max(vx_plot) / 20

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)

    img = plt.imread("Bild1.png")

    for ax, idx, t in zip(axes, indices, times):

        heatmap = ax.imshow(vx_plot[idx], cmap="coolwarm",
                            norm=Normalize(vmin=vmin, vmax=vmax),
                            origin="lower")

        ax.imshow(img, extent=[3, 33, 31, 67], origin="upper",
                  zorder=3, alpha=1)

        ax.set_xlim(0, vx_plot.shape[2])
        ax.set_ylim(0, vx_plot.shape[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.95, 0.95, rf"${t}\,\mathrm{{s}}$",
                transform=ax.transAxes,
                ha="right", va="top")

    cbar = fig.colorbar(heatmap, ax=axes, shrink=0.85, pad=0.005)

    cbar.set_ticks(np.linspace(-3, 3, 7))
    cbar.set_label(r"$v_x\ [\mathrm{cm/s}]$")

    fig.savefig("vx_snapshots.pdf", bbox_inches="tight")
    fig.savefig("vx_snapshots.svg", bbox_inches="tight")

    plt.close(fig)

# Creating the velocity snapshots
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#vx = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/vy_v1_T20_D.npy")
#create_velocity_snapshots(vx)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Creating the breathing model plot
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_breathing():

    set_scientific_style()
    fig, ax1 = plt.subplots(figsize=(4, 3))

    petrol = "#006d77"
    bordeaux = "#8d0801"

    x = np.linspace(0, 5, 500)

    v = np.piecewise(
        x,
        [x < 2.5, x >= 2.5],
        [0.7, -0.7]
    )

    V = np.piecewise(
        x,
        [x < 2.5, x >= 2.5],
        [
            lambda x: 0.7 - (0.7 / 2.5) * x,
            lambda x: (0.7 / 2.5) * (x - 2.5)
        ]
    )

    # Background regions
    ax1.axvspan(0, 2.5, color=bordeaux, alpha=0.08)
    ax1.axvspan(2.5, 5, color=petrol, alpha=0.08)

    # Labels
    ax1.text(1.25, 0.5, "Exhale", color=bordeaux, ha="center", fontsize=8.5, alpha=0.8)
    ax1.text(3.75, 0.5, "Inhale", color=petrol, ha="center", fontsize=8.5, alpha=0.8)

    # Velocity plot
    ax1.plot(x, v, color=petrol, linewidth=2)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("air velocity [m/s]", color=petrol)
    ax1.tick_params(axis='y', labelcolor=petrol)
    ax1.set_xlim(0, 5)

    # Volume plot
    ax2 = ax1.twinx()
    ax2.plot(x, V, color=bordeaux, linewidth=2)
    ax2.set_ylabel("lung volume [L]", color=bordeaux)
    ax2.tick_params(axis='y', labelcolor=bordeaux)

    ax1.text(0.98, 0.05, r"($\mathbf{b}$)", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig("breathing.pdf", bbox_inches="tight")
    fig.savefig("breathing.svg", bbox_inches="tight")
    plt.close(fig)

#Breathing Model
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#plot_breathing()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Plotting the CO2 concentration in the lung over time
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lung(c_data, rho_data, nostril, time, steps, dx, D_0, V_air):

    set_scientific_style()

    data_x = np.arange(0, time + time/steps, time/steps)
    fig, ax1 = plt.subplots()

    c_y = c_data[:, int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1].mean(axis=(1,2))
    rho_y = rho_data[:, int(nostril[0]):int(nostril[1]+1), int(nostril[2])-1, int(nostril[3]):int(nostril[4])+1].mean(axis=(1,2))

    m_co2 = c_y / ((V_air * 0.001) / (dx * D_0**2)) / 1000
    c_ppm = ((m_co2 / m_co2[0]) / (rho_y / rho_y[0]))
    c_ppm = c_ppm / c_ppm[1]

    ax1.plot(data_x, c_ppm / c_ppm[0], linewidth=3, label=r'$c(t)$')
    ax1.plot(data_x, m_co2 / m_co2[0], linestyle='--', linewidth=3, label=r'$m_{\mathrm{CO_2}}(t)$')
    ax1.plot(data_x, rho_y / rho_y[0], linestyle=':', linewidth=3, label=r'$\rho(t)$')

    ax1.scatter(data_x[0], c_ppm[0], color='black', zorder=3)
    ax1.scatter(data_x[-1], c_ppm[-1], color='black', zorder=3)

    ax1.annotate(r'$C_{\mathrm{init}}$', (data_x[0], c_ppm[0]), xytext=(data_x[0] + 0.10*time, c_ppm[0] * 0.92), arrowprops=dict(arrowstyle="->"), fontsize=18, fontweight='bold')
    ax1.annotate(r'$C_{\mathrm{final}}$', (data_x[-1], c_ppm[-1]), xytext=(data_x[-1] - 0.2*time, c_ppm[-1] * 2.5), arrowprops=dict(arrowstyle="->"), fontsize=18, fontweight='bold')
    #ax1.annotate(r'$C_{\mathrm{final}}$', (data_x[-1], c_ppm[-1]), xytext=(data_x[-1] - 0.1*time, c_ppm[-1] * 0.7), arrowprops=dict(arrowstyle="->"), fontsize=18, fontweight='bold')

    ax1.set_xlabel('t [s]')
    ax1.set_ylabel(r'normalized quantities')
    ax1.legend()

    ax1.text(0.90, 0.77, r"($\mathbf{a}$)", transform=ax1.transAxes, ha="left", va="bottom", fontsize=20)

    plt.savefig("lung.svg", bbox_inches='tight')
    plt.savefig("lung.pdf", bbox_inches='tight')

    plt.close()

# Plotting the CO2 concentration in the lung over time
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#dx, D_0 = 0.01, 0.02
#c = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/c_v1_T20_D.npy")
#rho = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/rho_v1_T20_D.npy")

#c = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/c_v1000_T20_D.npy")
#rho = np.load("D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Breathing Efficiency/Sim Results/20 degrees + dif/rho_v1000_T20_D.npy")

#h_source, l_source, b_source = int(round(0.5/dx,0)), int(round(0.3/dx,0)), int(round((0.2)/dx,0))
#length = int(round(0.02 / dx))
#nostril = np.array([h_source+1-length, h_source, l_source-1, b_source+1-np.ceil(length/2), b_source+int(length/2)]) 

#lung(c, rho, nostril, 5, 200, dx, D_0, 0.7)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the efficiency of the breathing process as a function of the Reynolds number
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_efficiency():
  
    set_scientific_style()

    x = np.array([9270, 927.16, 92.72, 18.54, 9.28, 1.85, 0.926, 0.0926])

    nCO2_D_20 = np.array([276901, 259274, 217286, 312747, 595217, 2644873, 3862533, 4648796])
    nCO2_D_33 = np.array([243778, 204979, 202583, 342614, 1481961, 4501796, 4682756, 4749400])
    nCO2_33 = np.array([182647, 169846, 191326, 330960, 1455866, 4734901, 4954725, 5036321])
    nCO2_20 = np.array([242187, 231932, 206007, 302578, 572167, 2651724, 3997505, 4925045])
    
    nCO2_start = 7000000

    nCO2_norm_D_20 = nCO2_D_20 / nCO2_start
    nCO2_norm_D_33 = nCO2_D_33 / nCO2_start
    nC02_33_norm = nCO2_33 / nCO2_start
    nCO2_20_norm = nCO2_20 / nCO2_start

    fig, ax = plt.subplots()

    ax.plot(x, nCO2_norm_D_20, 'o-', label=r"$T_{AIR}$ = 20°C + Diff.")
    ax.plot(x, nCO2_norm_D_33, 'o-', label=r"$T_{AIR}$ = 33°C + Diff.")
    ax.plot(x, nCO2_20_norm, 'o-', label=r"$T_{AIR}$ = 20°C")
    ax.plot(x, nC02_33_norm, 'o-', label=r"$T_{AIR}$ = 33°C")

    ax.text(0.12, 0.35, "Stokes", fontsize=12, color="darkgreen", fontweight="bold")
    ax.text(0.12, 0.3, "regime", fontsize=12, color="darkgreen", fontweight="bold")
    ax.text(20, 0.35, "Inertia-dominated regime", fontsize=12, color="darkred", fontweight="bold")

    ax.axvspan(0, 1, color='green', alpha=0.2)
    ax.axvspan(1, max(x)*1.2, color='red', alpha=0.2)

    i = 1
    ax.annotate(
    "Correct air viscosity",
    (x[i], nCO2_norm_D_20[i]),
    xytext=(x[i]*0.1, nCO2_norm_D_20[i]*3),
    arrowprops=dict(
        arrowstyle="->",
        mutation_scale=10,
        lw=2
    )
    )

    ax.set_xscale("log")
    ax.set_xlim(0.08, max(x)*1.2)

    ax.set_xlabel("Re Number")
    ax.set_ylabel(r"$\Phi$")
    plt.legend()
    plt.savefig("efficiency_plot.svg", bbox_inches='tight')
    plt.savefig("efficiency_plot.pdf", bbox_inches='tight')

# Plotting the efficiency of the breathing process as a function of the Reynolds number
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_efficiency()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the efficiency of the breathing process as a function of the exhalation time
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_efficiency2():
  
    set_scientific_style()

    x = np.array([0.25, 0.5, 1, 2, 2.5, 3, 4, 4.5, 4.75])

    nCO2_D_20 = np.array([122441, 142361, 179727, 238869, 259274, 279984 , 353147, 435351, 509254])

    nCO2_D_33 = np.array([105681, 118481, 140178, 181804, 204979, 231511 , 306802, 373172, 429138])
    
    nCO2_start = 7000000

    nCO2_norm_D_20 = nCO2_D_20 / nCO2_start

    fig, ax = plt.subplots()

    ax.plot(x, nCO2_norm_D_20, 'o-', label=r"$T_{AIR}$ = 20°C + Diffusion")
    ax.plot(x, nCO2_D_33 / nCO2_start, 'o-', label=r"$T_{AIR}$ = 33°C + Diffusion")

    ax.set_xlabel(r"$t_{ex}$ [s]")
    ax.set_ylabel(r"$\Phi$")
    plt.legend()
    plt.savefig("efficiency_plot2.svg", bbox_inches='tight')
    plt.savefig("efficiency_plot2.pdf", bbox_inches='tight')

# Plotting the efficiency of the breathing process as a function of the exhalation time
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_efficiency2()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the efficiency of the breathing process as a function of the spatial resolution for a Reynolds number of 464
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_efficiency_dx():

    set_scientific_style()

    dx = np.array([0.04, 0.02, 0.0133333333333333, 0.01, 0.008, 0.0066666666666667, 0.005])
    efficiency = np.array([0.10466142112900751, 0.10082477844595152, 0.08955263224261513, 0.08080064613700776, 0.07518235421757728, 0.07130061412177089, 0.06626147580896881])

    dx_fit_data = dx[-4:]
    eff_fit_data = efficiency[-4:]
    coeff = np.polyfit(dx_fit_data, eff_fit_data, 1)
    phi_0 = np.polyval(coeff, 0.0)

    dx_fit = np.linspace(0.0, dx.max(), 1000)
    eff_fit = np.polyval(coeff, dx_fit)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(dx * 100, efficiency, 'o-', lw=2, ms=7, label="Simulation", zorder=3)
    ax.plot(dx_fit * 100, eff_fit, '--', lw=2, label=fr"Linear Extrapolation" + "\n" + fr"($\Phi_0={phi_0:.3f}$)", zorder=2)
    ax.axvline(x=1.0, linestyle="--", linewidth=2, color="red")
    ax.text(1.0, 0.135, "used $\Delta x$", rotation=90, va="bottom", ha="right", color="red")

    ax.set_xlabel(r"$\Delta x$ [cm]")
    ax.set_ylabel(r"$\Phi$")
    ax.set_title(r"$\Delta x$ Convergence (Re = 464)")
    ax.invert_xaxis()
    ax.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig("Efficiency_vs_dx.svg", bbox_inches="tight")
    plt.savefig("Efficiency_vs_dx.pdf", bbox_inches="tight")
    plt.close()

# Plotting the efficiency of the breathing process as a function of the spatial resolution for a Reynolds number of 464
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_efficiency_dx()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the efficiency of the breathing process as a function of the spatial resolution for a Reynolds number of 0.46
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_efficiency_dx2():

    set_scientific_style()

    dx = np.array([0.04, 0.02, 0.0133333333333333, 0.01, 0.008, 0.0066666666666667, 0.005])
    efficiency = np.array([0.5041710976057373, 0.5568323507448822, 0.5812181716026748, 0.5952653212812442, 0.6047388849067666, 0.6113392710651025, 0.6205302167667936])

    dx_fit_data = dx[-4:]
    eff_fit_data = efficiency[-4:]
    coeff = np.polyfit(dx_fit_data, eff_fit_data, 1)
    phi_0 = np.polyval(coeff, 0.0)

    dx_fit = np.linspace(0.0, dx.max(), 1000)
    eff_fit = np.polyval(coeff, dx_fit)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(dx * 100, efficiency, 'o-', lw=2, ms=7, label="Simulation", zorder=3)
    ax.plot(dx_fit * 100, eff_fit, '--', lw=2, label=fr"Linear Extrapolation" + "\n" + fr"($\Phi_0={phi_0:.3f}$)", zorder=2)
    ax.axvline(x=1.0, linestyle="--", linewidth=2, color="red")
    ax.text(1.0, ax.get_ylim()[0], "used $\Delta x$", rotation=90, va="bottom", ha="right", color="red")

    ax.set_xlabel(r"$\Delta x$ [cm]")
    ax.set_ylabel(r"$\Phi$")
    ax.set_title(r"$\Delta x$ Convergence (Re = 0.46)")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()

    plt.savefig("Efficiency_vs_dx2.svg", bbox_inches="tight")
    plt.savefig("Efficiency_vs_dx2.pdf", bbox_inches="tight")
    plt.close()

# Plotting the efficiency of the breathing process as a function of the spatial resolution for a Reynolds number of 0.46
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_efficiency_dx2()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting the efficiency of the breathing process as a function of the time step size
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_efficiency_dt():

    set_scientific_style()

    dt = np.array([0.012, 0.011, 0.010, 0.009, 0.008, 0.007, 0.006, 0.005])
    dt = dt / 0.01

    efficiency = np.array([0.0807947819985333, 0.08079155708886179, 0.08080064573882496, 0.08079253616390943, 0.08079916796885901, 0.08079531469674408, 0.08079332407140835, 0.08079722626881872])

    mean_phi = np.mean(efficiency)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(dt, efficiency, 'o-', lw=2, ms=7, label="Simulation", zorder=3)

    ax.axhline(y=mean_phi, linestyle="--", linewidth=2, label=fr"Mean value" + "\n" + fr"($\Phi={mean_phi:.3f}$)", zorder=2)

    ax.axvline(x=1, linestyle="--", linewidth=2, color="red")
    ax.text(1, ax.get_ylim()[0], r"used $\Delta t$", rotation=90, va="bottom", ha="right", color="red")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_useOffset(True)  # Offset wieder erlauben

    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r"$\Delta t$ [$10^{-2}\,\mathrm{ms}$]")

    ax.set_ylabel(r"$\Phi$")

    ax.set_title(r"$\Delta t$ Convergence (Re = 0.46)")
    
    ax.invert_xaxis()

    ax.legend()

    plt.tight_layout()

    plt.savefig("Efficiency_vs_dt.svg", bbox_inches="tight")

    plt.savefig("Efficiency_vs_dt.pdf", bbox_inches="tight")

    plt.close()

# Plotting the efficiency of the breathing process as a function of the time step size
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_efficiency_dt()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------