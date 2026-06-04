import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import ScalarFormatter

# Set FFmpeg path in rcParams
mpl.rcParams["animation.ffmpeg_path"] = r"C:\Users\Lars\miniconda3\envs\Roli\Library\bin\ffmpeg.exe"
mpl.rcParams.dpi = 300

# Set a consistent scientific style for all plots
def set_scientific_style():
    plt.rcParams.update({

        "font.size": 16,                
        "axes.labelsize": 16,           
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
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


# Check 1: Fourier's law validation
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_h_validation(Ts, hs, t_idxs):

    set_scientific_style()

    x = np.linspace(0, 1, Ts[0].shape[2])
    plt.figure(figsize=(7,5))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    styles = ['-', ':', '--', '-.']
    styles = ['-', '-', '-', '-']
    halo_colors = ['red', 'blue', 'green', 'purple', 'orange']

    T0 = Ts[0][t_idxs[0], Ts[0].shape[1]//2, :, Ts[0].shape[3]//2]
    T_theo = T0[0] + (T0[-1] - T0[0]) * x

    plt.plot(x, T_theo, 'k--', label='Theory')

    for j, (t_idx, style, halo_color) in enumerate(zip(t_idxs, styles, halo_colors)):

        for i, (T, h) in enumerate(zip(Ts, hs)):

            y = T[t_idx, T.shape[1]//2, :, T.shape[3]//2]

            if i == 1: plt.plot(x, y, '-', linewidth=16, color=halo_color, alpha=0.15, solid_capstyle='round', zorder=1)

            plt.plot(x, y, linestyle=style, color=colors[i], label=fr"$h={h}\,\frac{{W}}{{m^2K}}$" if j == 0 else None, zorder=2)

    plt.text(0.05, 293.8, f"t={t_idxs[0]*100} s", color='red', fontsize=18, alpha=0.5)
    plt.text(0.48, 295.8, f"t={t_idxs[1]*100} s", color='blue', fontsize=18, alpha=0.5)

    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.legend()

    plt.savefig("Fourier.pdf", bbox_inches='tight')
    plt.savefig("Fourier.svg", bbox_inches='tight')
    plt.close()

#Check 1: Fourier law
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plot_h_validation([np.load(f'D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Fourier/T_h_{h}_sim_dx_0.01.npy') for h in [1,5,50]], [1,5,50], [1,100])

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Check 2: barometric formula
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_baro(rho, mode):
    set_scientific_style()

    n_t, Nx, Ny, Nz = rho.shape
    times = [0, n_t//5, n_t//2]

    y, z = Ny//2, Nz//2
    line = lambda t: rho[t, 1:-1, y, z]

    x = np.linspace(0, 1.0, Nx-2)

    gamma = 2
    H = 287.1 * (273.15 + 20) / 9.81

    rho0, x_ref = 1.18, 0.5
    dt = 10 / 200

    fig, ax = plt.subplots(figsize=(7,5))

    for t in times:
        ax.plot(x, line(t), label=f"t = {t*dt:.2f} s")

    if mode == "adiabatic":

        gammas = [1.4, 2.0]
        styles = ["r--", "k--"]

        for g, style in zip(gammas, styles):
            factor = 1.0 - ((g - 1.0) / g) * ((x - x_ref) / H)
            rho_theo = rho0 * factor ** (1.0 / (g - 1.0))

            if g == 1.4:
                label = rf"Adiabatic ($\gamma={g}$)"
            else:
                label = rf"Polytropic ($\gamma={g}$)"

            ax.plot(x, rho_theo, style, lw=2, label=label)

        ax.set_title("Adiabatic")

        ax.text(0.02, 0.02, r"($\mathbf{b}$)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=18)

    elif mode == "isothermal":
        rho_iso = rho0 * np.exp(-(x - x_ref) / H)
        ax.plot(x, rho_iso, "k-.", lw=2, label="Isothermal")
        ax.set_title("Isothermal")

        ax.text(0.02, 0.02, r"($\mathbf{a}$)", transform=ax.transAxes, ha="left", va="bottom", fontsize=18)

    else:
        raise ValueError("mode must be 'isothermal' or 'adiabatic'")

    plt.xlabel("h [m]")
    plt.ylabel(r"$\rho\ [kg/m^3]$")
    plt.xlim(0, 1.0)
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_useMathText(True)

    plt.savefig(f'Baro_evolution_{mode}.pdf', bbox_inches='tight')
    plt.savefig(f'Baro_evolution_{mode}.svg', bbox_inches='tight')
    plt.close()

#Check 2: barometric formula
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#rho_adia = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Baro/rho_adia_sim_dx_0.01.npy')
#rho_iso = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Baro/rho_iso_sim_dx_0.01.npy')
#plot_baro(rho_iso, 'isothermal')
#plot_baro(rho_adia, 'adiabatic')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Check 3: speed of sound
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_arrival_time(rho_iso, rho_adia, dx, dt, source_x, target_x):

    set_scientific_style()

    y, z = rho_iso.shape[2]//2, rho_iso.shape[3]//2
    sig_iso, sig_adia = rho_iso[:, target_x, y, z], rho_adia[:, target_x, y, z]
    fluc_iso, fluc_adia = sig_iso - np.mean(sig_iso[:10]), sig_adia - np.mean(sig_adia[:10])

    peaks_iso, _ = find_peaks(np.abs(fluc_iso), height=np.max(np.abs(fluc_iso))*0.1)
    peaks_adia, _ = find_peaks(np.abs(fluc_adia), height=np.max(np.abs(fluc_adia))*0.1)

    if len(peaks_iso) == 0: raise ValueError("No isothermal peak found")
    if len(peaks_adia) == 0: raise ValueError("No adiabatic peak found")

    peak_iso, peak_adia = peaks_iso[0], peaks_adia[0]

    t_iso, t_adia = peak_iso * dt * 1000, peak_adia * dt * 1000

    dist = abs(target_x - source_x) * dx

    c_iso, c_adia = dist / (peak_iso * dt), dist / (peak_adia * dt)

    t = np.arange(len(sig_iso)) * dt * 1000

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(t, sig_iso, lw=2, label=f"Isothermal\n$(c_s={c_iso:.1f}\\,\\mathrm{{m/s}})$", zorder=2)
    ax.plot(t, sig_adia, lw=2, label=f"Polytropic\n$(c_s={c_adia:.1f}\\,\\mathrm{{m/s}})$", zorder=1)

    ax.axvline(t_iso, color='C0', ls='--', lw=2)
    ax.axvline(t_adia, color='C1', ls='--', lw=2)

    ax.text(t_iso+0.17, np.max(sig_adia), rf"${t_iso:.2f}\,\mathrm{{ms}}$", rotation=90, va='top', ha='right', color='C0', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.text(t_adia+0.05, np.max(sig_adia), rf"${t_adia:.2f}\ \mathrm{{ms}}$", rotation=90, va='top', ha='left', color='C1', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

    ax.plot(t_iso, sig_iso[peak_iso], 'ko')
    ax.plot(t_adia, sig_adia[peak_adia], 'ko')

    ax.set_xlabel("t [ms]")
    ax.set_ylabel(r"$\rho\,[kg/m^3]$")

    ax.legend(loc='upper left')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_major_formatter().set_useMathText(True)

    plt.savefig("ArrivalTime_compare.pdf", bbox_inches='tight')
    plt.savefig("ArrivalTime_compare.svg", bbox_inches='tight')
    plt.close()


#Check 3: speed of sound
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#rho_adia = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Speed/rho_adia_sim_dx_0.01.npy')
#rho_iso = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Speed/rho_iso_sim_dx_0.01.npy')

#plot_arrival_time(rho_iso, rho_adia, 0.01, 0.003/300, 10, 90)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Check 4: Diffusion
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_second_moment(c, dt):

    set_scientific_style()

    n_t, Nx, Ny, Nz = c.shape
    x = y = z = np.linspace(0, 1.0, Nx)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r2 = (X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2
    psi = np.array([np.sum(c[t] * r2) / np.sum(c[t]) for t in range(n_t)])

    t = np.arange(n_t) * dt

    fit = t <= 1000

    slope, intercept = np.polyfit(t[fit], psi[fit], 1)

    t_fit = np.linspace(0, 2500, 200)

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(t_fit, slope*t_fit + intercept, 'k--', lw=3, label="Second-moment growth\n" r"rate $\partial_t\psi$")
    ax.plot(t, psi, lw=2, label=rf"Simulation $(D_s=\partial_t\psi/6$" "\n" rf"$={float(f'{slope/6:.2e}'.split('e')[0]):.2f}\times10^{{{int(f'{slope/6:.2e}'.split('e')[1])}}}\,\mathrm{{m^2/s}})$")

    ax.text(1000, intercept, rf"$\partial_t\psi = {float(f'{slope:.2e}'.split('e')[0]):.2f}\times10^{{{int(f'{slope:.2e}'.split('e')[1])}}}\,\mathrm{{m^2/s}}$", rotation=68, rotation_mode='anchor', fontsize=16, ha='left', va='bottom', backgroundcolor='white')


    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\psi\,[m^2]$")

    ax.set_title(r"Second moment $\psi$")
    ax.legend(loc='lower right')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax.yaxis.get_major_formatter().set_useMathText(True)

    plt.savefig("SecondMoment.pdf", bbox_inches='tight')
    plt.savefig("SecondMoment.svg", bbox_inches='tight')
    plt.close()

#Check 4: Diffusion
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#c = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Diffusion/c_sim_dx_0.01.npy')
#plot_second_moment(c, 10000/100)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Check 5: Conservation
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_conservation(rho, conc, vx, vy, vz, dt):

    set_scientific_style()

    n_t = rho.shape[0]

    mass = rho.sum(axis=(1, 2, 3))
    tracer = conc.sum(axis=(1, 2, 3))

    momentum_x = np.empty(n_t)
    momentum_y = np.empty(n_t)
    momentum_z = np.empty(n_t)

    for t in range(n_t):
        momentum_x[t] = np.sum(
            0.5 * (rho[t, :-1] + rho[t, 1:]) * vx[t]
        )
        momentum_y[t] = np.sum(
            0.5 * (rho[t, :, :-1] + rho[t, :, 1:]) * vy[t]
        )
        momentum_z[t] = np.sum(
            0.5 * (rho[t, :, :, :-1] + rho[t, :, :, 1:]) * vz[t]
        )

    mass_err = (mass - mass[0]) / mass[0]
    tracer_err = (tracer - tracer[0]) / tracer[0]

    momentum = np.sqrt(
        momentum_x**2 +
        momentum_y**2 +
        momentum_z**2
    )

    v_mean_t = np.sqrt(
        np.mean(vx**2, axis=(1, 2, 3)) +
        np.mean(vy**2, axis=(1, 2, 3)) +
        np.mean(vz**2, axis=(1, 2, 3))
    )

    v_mean = np.mean(v_mean_t)

    momentum_err = momentum / (mass * v_mean)

    time = np.arange(n_t) * dt

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.yaxis.set_major_formatter(formatter)

    ax.plot(time, mass_err, lw=2, label="Mass")
    ax.plot(time, tracer_err, lw=2, label="Concentration")
    ax.plot(time, momentum_err, lw=2, label="Total Momentum")

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("Relative error")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig("Conservation.pdf", bbox_inches="tight")
    plt.savefig("Conservation.svg", bbox_inches="tight")
    plt.close()


#Check 5: Conservation laws
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#rho = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Conservation/rho_sim_dx_0.01.npy')        
#vx = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Conservation/vx_sim_dx_0.01.npy')
#vy = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Conservation/vy_sim_dx_0.01.npy')
#vz = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Conservation/vz_sim_dx_0.01.npy')
#c = np.load('D:/Lars/Documents/FU_Physik/Master/Masterarbeit/Sim Check/Results/Conservation/c_sim_dx_0.01.npy')

#plot_conservation(rho, c, vx, vy, vz, 100/100)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

