import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpmath import mp, mpf

# ─── 1) Configure high precision ─────────────────────────────────
mp.dps = 80
eps0 = mpf('8.854187817e-12')
mu0  = mpf('4') * mp.pi * mpf('1e-7')
c_hp = mpf('1') / mp.sqrt(eps0 * mu0)

# ─── 2) Simulation parameters (mpf) ───────────────────────────────
domain_size = mpf('1.0')
dx          = mpf('1e-3')
S           = mpf('0.15')
dt_hp       = S * dx / c_hp
nt          = 2000
nx          = int(domain_size / dx)
src_idx     = nx // 2

# ─── 3) High‑precision field & material arrays ────────────────────
Ez_hp     = [mpf('0')] * (nx+1)
Hy_hp     = [mpf('0')] * nx
Jz_hp     = [mpf('0')] * (nx+1)
My_hp     = [mpf('0')] * nx
eps_r_hp  = [mpf('1')] * (nx+1)
mu_r_hp   = [mpf('1')] * nx
sigma_e_hp= [mpf('1e-5')] * (nx+1)
sigma_m_hp= [mpf('1e-8')] * nx

# ─── 4) Precompute high‑precision coefficients ─────────────────────
Ceze_hp  = [(2*eps_r_hp[i]*eps0 - dt_hp*sigma_e_hp[i]) /
            (2*eps_r_hp[i]*eps0 + dt_hp*sigma_e_hp[i])
            for i in range(nx+1)]
Cezhy_hp = [(2*dt_hp/dx) /
            (2*eps_r_hp[i]*eps0 + dt_hp*sigma_e_hp[i])
            for i in range(nx+1)]
Cezj_hp  = [(-2*dt_hp) /
            (2*eps_r_hp[i]*eps0 + dt_hp*sigma_e_hp[i])
            for i in range(nx+1)]

Chyh_hp  = [(2*mu_r_hp[i]*mu0 - dt_hp*sigma_m_hp[i]) /
            (2*mu_r_hp[i]*mu0 + dt_hp*sigma_m_hp[i])
            for i in range(nx)]
Chyez_hp = [(2*dt_hp/dx) /
            (2*mu_r_hp[i]*mu0 + dt_hp*sigma_m_hp[i])
            for i in range(nx)]
Chym_hp  = [(-2*dt_hp) /
            (2*mu_r_hp[i]*mu0 + dt_hp*sigma_m_hp[i])
            for i in range(nx)]

# ─── 5) High‑precision Gaussian source ────────────────────────────
time_hp   = [dt_hp * mpf(t) for t in range(nt)]
pulse_c   = mpf('3e-9')
pulse_w   = mpf('0.5e-9')
Jz_wave_hp= [mp.e**(-((t-pulse_c)/pulse_w)**2) for t in time_hp]

# ─── 6) Build float grids & figure ────────────────────────────────
x_positions = np.linspace(0.0, 1.0, nx+1)
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

# Draw “infinite” PEC planes once
plane_size = 1e166
for xp in [0.0, 1.0]:
    verts = np.array([
        [xp, -plane_size, -plane_size],
        [xp, -plane_size,  plane_size],
        [xp,  plane_size,  plane_size],
        [xp,  plane_size, -plane_size],
    ], dtype=float)
    ax.add_collection3d(Poly3DCollection([verts],
                                         facecolors='green',
                                         alpha=0.2,
                                         edgecolor='none'))

ax.set_xlabel('x [m]')
ax.set_ylabel('[A/m]')
ax.set_zlabel('[V/m]')
ax.set_xlim(0,1)
ax.view_init(elev=25, azim=-60)
ax.grid(True)
plt.ion()

# ─── 7) Initial placeholder lines ─────────────────────────────────
Ez_plot = np.zeros(nx+1)
Hy_plot = np.zeros(nx+1)
lez = ax.plot3D(x_positions, np.zeros_like(x_positions), Ez_plot, 'b', lw=3, label='Ez')[0]
lhy = ax.plot3D(x_positions, Hy_plot,                np.zeros_like(Hy_plot), 'r-.', lw=2, label='Hy×377')[0]
ax.legend()

# ─── 8) Main FDTD loop (hp updates + float cast + plot) ─────────
for t in range(nt):
    # 8.1) inject source
    Jz_hp[src_idx] = Jz_wave_hp[t]

    # 8.2) magnetic update (hp)
    for i in range(nx):
        Hy_hp[i] = (Chyh_hp[i]*Hy_hp[i]
                    + Chyez_hp[i]*(Ez_hp[i+1] - Ez_hp[i])
                    + Chym_hp[i]*My_hp[i])

    # 8.3) electric update (hp) + PEC
    for i in range(1, nx):
        Ez_hp[i] = (Ceze_hp[i]*Ez_hp[i]
                    + Cezhy_hp[i]*Hy_hp[i-1]
                    + Cezj_hp[i]*Jz_hp[i])
    Ez_hp[0] = Ez_hp[-1] = mpf('0')

    # ───8.4) Downcast to float64───────────────────────────────────
    Ez_f   = np.array([float(v) for v in Ez_hp])
    Hy_f   = np.array([float(v)*377 for v in Hy_hp]) 
    Hy_pad = np.concatenate([Hy_f, [0.0]])      # length = nx+1

    # ───8.5) Remove the old lines (we kept references as 'lez','lhy') 
    lez.remove()
    lhy.remove()

    # ───8.6) Redraw Ez as a blue curve: z = Ez_f, y = 0────────────
    lez = ax.plot3D(
        x_positions,
        np.zeros_like(x_positions),  # y = 0 plane
        Ez_f,                        # z = Ez
        'b', linewidth=3, label='Ez' if t==0 else None
    )[0]

    # ───8.7) Redraw Hy×377 as a red dashed curve: y = Hy_pad, z = 0─
    lhy = ax.plot3D(
        x_positions,
        Hy_pad,                      # y = Hy×377
        np.zeros_like(Hy_pad),       # z = 0 plane
        'r-.', linewidth=2, label='Hy×377' if t==0 else None
    )[0]

    # ───8.8) Rescale Y/Z (keep min ±0.2)──────────────────────────
    max_y = np.max(np.abs(Hy_pad))
    max_z = np.max(np.abs(Ez_f))
    y_lim = max(max_y*1.1, 0.2)
    z_lim = max(max_z*1.1, 0.2)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_zlim(-z_lim, z_lim)

    # ───8.9) Update title & pause──────────────────────────────────
    time_ns = float(t * dt_hp * 1e9)
    ax.set_title(f"step {t}, t={time_ns:.2f} ns")
    plt.pause(0.005)

plt.ioff()
plt.show()
