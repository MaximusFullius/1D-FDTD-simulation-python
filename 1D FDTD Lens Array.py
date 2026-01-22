import numpy as np
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────
eps0 = 8.854187817e-12
mu0  = 4e-7 * np.pi
c0   = 1.0 / np.sqrt(eps0 * mu0)
ETA0 = np.sqrt(mu0/eps0)  # ~377 Ω

# ── Lens array per slides ──────────────────────────────────────────
N_lenses = 20          # N
Gamma    = 9           # mesh blocks per lens (Γ)
dx       = 0.1         # Δ (m) per mesh block
nx       = N_lenses * Gamma
L        = nx * dx

# β from N:  N = ln(1000)/ln(β)  ⇒  β = exp(ln(1000)/N)
beta = float(np.exp(np.log(1000.0) / N_lenses))

# Index profile (piecewise-constant per lens) and epsilon_r(x)
n_lens = 1000.0 / (beta ** np.arange(N_lenses))      # n1=1000, geometric drop
eps_r  = np.repeat((n_lens**2).astype(np.float64), Gamma)     # size nx
eps_rE = np.concatenate([eps_r, [eps_r[-1]]])                 # size nx+1 for Ez

# μ_r = 1, no losses
mu_r   = np.ones(nx)
sigma_e = np.zeros(nx+1)
sigma_m = np.zeros(nx)

# ── Time setup from slides ─────────────────────────────────────────
dt   = dx / (Gamma * c0)                     # Δt = Δ/(Γ c0)
tmax = 1000 * N_lenses * Gamma * dx / c0     # recommended max time
nt   = int(np.ceil(tmax / dt))

# ── Source: ~50 kHz Gaussian injected inside X1 ───────────────────
f0      = 50e3
sigma_t = 1.0 / (2*np.pi*f0)
t       = np.arange(nt) * dt
t0      = 6 * sigma_t
src     = np.exp(-((t - t0)/sigma_t)**2)
src_i   = Gamma // 2   # near the start of lens X1

# ── Field arrays ───────────────────────────────────────────────────
Ez = np.zeros(nx+1, dtype=np.float64)
Hy = np.zeros(nx,   dtype=np.float64)

# Update coefficients (spatially varying ε)
chy = dt / (mu0 * mu_r * dx)                 # Hy update coeff (size nx)
cez = dt / (eps0 * eps_rE[:-1] * dx)         # Ez update coeff (use 1..nx-1)

# Mur-1 absorbing boundaries using local wave speeds
c_left  = c0 / np.sqrt(eps_r[0])
c_right = c0 / np.sqrt(eps_r[-1])
mur_L = (c_left*dt - dx)/(c_left*dt + dx)
mur_R = (c_right*dt - dx)/(c_right*dt + dx)
Ez_prev_left = 0.0
Ez_prev_right = 0.0

# ── Main FDTD loop (final snapshot only) ───────────────────────────
for n in range(nt):
    # Update H
    Hy += chy * (Ez[1:] - Ez[:-1])

    # Hard source on Ez
    Ez[src_i] += src[n]

    # Update EZ interior
    Ez[1:nx] += cez[1:nx] * (Hy[1:] - Hy[:-1])

    # Mur boundaries
    new_left  = Ez[1]  + mur_L * (Ez[1]  - Ez_prev_left)
    new_right = Ez[-2] + mur_R * (Ez[-2] - Ez_prev_right)
    Ez_prev_left, Ez_prev_right = Ez[0], Ez[-1]
    Ez[0], Ez[-1] = new_left, new_right

# ── Plot: 2D twin axes + colored lens bands ───────────────────────
x = np.linspace(0.0, L, nx+1)
Hy377 = np.concatenate([Hy, [0.0]]) * ETA0   # “restored” physical scaling

fig, ax1 = plt.subplots(figsize=(12, 6))

# Lens coloring as background bands (color by n)
cmap = plt.get_cmap('turbo')
n_min, n_max = n_lens.min(), n_lens.max()
for k in range(N_lenses):
    x0 = k * Gamma * dx
    x1 = x0 + Gamma * dx
    color = cmap((n_lens[k]-n_min)/(n_max - n_min + 1e-12))
    ax1.axvspan(x0, x1, color=color, alpha=0.18, linewidth=0)

# Left axis: Ez
ax1.plot(x, Ez, color='tab:blue', label='Ez [V/m]', lw=2)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('Ez [V/m]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Right axis: Hy×377
ax2 = ax1.twinx()
ax2.plot(x, Hy377, color='tab:red', ls='--', label='Hy×377 [A/m]', lw=2)
ax2.set_ylabel('Hy×377 [A/m]', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title with parameters (restored “variable calculations” line)
title = (f"1D FDTD Final Snapshot  •  N={N_lenses}, Γ={Gamma}, Δ={dx:.3f} m, "
         f"β≈{beta:.4f},  L={L:.2f} m,  dt={dt:.3e} s")
plt.title(title, pad=14)

# Legend (combine both axes)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right')

plt.tight_layout()
plt.show()
