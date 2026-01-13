import matplotlib.pyplot as plt
import numpy as np

data = []
with open('A.txt', 'r') as f:
    for line in f:
        if line.startswith('\\') or '|' in line or line.strip() == '':
            continue
        data.append(list(map(float, line.split())))
data = np.array(data)
t = data[:, 0]
rv = data[:, 1]
rv_err = data[:, 2]
plt.errorbar(t, rv, yerr=rv_err, fmt='.k', capsize=0)
plt.ylabel("radial velocity [m/s]")
plt.xlabel("time[JD]")
plt.title("RAw data - A")
plt.savefig("Raw_data_A.pdf")
plt.show()
t -= np.mean(t)
print(len(t))

lit_period = 2.64388312
plt.errorbar((t % lit_period) / lit_period, rv, yerr=rv_err, fmt=".k", capsize=0)
plt.ylabel("radial velocity [m/s]")
plt.xlabel("phase")
plt.title("Phase folded plot - A")
plt.savefig("Phase_Folded_A.pdf")
plt.show()

import pymc as pm
import corner
import emcee
import exoplanet as xo
import pytensor.tensor as pt
from pytensor import function
M_sym = pt.scalar("M")
e_sym = pt.scalar("e")
f_sym = xo.orbits.get_true_anomaly(M_sym, e_sym)
true_anomaly_func = function([M_sym, e_sym], f_sym)

logK_init = np.log(0.5 * (np.max(rv) - np.min(rv)))
logP_init = np.log(lit_period)
phi_init = 0.1
hk1_init, hk2_init = 0.01, 0.01
rv0_init = 0.0
rvtrend_init = 0.0
initial_guess = [logK_init, logP_init, phi_init, hk1_init, hk2_init, rv0_init, rvtrend_init]

def rv_model(theta, t):
    """
    theta = [logK, logP, phi, hk1, hk2, rv0, rvtrend]
    """
    logK, logP, phi, hk1, hk2, rv0, rvtrend = theta
    K = np.exp(logK)
    P = np.exp(logP)

    e = hk1**2 + hk2**2
    w = np.arctan2(hk2, hk1)

    M = 2 * np.pi * ((t / P) % 1) - (phi + w)

    f = np.array([float(np.array(true_anomaly_func(M_i, e))) for M_i in np.atleast_1d(M)])

    background = rv0 + rvtrend * (t/365.25)

    return background - K * (np.cos(f + w) + e * np.cos(w))

def log_likelihood(theta, t, rv, rv_err):
    model = rv_model(theta, t)
    diff = ((rv - model) / rv_err)**2 + np.log(2 * np.pi * rv_err**2)
    return -0.5 * np.sum(np.array(diff))

def log_prior(theta):
    logK, logP, phi, hk1, hk2, rv0, rvtrend = theta

    if not (0 < logK < np.log(200)):
        return -np.inf
    if not (0 < logP < np.log(10)):
        return -np.inf
    if not (0 < phi < 2 * np.pi):
        return -np.inf
    if hk1**2 + hk2**2 >= 1:
        return -np.inf
    if not (-50 < rv0 < 50):
        return -np.inf
    if not (-10 < rvtrend < 10):
        return -np.inf
    return 0.0

def log_probability(theta, t, rv, rv_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    val = lp + log_likelihood(theta, t, rv, rv_err)
    return np.array(val).item()

import time
ndim, nwalkers = 7, 32
initial_pos = np.array([initial_guess + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)])
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t, rv, rv_err))
start_time = time.time()
sampler.run_mcmc(initial_pos, 5000)
end_time = time.time()
total_time = end_time - start_time
hours = total_time // 3600
minutes = (total_time % 3600) // 60
seconds = total_time % 60
print(f"MCMC run completed in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
samples = sampler.get_chain(discard=1000, thin=10, flat=True)
print("Samples shape:", samples.shape)
best_params = np.median(samples, axis=0)
logK_fit, logP_fit, phi_fit, hk1_fit, hk2_fit, rv0_fit, rvtrend_fit = best_params
K_fit = np.exp(logK_fit)
P_fit = np.exp(logP_fit)
e_fit = hk1_fit**2 + hk2_fit**2
w_fit = np.arctan2(hk2_fit, hk1_fit)
print("Best-fit parameters:")
print(f"K = {K_fit:.3f} m/s")
print(f"P = {P_fit:.3f} days")
print(f"e = {e_fit:.3f}")
print(f"w = {w_fit:.3f} rad")
print(f"rv0 = {rv0_fit:.3f} m/s")
print(f"rvtrend = {rvtrend_fit:.3f} m/s/day")
rv_fit = rv_model(best_params, t)
background = rv0_fit + rvtrend_fit * (t/365.25)
rv_planet = rv - background
phase_data = (t % P_fit)

phase_grid = np.linspace(0, P_fit, 500)
M_grid = 2 * np.pi * (phase_grid / P_fit) - (phi_fit + w_fit)
f_grid = np.array([float(true_anomaly_func(M, e_fit)) for M in M_grid])
rv_phase_model = - K_fit * (np.cos(f_grid + w_fit) + e_fit * np.cos(w_fit))

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(phase_data, rv_planet, yerr=rv_err, fmt=".k", label="Phase-Folded RV")
ax.plot(phase_grid, rv_phase_model, color="C1", lw=1.5, label="RV Model")
ax.set_xlabel("Phase [days]")
ax.set_ylabel("Radial Velocity [m/s]")
ax.set_ylim(-110, 110)
ax.legend()
plt.title("Model fitted Plot - A Peg")
plt.savefig("model_Fitted_A.pdf")
plt.tight_layout()
plt.show()

K_samples = np.exp(samples[:, 0])
P_samples = np.exp(samples[:, 1])
e_samples = samples[:, 3]**2 + samples[:, 4]**2
w_samples = np.arctan2(samples[:, 4], samples[:, 3])
derived_samples = np.column_stack([K_samples, P_samples, e_samples, w_samples])
labels = [r"$K$", r"$P$", r"$e$", r"$\omega$"]
fig_corner = corner.corner(
    derived_samples,
    labels=labels,
    bins=30,
    smooth=1.0,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
plt.show()

import astropy.constants as const
import numpy as np
G = const.G.value  # m^3 kg^-1 s^-2
M_sun = const.M_sun.value  # kg
M_jup = const.M_jup.value  # kg
au = const.au.value  # m

M_star = 0.425 * M_sun

P_sec = P_fit * 86400.0  # days → seconds
Mpsini_kg = (
    K_fit
    * np.sqrt(1.0 - e_fit**2)
    * M_star**(2.0/3.0)
    * P_sec**(1.0/3.0)
) / ((2.0 * np.pi * G)**(1.0/3.0))
Mpsini_Mjup = Mpsini_kg / M_jup
a_m = (G * M_star * P_sec**2 / (4.0 * np.pi**2)) ** (1.0/3.0)
a_au = a_m / au
print(f"Minimum mass M_p sin i = {Mpsini_Mjup:.3f} M_jup")
print(f"Semi-major axis a = {a_au:.3f} AU")
