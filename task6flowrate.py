import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters from the Bhopal incident
l = 700
epsilon = 10.0
gamma = 1.0
R = 8.314
E_a = 65400.0
T_s = 292.0
theta_s = (R * T_s) / E_a
theta_a_fixed = 0.0379


def mic_sys(tau, state, f):
    '''
    (number, number, number) -> number
    this is the ode system given in the paper
    '''
    u, theta = state
    #this has saved me a few errors
    if theta < 1e-4:
        rate = 0
    else:
        rate = np.exp(1 / theta_s - 1 / theta)

    du = -u * rate + f * (1 - u)
    dtheta = (u * rate + epsilon * f * (gamma * theta_a_fixed - theta)
              - l * (theta - theta_a_fixed)) / epsilon
    return [du, dtheta]


# Sweep f
f_vals = np.linspace(0.5, 2, 100)

theta_steady = []
f_steady = []
theta_max = []
theta_min = []
f_osc = []

tau_span = (0, 100)
#tau_eval = np.linspace(0, 100, 10000)
init_state = [1.0, 0.035]

for f in f_vals:
    sol = solve_ivp(mic_sys, tau_span, init_state, args=(f,), method='Radau')

    tails = int(0.7 * len(sol.t)) #looks at the steady state part of the solution only, is this equivalent to 4 tau?
    theta_ss = sol.y[1][tails:]

#used to detect an oscilation
    t_max = np.max(theta_ss)
    t_min = np.min(theta_ss)

#singifies an oscillation instead of a steady state value
    if t_max - t_min > 1e-4:
        peaks, _ = find_peaks(theta_ss, distance=50) #shoutout to scipy for doing this automatically
        troughs, _ = find_peaks(-theta_ss, distance=50)

        if len(peaks) > 0 and len(troughs) > 0: #stores the peaks and trough that are automatically detected
            theta_max.append(np.mean(theta_ss[peaks]))
            theta_min.append(np.mean(theta_ss[troughs]))
            f_osc.append(f)
    else: #no oscillation detected, just take average ss value
        theta_steady.append(np.mean(theta_ss))
        f_steady.append(f)
    #continuing from previous solution for smoothness
    init_state = [sol.y[0][-1],sol.y[1][-1]]
# Plot the thing
plt.figure(figsize=(10, 6))
plt.plot(f_steady, theta_steady, 'k.', markersize=6, label='Steady State')
plt.plot(f_osc, theta_max, 'r.', markersize=4, label='Periodic Solution Max')
plt.plot(f_osc, theta_min, 'b.', markersize=4, label='Periodic Solution Min')

plt.axvline(x=1.63, color='gray', linestyle='--', label="Critical f Value")
plt.xlabel(r'Dimensionless flow rate ($f$)')
plt.ylabel(r'Dimensionless temperature ($\theta$)')
plt.title(r'Bifurcation Diagram at $\theta_a = 0.0379$')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()