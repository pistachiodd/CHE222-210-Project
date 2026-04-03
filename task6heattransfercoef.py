# -----------------------------------------------------------------
# GROUP INFORMATION
# Name: Tony Huang
# Student Number: 1010928702
# -----------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# System Parameters (from paper)
f = 1.7                      # dimensionless flow rate
epsilon = 10.0               # heat capacity ratio term
gamma = 1.0                  # heat transfer scaling
R = 8.314  
E_a = 65400.0
T_s = 292.0
theta_s = (R * T_s) / E_a    # dimensionless activation temperature
theta_a_fixed = 0.0379       # fixed ambient temperature


# ode system (MIC hydrolysis model)
def mic_sys(tau, state, l):
    u, theta = state

    # prevent numerical blow-up if theta becomes extremely small
    if theta < 1e-4:
        rate = 0
    else:
        # Arrhenius-type reaction rate 
        rate = np.exp(1 / theta_s - 1 / theta)

    # Mass balance
    du = -u * rate + f * (1 - u)

    # Energy balance:
    dtheta = (u * rate + epsilon * f * (gamma * theta_a_fixed - theta) - l * (theta - theta_a_fixed)) / epsilon
    return [du, dtheta]


# PARAMETER SWEEP (BIFURCATION)
# we vary the heat transfer coefficient (l) to control stability
l_vals = np.linspace(600, 1800, 200)

# storage for results
theta_steady = [] # stable steady states
l_steady = []

theta_max = [] # oscillation upper bound (limit cycle)
theta_min = [] # oscillation lower bound
l_osc = []

tau_span = (0, 100) # time span long enough to reach steady state or oscillations
tau_eval = np.linspace(0, 100, 10000)

# Initial condition 
init_state = [1.0, 0.035]


#------------------------------------------------------------------------------
# MAIN LOOP
for l_val in l_vals:

    # solve ODE system for each value of l
    sol = solve_ivp(mic_sys, tau_span, init_state, t_eval=tau_eval, args=(l_val,),method='Radau')

    # only analyze the tail (steady-state behaviour)
    tails = int(0.7 * len(sol.t))
    theta_ss = sol.y[1][tails:]

    t_max = np.max(theta_ss)
    t_min = np.min(theta_ss)

    # Detect oscillations
    if t_max - t_min > 1e-4:
        # if variation exists --> system is oscillating

        # find peaks and troughs (limit cycle envelope)
        peaks, _ = find_peaks(theta_ss, distance=50)
        troughs, _ = find_peaks(-theta_ss, distance=50)

        if len(peaks) > 0 and len(troughs) > 0:
            # store average max/min values (envelope of oscillation)
            theta_max.append(np.mean(theta_ss[peaks]))
            theta_min.append(np.mean(theta_ss[troughs]))
            l_osc.append(l_val)

    else: # steady state (no oscillation)
        theta_steady.append(np.mean(theta_ss))
        l_steady.append(l_val)

#------------------------------------------------------------------------------
# plotting the bifurcation diagram
plt.figure(figsize=(10, 6))

# stable steady states
plt.plot(l_steady, theta_steady, 'k.', markersize=6, label='Stable Steady State')

# oscillation envelope (limit cycle)
plt.plot(l_osc, theta_max, 'r.', markersize=4, label='Oscillation Max')
plt.plot(l_osc, theta_min, 'b.', markersize=4, label='Oscillation Min')

# approximate critical point (where behaviour changes)
plt.axvline(x=743, color="grey", linestyle="--", label=r'Critical $\ell$ Value')

plt.xlabel(r'Heat Transfer Parameter ($\ell$)')
plt.ylabel(r'Dimensionless Temperature ($\theta$)')
plt.title(r'Bifurcation Diagram: Effect of Heat Removal on Reactor Stability')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
