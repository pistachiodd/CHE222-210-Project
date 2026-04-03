# -----------------------------------------------------------------
# GROUP INFORMATION
# Name: Tony Huang
# Student Number: 1010928702
# -----------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# system parameters (from paper)
l = 700
epsilon = 10.0
gamma = 1.0
R = 8.314
E_a = 65400.0
T_s = 292.0
theta_s = (R * T_s) / E_a
theta_a_fixed = 0.0379


# ode system (mic hydrolysis model)
def mic_sys(tau, state, f):
    u, theta = state

    # prevent numerical blow-up if theta becomes extremely small
    if theta < 1e-4:
        rate = 0
    else:
        # Arrhenius-type reaction rate
        rate = np.exp(1 / theta_s - 1 / theta)

    # Mass balance
    du = -u * rate + f * (1 - u)

    # Energy balance
    dtheta = (u * rate + epsilon * f * (gamma * theta_a_fixed - theta)
              - l * (theta - theta_a_fixed)) / epsilon
    return [du, dtheta]


# parameter sweep 
# we vary the flow rate (f) to analyze its effect on system stability
f_vals = np.linspace(0.5, 2, 100)

# storage for results
theta_steady = [] # stable steady states
f_steady = []

theta_max = [] # oscillation upper bound (limit cycle)
theta_min = [] # oscillation lower bound
f_osc = []

tau_span = (0, 100) # time span long enough to reach steady state or oscillations

# initial condition
init_state = [1.0, 0.035]


#------------------------------------------------------------------------------
# MAIN LOOP
for f in f_vals:

    # solve ode system for each value of f
    sol = solve_ivp(mic_sys, tau_span, init_state, args=(f,), method='Radau')

    # only analyze the tail (steady-state behaviour)
    tails = int(0.7 * len(sol.t))
    theta_ss = sol.y[1][tails:]

    t_max = np.max(theta_ss)
    t_min = np.min(theta_ss)

    # detect oscillations
    if t_max - t_min > 1e-4:
        # if variation exists --> system is oscillating

        # find peaks and troughs (limit cycle envelope)
        peaks, _ = find_peaks(theta_ss, distance=50)
        troughs, _ = find_peaks(-theta_ss, distance=50)

        if len(peaks) > 0 and len(troughs) > 0:
            # store average max/min values (envelope of oscillation)
            theta_max.append(np.mean(theta_ss[peaks]))
            theta_min.append(np.mean(theta_ss[troughs]))
            f_osc.append(f)

    else: # steady state (no oscillation)
        theta_steady.append(np.mean(theta_ss))
        f_steady.append(f)

    # continuing from previous solution for smoothness
    init_state = [sol.y[0][-1], sol.y[1][-1]]


#------------------------------------------------------------------------------
# plotting the bifurcation diagram
plt.figure(figsize=(10, 6))

# stable steady states
plt.plot(f_steady, theta_steady, 'k.', markersize=6, label='Stable Steady State')

# oscillation envelope (limit cycle)
plt.plot(f_osc, theta_max, 'r.', markersize=4, label='Oscillation Max')
plt.plot(f_osc, theta_min, 'b.', markersize=4, label='Oscillation Min')

# approximate critical point (where behaviour changes)
plt.axvline(x=1.63, color='gray', linestyle='--', label=r'Critical $f$ Value')

plt.xlabel(r'Dimensionless flow rate ($f$)')
plt.ylabel(r'Dimensionless temperature ($\theta$)')
plt.title(r'Bifurcation Diagram (Varying $f$) at $\theta_a = 0.0379$')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
