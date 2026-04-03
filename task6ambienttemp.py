import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters from the Bhopal incident
f = 1.7
epsilon = 10.0
gamma = 1.0
l = 700.0
R = 8.314
E_a = 65400.0
T_s = 292.0
theta_s = (R * T_s) / E_a


def mic_sys(tau, state, theta_a):
    '''
    (number, number, number) -> number
    this is the ode system given in the paper
    '''
    u, theta = state
    # this has saved me a few errors
    if theta < 1e-4:
        rate = 0
    else:
        rate = np.exp(1 / theta_s - 1 / theta)

    # ODE equations
    du = -u * rate + f * (1 - u)
    dtheta = (u * rate + epsilon * f * (gamma * theta_a - theta) - l * (theta - theta_a)) / epsilon
    return [du, dtheta]


# sweeping from 0.035 to 0.045 or about 275K to 354K
theta_a_vals = np.linspace(0.035, 0.045, 150)
theta_steady = []
theta_a_steady = []
theta_max = []
theta_min = []
theta_a_osc = []

tau_span = (0, 100)  # is this enough to reach steady state
tau_eval = np.linspace(0, 100, 10000)
init_state = [1.0, 0.035]  #pure MIC at 275K

for ta in theta_a_vals:
    # Solve the system using Radau, a stiff solver
    sol = solve_ivp(mic_sys, tau_span, init_state, t_eval=tau_eval, args=(ta,), method='Radau')

    # Extract the last 30% of the time series to check for steady states or limit cycles
    tail_idx = int(0.7 * len(sol.t))
    theta_tail = sol.y[1][tail_idx:]

    t_max = np.max(theta_tail)
    t_min = np.min(theta_tail)

    # If the amplitude is greater than a small threshold, treat it as oscillatory
    if t_max - t_min > 1e-4:
        peaks, _ = find_peaks(theta_tail, distance=50)
        troughs, _ = find_peaks(-theta_tail, distance=50)

        if len(peaks) > 0 and len(troughs) > 0:
            theta_max.append(np.mean(theta_tail[peaks]))
            theta_min.append(np.mean(theta_tail[troughs]))
            theta_a_osc.append(ta)
        else:
            # Fallback if the wave is too erratic to find clean peaks
            theta_max.append(t_max)
            theta_min.append(t_min)
            theta_a_osc.append(ta)
    else:
        # System has settled to a steady state
        theta_steady.append(np.mean(theta_tail))
        theta_a_steady.append(ta)

    #dont touch this, this makes the graph smooth so the next iteration continues from the previous
    init_state = [sol.y[0][-1],sol.y[1][-1]]

# Plotting the diagram
plt.figure(figsize=(10, 6))
plt.plot(theta_a_steady, theta_steady, 'k.', markersize=6, label='Steady State')
plt.plot(theta_a_osc, theta_max, 'r.', markersize=4, label='Perodic Solution Max')
plt.plot(theta_a_osc, theta_min, 'b.', markersize=4, label='Perodic Solution Min')

plt.xlabel(r'Dimensionless cooling temperature ($\theta_a$)')
plt.ylabel(r'Dimensionless temperature ($\theta$)')
plt.title('Bifurcation Diagram: MIC Hydrolysis Temperature Dynamics')
#close enough
plt.axvline(x=0.0374, color='gray', linestyle='--', label='Region 1 to 2')
plt.axvline(x=0.0393, color='gray', linestyle=':', label='Region 2 to 3')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()