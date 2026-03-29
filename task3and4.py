import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#constants (from Toro et al.)
f = 1.7
gamma = 1
epsilon = 10
ThetaA = 0.037167029
ThetaS = ThetaA

# Time span
tau_span = (0, 0.5)
tau_eval = np.linspace(0, 0.5, 1000)

# Initial conditions
theta0 = 0.0335
Y0 = [1, theta0]

# Heat transfer values, 700 is done individually
Lvalues = [1700, 1600, 1590, 1588, 1587.4, 1587.3]

# Store results
results_matrix = []


# ODE system
def task3n4func(tau, y, f, gamma, epsilon, theta_a, l_value, theta_s):
    u, theta = y

    reaction = np.exp(1 / theta_s - 1 / theta)

    dudt = -u * reaction + f * (1 - u)

    dthetadt = (reaction * u + epsilon * f * (gamma * theta_a - theta) - l_value * (theta - theta_a) ) / epsilon

    return [dudt, dthetadt]


# Solve ODEs
for L in Lvalues:
    sol = solve_ivp(task3n4func, tau_span, Y0, args=(f, gamma, epsilon, ThetaA, L, ThetaS), t_eval=tau_eval, method='BDF')
    tau = sol.t
    Y = sol.y.T  # transpose to match MATLAB shape

    results_matrix.append((tau, Y))

L = 700
sol = solve_ivp(task3n4func, tau_span, Y0, args=(f, gamma, epsilon, ThetaA, L, ThetaS), t_eval=tau_eval, method='BDF')
tau = sol.t
Y = sol.y.T
results_matrix.append((tau,Y))

# -----------------------------
# Plot 1: Concentration (u vs τ)
# -----------------------------
plt.figure()
for i, L in enumerate(Lvalues):
    tau, data = results_matrix[i]
    u = data[:, 0]
    plt.plot(tau, u, label=f'L = {L}', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel('u')
plt.title('Dimensionless Concentration vs τ')
plt.legend()
plt.grid()
plt.ylim([0.52, 1])
plt.xlim([0, 0.5])

# -----------------------------
# Plot 2: Temperature (θ vs τ)
# -----------------------------
plt.figure()
for i, L in enumerate(Lvalues):
    tau, data = results_matrix[i]
    theta = data[:, 1]
    plt.plot(tau, theta, label=f'L = {L}', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\theta$')
plt.title('Dimensionless Temperature vs τ')
plt.legend()
plt.grid()
plt.ylim([0.0335, 0.042])
plt.xlim([0, 0.2])

plt.show()

