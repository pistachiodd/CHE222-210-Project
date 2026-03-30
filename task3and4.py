import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#constants (from Toro et al.)
f = 1.7
gamma = 1
epsilon = 10
ThetaA = 0.037167029 #original code used 0.0379
ThetaS = ThetaA

#time span
tau_span = (0, 0.5) #simulate from tau = 0 to 0.5 
tau_eval = np.linspace(0, 0.5, 1000)

#initial conditions
theta0 = 0.0339 #this is also a problem, i have no clue what this should be
Y0 = [1, theta0]

#heat transfer values
Lvalues = [1700, 1600, 1590, 1588, 1587.4, 1587.3]

#store results
results = []

#ODE system
def task3n4func(tau, y, f, gamma, epsilon, theta_a, l_value, theta_s):
    '''
    Defines the system of ODE's:
    u = concentration
    theta = temperature
    
    Returns du/dtau and dtheta/dtau
    '''
    u, theta = y
    reaction = np.exp(1 / theta_s - 1 / theta)
    dudt = -u * reaction + f * (1 - u) #mass balance
    dthetadt = (reaction * u + epsilon * f * (gamma * theta_a - theta) - l_value * (theta - theta_a)) / epsilon #energy balance
    return [dudt, dthetadt]

#solve ODEs for main L values
for L in Lvalues:
    sol = solve_ivp(task3n4func, tau_span, Y0, args=(f, gamma, epsilon, ThetaA, L, ThetaS), t_eval=tau_eval, method='BDF') #BDF is a stiff solver
    tau = sol.t
    Y = sol.y
    results.append((tau, Y))

#solve separately for L = 700 (simulates heat removal conditions during the disaster)
L_700 = 700
sol_700 = solve_ivp(task3n4func, tau_span, Y0, args=(f, gamma, epsilon, ThetaA, L_700, ThetaS), t_eval=tau_eval, method='BDF')
tau_700 = sol_700.t
Y_700 = sol_700.y

# -----------------------------
# TASK 3: Concentration (u vs τ)
# -----------------------------
plt.figure()
for i, L in enumerate(Lvalues):
    tau_i, data = results[i]
    plt.plot(tau_i, data[0, :], label=f'L = {L}', linewidth=1.5)

plt.plot(tau_700, Y_700[0, :], label='L = 700', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel('u')
plt.title('Dimensionless Concentration vs τ')
plt.legend()
plt.grid()
plt.ylim([0.6, 1])
plt.xlim([0, 0.5])

# -----------------------------
# TASK 4: Temperature (θ vs τ)
# -----------------------------
plt.figure()
for i, L in enumerate(Lvalues):
    tau_i, data = results[i]
    plt.plot(tau_i, data[1, :], label=f'L = {L}', linewidth=1.5)

plt.plot(tau_700, Y_700[1, :], label='L = 700', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\theta$')
plt.title('Dimensionless Temperature vs τ')
plt.legend()
plt.grid()
plt.ylim([0.0335, 0.042])
plt.xlim([0, 0.2])

plt.show()
