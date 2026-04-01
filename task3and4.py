import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#-----------------------------------------------------------------
# GROUP INFORMATION 
# Names: Teo Tan and Aaron Greaves 
# Student Numbers: 1011046253, 1011047984
#-----------------------------------------------------------------

# MODEL PARAMETERS (from Toro et al.)
# dimensionless parameters governing reaction kinetics, heat generation, and heat removal
f = 1.7              # dimensionless flow rate
gamma = 1            # dimensionless heat capacity ratio
epsilon = 10         # heat generation parameter
ThetaA = 0.0379      # ambient temperature (dimensionless)
ThetaS = ThetaA      # assumed equal based on given model

# SIMULATION DOMAIN, where τ represents dimensionless time
tau_span = (0, 0.5)
tau_eval = np.linspace(0, 0.5, 1000)

# INITIAL CONDITIONS
    # u = 1  --> initially pure reactant
    # theta0 --> slightly above ambient to initiate reaction
theta0 = 0.03912
Y0 = [1, theta0]

# HEAT REMOVAL PARAMETERS
# L controls heat removal strength:
    # - High L --> strong cooling 
    # - Low L  --> weak cooling 
Lvalues = [1700, 1600, 1590, 1588, 1587.4, 1587.3]

# Storage for simulation results
results = []


# ODE SYSTEM DEFINITION
# This function defines the coupled mass and energy balances
    # u = dimensionless concentration
    # θ = dimensionless temperature
# Returns:
    # du/dτ --> consumption of reactant
    # dθ/dτ --> change in temperature
def task3n4func(tau, y, f, gamma, epsilon, theta_a, l_value, theta_s):

    u, theta = y

    # Arrhenius-type reaction rate
    # Strongly temperature dependent → source of stiffness
    reaction = np.exp(1 / theta_s - 1 / theta)

    # Mass balance:
    dudt = -u * reaction + f * (1 - u)

    # Energy balance:
    dthetadt = (reaction * u + epsilon * f * (gamma * theta_a - theta) - l_value * (theta - theta_a) ) / epsilon

    return [dudt, dthetadt]


# MAIN SIMULATIONS (TASK 3 & 4)
# Solve ODEs for different heat removal values (L)
# BDF method is used because system is stiff due to exponential reaction term
for L in Lvalues:

    sol = solve_ivp(
        task3n4func,
        tau_span,
        Y0,
        args=(f, gamma, epsilon, ThetaA, L, ThetaS),
        t_eval=tau_eval,
        method='BDF',
        rtol=1e-15,
        atol=1e-18
    )

    results.append((sol.t, sol.y))

# DISASTER SCENARIO (L = 700)
# Represents Bhopal Disaster conditions
# Shorter time span used because system experiences rapid thermal runaway
L_700 = 700

tau_span_700 = (0, 0.025)
tau_eval_700 = np.linspace(0, 0.025, 1000)

sol_700 = solve_ivp(
    task3n4func,
    tau_span_700,
    Y0,
    args=(f, gamma, epsilon, ThetaA, L_700, ThetaS),
    t_eval=tau_eval_700,
    method='BDF'
)

tau_700 = sol_700.t
Y_700 = sol_700.y

#--------------------------------------------------------------------------

# TASK 3: CONCENTRATION PROFILES
# Plot u vs τ for all L values
plt.figure()

for i, L in enumerate(Lvalues):
    tau_i, data = results[i]
    plt.plot(tau_i, data[0, :], label=f'L = {L}', linewidth=1.5)

# Overlay disaster case
plt.plot(tau_700, Y_700[0, :], label='L = 700', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel('u')
plt.title('Dimensionless Concentration vs Time')
plt.legend()
plt.grid()
plt.ylim([0.55, 1])
plt.xlim([0, 0.5])

#--------------------------------------------------------------------------

# TASK 4: TEMPERATURE PROFILES
# Plot θ vs τ for all L values
plt.figure()

for i, L in enumerate(Lvalues):
    tau_i, data = results[i]
    plt.plot(tau_i, data[1, :], label=f'L = {L}', linewidth=1.5)

# Overlay disaster case
plt.plot(tau_700, Y_700[1, :], label='L = 700', linewidth=1.5)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\theta$')
plt.title('Dimensionless Temperature vs Time')
plt.legend()
plt.grid()
plt.ylim([0.038, 0.046])
plt.xlim([0, 0.5])

plt.show()
