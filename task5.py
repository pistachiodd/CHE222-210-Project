import numpy as np  
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------
# GROUP INFORMATION
# Name: Tony Huang
# Student Number: 1010928702
# -----------------------------------------------------------------

# MODEL PARAMETERS
# Constants define the dimensionless reactor system based on the MIC hydrolysis model (Toro et al.)
f = 1.7                            # dimensionless flow rate
epsilon = 10                       # heat generation parameter
gamma = 1                          # heat capacity ratio
theta_s = 8.3145 * 292 / 65400     # Arrhenius scaling term (activation temperature)
theta_a_values = [0.0373, 0.0379]  # ambient temperature values used to test sensitivity
dimless_heat_transfer = 700        # heat removal parameter


# SYSTEM OF ODEs
# Defines the coupled mass and energy balances:
    # u = dimensionless concentration
    # theta = dimensionless temperature
# Returns:
    # du/dt --> rate of reactant consumption
    # dtheta/dt --> rate of temperature change
def system(t, y, theta_a):

    u, theta = y

    # Temperature-dependent reaction rate: introduces stiffness
    exponential_term = np.exp((1 / theta_s) - (1 / theta))

    # Mass balance:
    dudt = -u * exponential_term + f * (1 - u)

    # Energy balance:
    d_theta_dt = (u * exponential_term + epsilon * f * (gamma * theta_a - theta) - dimless_heat_transfer * (theta - theta_a)) / epsilon

    return [dudt, d_theta_dt]

# SIMULATION TIME DOMAIN
# Solve from τ = 0 to τ = 4 (same as paper)
eval_time = (0, 4)



# NUMERICAL SOLUTION
# The system is solved for different ambient temperatures to observe sensitivity and potential thermal runaway behavior
for theta_a in theta_a_values:

    # Initial conditions:
        # u = 1 --> system starts with full reactant concentration
        # theta = theta_a --> initial temperature equals ambient
    initial_conditions = [1, theta_a]

    # Wrapper function required because solve_ivp expects f(t, y)
    def reaction_system(t, y):
        return system(t, y, theta_a)

    # Solve ODE system numerically
    solution = solve_ivp(reaction_system, eval_time, initial_conditions)

    # Plot temperature profile over time
    plt.plot(solution.t, solution.y[1], label=fr'$\theta_a = {theta_a}$')

plt.xlabel(r'$\tau$ (Dimensionless Time)')
plt.ylabel(r'$\theta$ (Dimensionless Temperature)')
plt.title('Time Sensitivity of MIC Hydrolysis Process')
plt.legend()
plt.show()
