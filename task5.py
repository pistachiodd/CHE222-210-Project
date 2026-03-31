import numpy as np #HI TONY
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

f = 1.7
epsilon = 10
gamma = 1
theta_s = 8.3145 * 292 / 65400
theta_a_values = [0.0373, 0.0379]
dimless_heat_transfer = 700

def system(t,y,theta_a):
    u, theta = y

    exponential_term = np.exp((1 / theta_s) - (1 / theta))
    dudt = -u * exponential_term + f * (1 - u)
    d_theta_dt = (u * exponential_term + epsilon*f*(gamma * theta_a - theta) - dimless_heat_transfer * (theta - theta_a)) / epsilon

    return [dudt, d_theta_dt]

eval_time = (0,4)

for theta_a in theta_a_values:
    initial_conditions = [1, theta_a]
    def reaction_system(t,y):
        return system(t, y, theta_a) #this runs really slowly
    solution = solve_ivp(reaction_system, eval_time, initial_conditions)

    plt.plot(solution.t, solution.y[1], label=fr'$\theta_a = {theta_a}$')

plt.xlabel(r'$\tau$ (Dimensionless Time)')
plt.ylabel(r'$\theta$ (Dimensionless Temperature)')
plt.title('Time Sensitivity of MIC Hydrolysis Process')

plt.legend()
#plt.grid()

plt.show()
