import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------
# GROUP INFORMATION 
# Names: Calton Louis Paul and Colin Zhai
# Student Numbers: 1011278289, 1011541501
#-----------------------------------------------------------------

# TASK 1: HEAT REMOVAL

# Given parameters
L = 700                 # Heat transfer coefficient (W/K)
F = 1                   # Mass flow rate (kg/s)
Cp = 4200               # Heat capacity (J/kg·K)
T_a = 288.15            # Ambient temperature (K)

# Temperature range
T_reactor = np.linspace(280, 450, 200)

# Heat removal term
Q_removal = (L + F * Cp) * (T_reactor - T_a)

# Plot
plt.figure(figsize=(7,5))
plt.plot(T_reactor, Q_removal, linewidth=2, label='Heat Removal')

plt.title('Heat Removal vs. Reactor Temperature')
plt.xlabel('Temperature, T (K)')
plt.ylabel(r'Heat Removal Rate, $Q_{rem}$ (W)')
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.grid(alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()


#-----------------------------------------------------------------

# TASK 2: HEAT GENERATION

# Given parameters
A = 4.13e8              # Frequency factor (s^-1)
Ea = 65400              # Activation energy (J/mol)
R = 8.314               # Gas constant (J/mol·K)
dH = -80000             # Heat of reaction (J/mol)
mc = 5                  # m * c term

# Temperature range
T = np.linspace(280, 700, 400)

# Heat generation term
Q_gen = -dH * mc * A * np.exp(-Ea / (R * T))

# Plot
plt.figure(figsize=(7,5))
plt.plot(T, Q_gen, linewidth=2, color = 'red', label='Heat Generation')

plt.title('Heat Generation vs. Reactor Temperature')
plt.xlabel('Temperature, T (K)')
plt.ylabel(r'Heat Generation Rate, $Q_{gen}$ (W)')
plt.grid(alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
