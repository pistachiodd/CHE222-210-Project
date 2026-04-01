import numpy as np
import matplotlib.pyplot as plt


L = 700     # Heat transfer coefficient (W/K)
F = 1        # Mass flow rate (kg/s)
Cp = 4200     # Specific heat capacity (J/kg.K)
T_a = 288.15  # Ambient/Cooling temperature (K)

delta_H = -80400  # Heat of reaction (J/mol)
E_a = 65400       # Activation energy (J/mol)
A = 4.13e8        # Frequency factor (s^-1)
R = 8.314         # Ideal gas constant (J/mol.K)

mc = 500   #5, for normal 500 for asymptote, 5000 for imminent runaway

T = np.linspace(200, 450, 200)
Q_removal = (L + F * Cp) * (T - T_a)

Q_generation = (-delta_H) * mc * A * np.exp(-E_a / (R * T))

plt.figure(figsize=(10, 6))
plt.plot(T, Q_removal, color='blue', linewidth=2, label='Heat Removal')
plt.plot(T, Q_generation, color='red', linewidth=2, label='Heat Generation')
plt.plot(T, Q_generation - Q_removal, color='green', linewidth=2, label='Net Generation')

plt.xlabel('Reactor Temperature, T (K)', fontsize=12)
plt.ylabel('Heat Rate (W)', fontsize=12)
plt.ylim(-3e5, 4e6 )
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':')


plt.legend(fontsize=11)
plt.show()