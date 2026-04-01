import numpy as np
import matplotlib.pyplot as plt

L = 5000      # Heat transfer coefficient (W/K) - Placeholder
F = 10        # Mass flow rate (kg/s) - Placeholder
Cp = 2000     # Specific heat capacity (J/kg.K) - Placeholder
T_a = 273+11  # Ambient/Cooling temperature (K)

T_reactor = np.linspace(280, 450, 200)

Q_removal = (L + F * Cp) * (T_reactor - T_a)

plt.figure(figsize=(8, 6))
plt.plot(T_reactor, Q_removal, color='blue', linewidth=2, label='Heat Removal')

plt.title('Task 1: Heat Removal vs. Reactor Temperature (Dimensional)', fontsize=14)
plt.xlabel('Reactor Temperature, T (K)', fontsize=12)
plt.ylabel('Heat Removal Rate, $Q{rem}$ (W)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add a zero line
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

plt.show()