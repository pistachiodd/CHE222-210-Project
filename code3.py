import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Constants
# -----------------------
A = 1e7
Ea = 75000
R = 8.314
dH = -60000
m = 1
c = 1

# -----------------------
# Temperature range
# -----------------------
T = np.linspace(280, 700, 500)  # extend range to include steep part of exponential

# -----------------------
# Heat generation (exponential)
# -----------------------
Qgen = -dH * m * A * np.exp(-Ea / (R * T)) * c

# -----------------------
# Choose a tangent point in the steeper part of the curve
# -----------------------
T_crit = 500  # higher temperature → more curvature

# Calculate slope of Q_gen at T_crit
dQgen_dT = (-dH * m * A * Ea / (R * T_crit**2)) * np.exp(-Ea / (R * T_crit)) * c

# Tangent line at T_crit
Qrem = dQgen_dT * (T - T_crit) + Qgen[np.argmin(np.abs(T - T_crit))]

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,6))
plt.plot(T, Qgen, 'r', label='Heat Generation (Q_gen)', linewidth=2)
plt.plot(T, Qrem, 'b', label='Heat Removal (Q_rem)', linewidth=2)
plt.scatter(T_crit, Qgen[np.argmin(np.abs(T - T_crit))], color='black', zorder=5)
#plt.text(T_crit+5, Qgen[np.argmin(np.abs(T - T_crit))]*1.05, "Critical Tangent", color='green')

plt.xlabel("Temperature (K)")
plt.ylabel("Heat Rate")
plt.ylim(-0.05e6,0.2e6)
plt.title("Critical Tangent at Steep Part of Q_gen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()