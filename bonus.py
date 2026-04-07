import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Model Parameters ---
Q = 50           # Emission rate of MIC (kg/s) - Estimate
u = 2.5          # Wind speed (m/s) - Typical light wind
H = 10           # Release height (m) - Stack/tank height

# Grid setup for downwind (x) and crosswind (y) distances (meters)
# We use numpy to create the arrays and meshgrid
x = np.linspace(100, 5000, 100)   # 100m to 5km downwind
y = np.linspace(-1000, 1000, 100) # 1km crosswind
X, Y = np.meshgrid(x, y)

# --- 2. Calculate Dispersion Coefficients ---
# Assuming Moderately Stable Atmosphere (Pasquill Class F for night time)
# Using standard Briggs empirical formulas for rural conditions
sigma_y = 0.04 * X * (1 + 0.0001 * X)**(-0.5)
sigma_z = 0.016 * X * (1 + 0.0003 * X)**(-1)

# --- 3. Gaussian Plume Equation ---
# Equation simplified for ground level (z = 0)
Term1 = Q / (np.pi * u * sigma_y * sigma_z)
Term2 = np.exp(-(Y**2) / (2 * sigma_y**2))
Term3 = np.exp(-(H**2) / (2 * sigma_z**2))

C = Term1 * Term2 * Term3 # Concentration in kg/m^3
C_ppm = (C * 1e6 * 24.45) / 57.05 # Convert to ppm (approximate for MIC)

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))

# Create the filled contour plot
contour = plt.contourf(X, Y, C_ppm, levels=20, cmap='jet')
plt.colorbar(contour, label='Concentration (ppm)')

# Add titles and labels
plt.title('MIC Ground-Level Gas Dispersion Plume')
plt.xlabel('Downwind Distance, x (m)')
plt.ylabel('Crosswind Distance, y (m)')

# Overlay IDLH limit contour (MIC IDLH is ~5 ppm)
# We plot a specific contour line at the 5 ppm level
plt.contour(X, Y, C_ppm, levels=[5], colors='red', linewidths=2)

# Create a dummy plot to add the IDLH line to the legend
plt.plot([], [], color='red', linewidth=2, label='IDLH Limit (5 ppm)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
