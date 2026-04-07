import numpy as np
import matplotlib.pyplot as plt

# define model parameters 
Q = 50           # emission rate of mic (kg/s) - rough estimate
u = 2.5          # wind speed (m/s) - light wind
H = 10           # release height (m) - tank/stack height

# grid setup for downwind (x) and crosswind (y)
# just making a mesh so we can evaluate everywhere
x = np.linspace(100, 5000, 100)    # 100 m to 5 km downwind
y = np.linspace(-1000, 1000, 100)  # +/- 1 km crosswind
X, Y = np.meshgrid(x, y)


# dispersion coefficients 
# assuming moderately stable atmosphere (class f, like night conditions)
# using briggs empirical correlations
sigma_y = 0.04 * X * (1 + 0.0001 * X)**(-0.5)
sigma_z = 0.016 * X * (1 + 0.0003 * X)**(-1)


# gaussian plume equation 
# simplified at ground level (z = 0)

# main prefactor (source strength + spreading)
Term1 = Q / (np.pi * u * sigma_y * sigma_z)

# crosswind spreading
Term2 = np.exp(-(Y**2) / (2 * sigma_y**2))

# vertical dispersion + stack height effect
Term3 = np.exp(-(H**2) / (2 * sigma_z**2))

# final concentration (kg/m^3)
C = Term1 * Term2 * Term3

# convert to ppm (approx, using ideal gas assumption)
C_ppm = (C * 1e6 * 24.45) / 57.05


# visualization 
plt.figure(figsize=(10, 6))

# filled contour plot (this is the actual plume)
contour = plt.contourf(X, Y, C_ppm, levels=20, cmap='jet')
plt.colorbar(contour, label='Concentration (ppm)')

# labels and title
plt.title('MIC Ground-Level Gas Dispersion Plume')
plt.xlabel('Downwind Distance, x (m)')
plt.ylabel('Crosswind Distance, y (m)')

# overlay idlh limit (~5 ppm for mic)
# this shows the dangerous region boundary
plt.contour(X, Y, C_ppm, levels=[5], colors='red', linewidths=2)

# dummy plot just so the legend shows the red line
plt.plot([], [], color='red', linewidth=2, label='IDLH Limit (5 ppm)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
