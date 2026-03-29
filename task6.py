# TAKEN DIRECTLY FROM OTHER GROUP

import numpy as np

# Given Parameters (from your earlier tasks)
L = 700
Tss = 302.6513
Css = 1.3316

Hr = -80.4e3
A = 4.13e8
R = 8.314
Ea = 65.4e3

# From Task 1 (IMPORTANT - now correctly defined)
m = 100
F = 1
Cp = 4200

# Common exponential term (cleaner + avoids repetition)
exp_term = np.exp(-Ea / (R * Tss))

# Transfer Function Coefficients
a11 = -A * exp_term - F

a12 = A * exp_term * Ea / (R * Tss**2) * Css

a21 = (-Hr * m * A * exp_term) / (m * Cp)

a22 = (
    -Hr * m * A * exp_term * Ea / (R * Tss**2) * Css
    - F * Cp
    - L
) / m * Cp

b2 = F / m

# Gain and Time Constant
gain = a12 * b2

tau = np.sqrt(1 / (a11 * a22 - a12 * a21))

# Display results
print(
    "Transfer Function Coefficients:\n"
    f"a11 = {a11:.4e}\n"
    f"a12 = {a12:.4e}\n"
    f"a21 = {a21:.4e}\n"
    f"a22 = {a22:.4e}\n"
    f"b2  = {b2:.4e}\n\n"
    f"System Gain = {gain:.4e}\n"
    f"Time Constant (Tau) = {tau:.4f} s\n"
)