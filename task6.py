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


print("transfer function coefficients:")
print("a11 =", a11)
print("a12 =", a12)
print("a21 =", a21)
print("a22 =", a22)
print("b2  =", b2)

print("\nsystem gain =", gain)
print("time constant (tau) =", tau)
