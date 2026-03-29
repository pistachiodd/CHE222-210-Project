import numpy as np

#fixed parameters
Tss = 302.6513
Css = 1.3316

Hr = -80.4e3
A = 4.13e8
R = 8.314
Ea = 65.4e3

#baseline values (from your system)
m = 100
F_base = 1
Cp_base = 4200
L_base = 700

#function to compute system behavior
def compute_system(L, F, Cp):
    exp_term = np.exp(-Ea / (R * Tss))

    a11 = -A * exp_term - F
    a12 = A * exp_term * Ea / (R * Tss**2) * Css
    a21 = (-Hr * m * A * exp_term) / (m * Cp)

    a22 = (
        -Hr * m * A * exp_term * Ea / (R * Tss**2) * Css
        - F * Cp
        - L
    ) / m * Cp

    b2 = F / m

    gain = a12 * b2
    tau = np.sqrt(1 / (a11 * a22 - a12 * a21))

    return gain, tau


# -----------------------------
# test 1: vary heat removal (L)
# -----------------------------
print("\nchanging L (cooling strength):")
for L in [700, 1000, 1500, 2000]:
    gain, tau = compute_system(L, F_base, Cp_base)
    print(f"L = {L} -> tau = {tau:.4f}, gain = {gain:.3e}")


# -----------------------------
# test 2: vary flow rate (F)
# -----------------------------
print("\nchanging F (flow rate):")
for F in [0.5, 1, 2, 3]:
    gain, tau = compute_system(L_base, F, Cp_base)
    print(f"F = {F} -> tau = {tau:.4f}, gain = {gain:.3e}")


# -----------------------------
# test 3: vary heat capacity (Cp)
# -----------------------------
print("\nchanging Cp (thermal buffering):")
for Cp in [2000, 4200, 8000]:
    gain, tau = compute_system(L_base, F_base, Cp)
    print(f"Cp = {Cp} -> tau = {tau:.4f}, gain = {gain:.3e}")
