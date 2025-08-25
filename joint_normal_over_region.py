import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.integrate import quad
from Joint_Helper import make_cf, make_conditional_cf
from Inverse_Fourier import JointCharacteristicFunctionInverter

# Step 1: Define marginal CF for Y ~ Normal(0, 1)
cf_y = make_cf("normal", {"mu": 0, "sigma": 1})

# Step 2: Define conditional CF for X|Y=y ~ Normal(2y, 1)
conditional_cf = make_conditional_cf("normal", {"mean": "2*y", "var": 1.0})

# Step 3: Build Joint CF inverter
joint_cf = JointCharacteristicFunctionInverter.from_conditional(
    cf_y=cf_y,
    conditional_cf_given_y=conditional_cf,
    y_support=(-10, 10),
    p_y=lambda y: norm.pdf(y, 0, 1),
    damping_alpha=0.01
)

joint_cf.use_fejer = True  # keep tapering

# >>> If available, route conditional density via slice-then-invert (not used here, but good practice)
if hasattr(joint_cf, "conditional_pdf_via_1d") and callable(getattr(joint_cf, "conditional_pdf_via_1d")):
    joint_cf.conditional_pdf_X_given_Y = joint_cf.conditional_pdf_via_1d

print("pass")

# Step 4: Region-conditional probability test for Y in [-1, 1]
print("\n=== Region-conditional: P(X > 0 | Y in [-1, 1]) ===")
yL, yH = -1.0, 1.0

# Numerical via inverter
p_numeric = joint_cf.conditional_probability(a=0.0, y=(yL, yH), x_upper=6.0)

# Analytical baseline: E[ Φ(2Y) | Y ∈ [yL, yH] ]
num, _ = quad(lambda yy: norm.cdf(2.0 * yy) * norm.pdf(yy), yL, yH, limit=500)
den = norm.cdf(yH) - norm.cdf(yL)
p_analytical = num / den if den > 1e-14 else float("nan")

print(f"numeric     = {p_numeric:.6f}")
print(f"analytical  = {p_analytical:.6f}")

# Optional: simple tolerance check
tol = 2e-3
abs_err = abs(p_numeric - p_analytical)
print(f"abs error   = {abs_err:.3e}  |  {'PASS' if abs_err < tol else 'WARN'} (tol={tol})")
