import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
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

# >>> NEW: route conditional density calls through the 1D slice-then-invert
joint_cf.conditional_pdf_X_given_Y = joint_cf.conditional_pdf_via_1d

# Step 4: Visualization of reconstructed conditional PDFs
y_values = [0, 1, -1]  # Y points for conditioning
true_conditional_pdf = lambda x, y: norm.pdf(x, loc=2*y, scale=1.0)

print("\n=== Plotting Conditional PDFs for Visual Check ===")
joint_cf.show_expression()
joint_cf.plot_conditional_pdf(
    y_values=y_values,
    x_range=(-10, 10),
    true_pdf=true_conditional_pdf
)

# Step 5: Compute P(X > 0 | Y=y) numerically (no x-domain damping → unbiased)
print("\n=== Numeric vs Analytical: P(X>0 | Y=y) ===")
for y_val in y_values:  # (fixed: was a_values)
    prob_numeric = joint_cf.conditional_probability(a=0.0, y=y_val, x_upper=10, damping_alpha=0.0)
    prob_analytical = norm.cdf(2 * y_val)   # since X|Y=y ~ N(2y, 1)
    print(f"y={y_val:>3}: numeric={prob_numeric:.6f} | analytical={prob_analytical:.6f}")