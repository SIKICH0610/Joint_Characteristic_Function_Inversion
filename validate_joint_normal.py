import numpy as np
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
    y_support=(-5, 5),                      # integration support for Y
    p_y=lambda y: norm.pdf(y, 0, 1),        # PDF of Y
    damping_alpha=0.01                      # small damping for stability
)

print(1)
# Step 4: Compute P(X > 0 | Y = a)
a_values = [0, 1, -1]
for y_val in a_values:
    prob = joint_cf.conditional_probability(a=0.0, y=y_val)   # P(X>0 | Y=y)
    print(f"P(X > 0 | Y={y_val}) = {prob:.6f}")

# Analytical check: P(X > 0 | Y=y) = Phi(2*y)
for y_val in a_values:
    analytical = 1 - norm.cdf(0, loc=2*y_val, scale=1)
    print(f"Analytical: P(X > 0 | Y={y_val}) = {analytical:.6f}")
