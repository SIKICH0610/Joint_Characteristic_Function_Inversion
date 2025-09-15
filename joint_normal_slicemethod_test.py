import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from Joint_Helper import make_cf, make_conditional_cf
from Inverse_Fourier_Interface import JointCharacteristicFunctionInverter

# Define Y ~ N(0,1), X|Y=y ~ N(2y,1)
cf_y = make_cf("normal", {"mu": 0, "sigma": 1})
conditional_cf = make_conditional_cf("normal", {"mean": "2*y", "var": 1.0})

joint_cf = JointCharacteristicFunctionInverter.from_conditional(
    cf_y=cf_y,
    conditional_cf_given_y=conditional_cf,
    y_support=(-10, 10),
    p_y=lambda y: norm.pdf(y, 0, 1),
    damping_alpha=0.01
)
joint_cf.use_fejer = True

# Region Y ∈ [-1,1]
y_range = (-1.0, 1.0)
print("check")
# === Only test with 10 slices ===
p_numeric = joint_cf.conditional_probability_region_via_slices(
    a=0.0,
    y_range=(-1.0, 1.0),
    num_points=5,
    x_upper=12.0
)

print("Numeric region probability with 10 slices =", p_numeric)


# Analytical baseline
num, _ = quad(lambda yy: norm.cdf(2.0 * yy) * norm.pdf(yy), y_range[0], y_range[1], limit=500)
den = norm.cdf(y_range[1]) - norm.cdf(y_range[0])
p_analytical = num / den

print(f"[num_points=10] numeric={p_numeric:.6f}, analytical={p_analytical:.6f}, "
      f"abs_err={abs(p_numeric - p_analytical):.3e}")
