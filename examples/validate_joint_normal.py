import numpy as np
from scipy.stats import norm
from Joint_Helper import make_cf, make_conditional_cf
from JointCharacteristicFunctionInverter import JointCharacteristicFunctionInverter

# Define Y ~ Normal(0,1)
cf_y = make_cf("normal", {"mu": 0, "sigma": 1})

# Define X | Y=y ~ Normal(2y, 1)
conditional_cf = make_conditional_cf("normal", {"mean": "2*y", "var": 1.0})

# Build joint CF inverter
joint_cf = JointCharacteristicFunctionInverter.from_conditional(
    cf_y=cf_y,
    conditional_cf_given_y=conditional_cf,
    y_support=(-5, 5),
    p_y=lambda y: norm.pdf(y, 0, 1),
    damping_alpha=0.01
)

# Plot joint PDF (contour and 3D)
joint_cf.plot_joint_pdf(x_range=(-5, 5), y_range=(-5, 5), num_points=150, plot_type='contour')
joint_cf.plot_joint_pdf(x_range=(-5, 5), y_range=(-5, 5), num_points=80, plot_type='surface')
