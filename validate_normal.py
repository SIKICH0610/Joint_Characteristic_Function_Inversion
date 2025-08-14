from scipy.stats import norm
import numpy as np
from Joint_Helper import NormalCF

# Create Normal CF inverter
normal_cf = NormalCF()

# Define range for plotting
x_vals = np.linspace(-4, 4, 300)
true_normal_pdf = norm.pdf(x_vals, loc=0, scale=1)

# Plot recovered PDF vs true PDF with LaTeX distribution expression
normal_cf.plot_pdf(
    x_vals,
    true_pdf=true_normal_pdf,
    label='Recovered Normal PDF',
    dist_expression=r"$N(0,1)$"
)

# Compute tail probability P(X > 1)
tail_prob = normal_cf.tail_probability(1, upper=8)
print(f"[Normal] P(X > 1) ≈ {tail_prob:.6f} (True: {1 - norm.cdf(1):.6f})")
