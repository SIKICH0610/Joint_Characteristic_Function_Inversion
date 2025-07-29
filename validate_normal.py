import numpy as np
from Joint_Helper import NormalCF
from scipy.stats import norm

# Normal Distribution Example
normal_cf = NormalCF()
x_vals = np.linspace(-4, 4, 300)
true_normal_pdf = norm.pdf(x_vals)

# Plot recovered PDF vs true PDF
normal_cf.plot_pdf(x_vals, true_pdf=true_normal_pdf, label='Recovered Normal PDF')

# Compute tail probability P(X > 1)
tail_prob = normal_cf.tail_probability(1, upper=8)
print(f"[Normal] P(X > 1) ≈ {tail_prob:.6f} (True: {1 - norm.cdf(1):.6f})")
