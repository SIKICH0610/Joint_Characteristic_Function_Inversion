import numpy as np
from scipy.stats import expon
from Joint_Helper import ExponentialCF

# Exponential Distribution Example
exp_cf = ExponentialCF(lam=1.0)
x_vals = np.linspace(0, 8, 300)
true_exp_pdf = expon.pdf(x_vals, scale=1.0)

# Plot recovered PDF vs true PDF
exp_cf.plot_pdf(x_vals, true_pdf=true_exp_pdf, label='Recovered Exponential PDF')

# Compute tail probability P(X > 2)
tail_prob_exp = exp_cf.tail_probability(2, upper=20)
print(f"[Exponential] P(X > 2) ≈ {tail_prob_exp:.6f} (True: {1 - expon.cdf(2):.6f})")
