import numpy as np
from scipy.stats import uniform
from Joint_Helper import UniformCF

# Uniform Distribution Example
uniform_cf = UniformCF(a=0, b=2)
x_vals = np.linspace(-1, 3, 400)
true_uniform_pdf = uniform.pdf(x_vals, loc=0, scale=2)

# Plot recovered PDF vs true PDF
uniform_cf.plot_pdf(x_vals, true_pdf=true_uniform_pdf, label='Recovered Uniform PDF')

# Compute tail probability P(X > 1.5)
tail_prob_uni = uniform_cf.tail_probability(1.5, upper=3)
print(f"[Uniform] P(X > 1.5) ≈ {tail_prob_uni:.6f} (True: {1 - uniform.cdf(1.5, loc=0, scale=2):.6f})")
