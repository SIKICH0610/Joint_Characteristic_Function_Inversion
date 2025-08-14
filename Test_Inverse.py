from sympy import symbols, exp, I, pi, simplify, inverse_fourier_transform, init_printing

# Pretty printing
init_printing()

# Define variables
s, t, x, y = symbols('s t x y', real=True)

# Plug in a=1, b=0.5, c=1
phi_st = exp(-0.5 * (s**2 + t**2 + s*t))

# Step 1: Invert in s
f_xt = inverse_fourier_transform(phi_st * exp(-I * s * x), s, x)

# Step 2: Invert in t
f_xy = inverse_fourier_transform(f_xt * exp(-I * t * y), t, y)

# Simplify final result
f_xy_simplified = simplify(f_xy)
print(f_xy_simplified)
