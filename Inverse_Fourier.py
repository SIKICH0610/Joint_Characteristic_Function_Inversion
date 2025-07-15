import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, expon, uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Joint_Helper import make_cf, make_conditional_cf, CharacteristicFunctionInverter, NormalCF

class JointCharacteristicFunctionInverter:
    def __init__(self, cf_x, cf_y, joint_phi):
        self.cf_x = cf_x
        self.cf_y = cf_y
        self.phi_joint = joint_phi
        self.Lx = cf_x.L
        self.Ly = cf_y.L

    @classmethod
    def from_conditional(cls, cf_y, conditional_cf_given_y, y_support=(0, 20), p_y=None, damping_alpha=0.01):
        """
        Build joint CF from:
        - cf_y: CharacteristicFunctionInverter for Y
        - conditional_cf_given_y: function y ↦ φ_{X|Y=y}(s)
        - y_support: integration support for Y
        - p_y: PDF of Y (optional), required if not known
        - damping_alpha: controls Gaussian damping e^{-α y^2}
        """

        def phi_joint(s, t):
            integrand = lambda y: np.real(
                np.exp(1j * t * y) *
                conditional_cf_given_y(y)(s) *
                (p_y(y) if p_y else 1.0) *
                np.exp(-damping_alpha * y ** 2)  # <--- damping here
            )
            val, _ = quad(integrand, *y_support, limit=200)
            return val

        dummy_cf_x = CharacteristicFunctionInverter(
            phi=lambda s: 1 / (1 + s ** 2),  # dummy
            integration_limit=cf_y.L
        )

        return cls(cf_x=dummy_cf_x, cf_y=cf_y, joint_phi=phi_joint)

    def joint_pdf(self, x, y):
        integrand = lambda s, t: np.real(
            np.exp(-1j * (s * x + t * y)) * self.phi_joint(s, t)
        )
        val, _ = dblquad(integrand, -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        return val / (4 * np.pi**2)

    def marginal_pdf_Y(self, y):
        integrand = lambda t: np.real(np.exp(-1j * t * y) * self.phi_joint(0, t))
        val, _ = quad(integrand, -self.Ly, self.Ly)
        return val / (2 * np.pi)

    def conditional_pdf_X_given_Y(self, x, y):
        joint = self.joint_pdf(x, y)
        marginal = self.marginal_pdf_Y(y)
        if marginal < 1e-12:
            return 0.0
        return joint / marginal

    def conditional_probability(self, a, y, x_upper=10):
        """
        Compute P(X > a | Y = y)         if y is a number,
        or     P(X > a | Y in y_range)   if y is a tuple (low, high)
        """
        if isinstance(y, tuple):
            y_low, y_high = y

            # Numerator: P(X > a and Y in [y_low, y_high])
            def joint_integrand(y_, x):
                return self.joint_pdf(x, y_)

            joint_prob, _ = dblquad(joint_integrand,
                                    y_low, y_high,
                                    lambda _: a, lambda _: x_upper)

            # Denominator: P(Y in [y_low, y_high])
            def marginal_y(y_):
                return self.marginal_pdf_Y(y_)

            marginal_prob, _ = quad(marginal_y, y_low, y_high)

            if marginal_prob < 1e-12:
                return 0.0

            return joint_prob / marginal_prob

        else:
            # y is a single float value — point conditioning
            integrand = lambda x: self.conditional_pdf_X_given_Y(x, y)
            prob, _ = quad(integrand, a, x_upper)
            return prob

    # def conditional_probability_point(self, a, y, x_upper=10):
    #     integrand = lambda x: self.conditional_pdf_X_given_Y(x, y)
    #     prob, _ = quad(integrand, a, x_upper)
    #     return prob

    def plot_joint_pdf(self, x_range, y_range, num_points=100, plot_type='contour'):
        """
        Plot the joint PDF over a 2D grid.

        plot_type: 'contour' or 'surface'
        """
        x_vals = np.linspace(*x_range, num_points)
        y_vals = np.linspace(*y_range, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        # Compute joint_pdf at each grid point
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.joint_pdf(X[i, j], Y[i, j])

        if plot_type == 'contour':
            plt.figure(figsize=(8, 6))
            cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Joint PDF f(x, y)')
            plt.colorbar(cp, label='Density')
            plt.show()

        elif plot_type == 'surface':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Density')
            ax.set_title('Joint PDF f(x, y)')
            plt.show()

# ==== TEST ====

# Normal Example
# normal_cf = NormalCF()
# x_vals = np.linspace(-4, 4, 300)
# true_normal = norm.pdf(x_vals)
# normal_cf.plot_pdf(x_vals, true_pdf=true_normal, label='From Normal CF')
#
# # Compute P(X > 1)
# tail = normal_cf.tail_probability(1, upper=8)
# print(f"[Normal] P(X > 1) ≈ {tail:.6f} (true: {1 - norm.cdf(1):.6f})")

# # Exponential Example
# exp_cf = ExponentialCF(lam=1.0)
# x_vals_exp = np.linspace(0, 8, 300)
# true_exp = expon.pdf(x_vals_exp, scale=1.0)
# exp_cf.plot_pdf(x_vals_exp, true_pdf=true_exp, label='From Exponential CF')
#
# # Compute P(X > 2)
# tail_exp = exp_cf.tail_probability(2, upper=20)
# print(f"[Exponential] P(X > 2) ≈ {tail_exp:.6f} (true: {1 - expon.cdf(2):.6f})")
#
# # Uniform Example
# uniform_cf = UniformCF(a=0, b=2)
# x_vals_uniform = np.linspace(-1, 3, 400)
#
# true_uniform = uniform.pdf(x_vals_uniform, loc=0, scale=2)
# uniform_cf.plot_pdf(x_vals_uniform, true_pdf=true_uniform, label='From Uniform CF')
#
# # Compute P(X > 1.5)
# tail_uni = uniform_cf.tail_probability(1.5, upper=3)
# print(f"[Uniform] P(X > 1.5) ≈ {tail_uni:.6f} (true: {1 - uniform.cdf(1.5, loc=0, scale=2):.6f})")

cf_y = make_cf("normal", {"mu": 0, "sigma": 1})
# Create conditional CF factory: y ↦ φ_{X|Y=y}(s)
conditional_cf = make_conditional_cf("normal", {"mean": "2*y", "var": 1.0})
conditional_cf.show()
# Provide Y support and its PDF
y_support = (-5, 5)
p_y = lambda y: norm.pdf(y, loc=0, scale=1)

# Build the joint characteristic function inverter

joint_cf = JointCharacteristicFunctionInverter.from_conditional(
    cf_y=cf_y,
    conditional_cf_given_y=conditional_cf,
    y_support=(-5, 5),
    p_y=lambda y: norm.pdf(y, 0, 1),
    damping_alpha=1  # or experiment with 0.001, 0.1
)

joint_cf.plot_joint_pdf(x_range=(-5, 5), y_range=(-5, 5), num_points=200, plot_type='contour')
joint_cf.plot_joint_pdf(x_range=(-5, 5), y_range=(-5, 5), num_points=100, plot_type='surface')
