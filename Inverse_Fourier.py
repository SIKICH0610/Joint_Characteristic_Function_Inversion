import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, expon, uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Joint_Helper import make_cf, make_conditional_cf, CharacteristicFunctionInverter, NormalCF
from numpy.fft import fftshift, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator

class JointCharacteristicFunctionInverter:
    def __init__(self, cf_x, cf_y, joint_phi, use_fejer=True):
        self.cf_x = cf_x
        self.cf_y = cf_y
        self.phi_joint = joint_phi
        self.Lx = cf_x.L
        self.Ly = cf_y.L
        self.use_fejer = use_fejer  # Toggle Fejér summation

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
            def g(y):
                return (np.exp(1j * t * y) *
                        conditional_cf_given_y(y)(s) *
                        (p_y(y) if p_y else 1.0) *
                        np.exp(-damping_alpha * y * y))

            re, _ = quad(lambda yy: np.real(g(yy)), *y_support, limit=200)
            im, _ = quad(lambda yy: np.imag(g(yy)), *y_support, limit=200)
            return re + 1j * im

        dummy_cf_x = CharacteristicFunctionInverter(phi=lambda s: 1 / (1 + s ** 2), integration_limit=cf_y.L)
        return cls(cf_x=dummy_cf_x, cf_y=cf_y, joint_phi=phi_joint)

    # --- NEW: compute ψ_y(s) = (1/2π) ∫ e^{-i t y} φ_XY(s,t) dt ---
    def conditional_cf_slice(self, y):
        def psi(s):
            # integrate complex integrand over t
            re, _ = quad(lambda t: np.real(np.exp(-1j * t * y) * self.phi_joint(s, t)), -self.Ly, self.Ly, limit=300)
            im, _ = quad(lambda t: np.imag(np.exp(-1j * t * y) * self.phi_joint(s, t)), -self.Ly, self.Ly, limit=300)
            return (re + 1j * im) / (2 * np.pi)

        return psi

    # --- NEW: 1D inversion in s to get f_{X|Y=y}(x) ---
    def conditional_pdf_via_1d(self, x, y):
        fY = self.marginal_pdf_Y(y)
        if fY < 1e-14:
            return 0.0
        psi = self.conditional_cf_slice(y)

        def integrand(s):
            w_s = self._fejer_weight(s, self.Lx)
            return np.exp(-1j * s * x) * (psi(s) / fY) * w_s

        re, _ = quad(lambda s: np.real(integrand(s)), -self.Lx, self.Lx, limit=300)
        im, _ = quad(lambda s: np.imag(integrand(s)), -self.Lx, self.Lx, limit=300)
        return np.real((re + 1j * im) / (2 * np.pi))

    def _fejer_weight(self, s, L):
        """Fejér kernel weight: (1 - |s|/L)"""
        return max(1 - abs(s) / L, 0) if self.use_fejer else 1.0

    def joint_pdf(self, x, y):
        def integrand(t, s):  # note: dblquad calls f(y,x): outer var first
            w_s = self._fejer_weight(s, self.Lx)
            w_t = self._fejer_weight(t, self.Ly)
            return np.exp(-1j * (s * x + t * y)) * self.phi_joint(s, t) * w_s * w_t

        # dblquad expects integrand(t, s): inner limits are s
        val_re, _ = dblquad(lambda tt, ss: np.real(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        val_im, _ = dblquad(lambda tt, ss: np.imag(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        return np.real((val_re + 1j * val_im) / (4 * np.pi ** 2))

    def marginal_pdf_Y(self, y):
        """Compute marginal PDF f_Y(y) with Fejér weighting"""
        def integrand(t):
            w_t = self._fejer_weight(t, self.Ly)
            return np.real(np.exp(-1j * t * y) * self.phi_joint(0, t) * w_t)
        val, _ = quad(integrand, -self.Ly, self.Ly, limit=300)
        print(2)
        return val / (2 * np.pi)

    def conditional_pdf_X_given_Y(self, x, y):
        """Compute conditional PDF f_{X|Y}(x|y) using joint and marginal"""
        joint = self.joint_pdf(x, y)
        marginal = self.marginal_pdf_Y(y)
        return 0.0 if marginal < 1e-12 else joint / marginal

    def conditional_probability(self, a, y, x_upper=5, damping_alpha=0.1):
        """
        Compute P(X > a | Y = y)         if y is a number,
        or     P(X > a | Y in y_range)   if y is a tuple (low, high)
        """
        if isinstance(y, tuple):
            y_low, y_high = y

            # Numerator: P(X > a and Y in [y_low, y_high])
            def joint_integrand(y_, x):
                return self.joint_pdf(x, y_)
            
            print(2)
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
            # integrand = lambda x: self.conditional_pdf_X_given_Y(x, y)
            def integrand(x):
                pdf_val = self.conditional_pdf_X_given_Y(x, y)
                return pdf_val * np.exp(-damping_alpha * x**2)  # Gaussian damping on conditional PDF
            print(4)
            prob, _ = quad(integrand, a, x_upper, epsabs=1e-8, epsrel=1e-6, limit=200)
            # prob, _ = quad(integrand, a, x_upper)
            # x_vals = np.linspace(a, x_upper, 2000)
            # pdf_vals = np.array([self.conditional_pdf_X_given_Y(x, y) for x in x_vals])
            # prob = np.trapz(pdf_vals, x_vals)
            return prob

    # def conditional_probability_point(self, a, y, x_upper=10):
    #     integrand = lambda x: self.conditional_pdf_X_given_Y(x, y)
    #     prob, _ = quad(integrand, a, x_upper)
    #     return prob

    def show_expression(self, s=0.0, t=0.0, y_sample=0.0):
        """
        Display the underlying CF and PDF structure for debugging.
        - Shows the expression of marginal CF, conditional CF, and joint CF.
        - Evaluates them at sample points to confirm they work.
        """
        print("=== Expression Inspection ===")

        # Marginal CF (Y)
        if hasattr(self.cf_y, "expression_str") and self.cf_y.expression_str:
            print(f"Marginal CF φ_Y(t): {self.cf_y.expression_str}")
        else:
            print("Marginal CF φ_Y(t): (lambda, no expression string available)")
        print(f"  φ_Y({t}) = {self.cf_y.phi(t)}")

        # Joint CF (s,t)
        print("\nJoint CF φ_{X,Y}(s,t):")
        print("  Defined as integral over Y of φ_{X|Y}(s) * exp(i t y) * f_Y(y)")
        print(f"  φ_{{X,Y}}({s},{t}) = {self.phi_joint(s,t)}")

        # Conditional PDF check
        print("\nSample Conditional PDF f_{X|Y}(x|y):")
        x_test = 0.0
        cond_pdf = self.conditional_pdf_X_given_Y(x_test, y_sample)
        print(cond_pdf)
        print(f"  f_{'{'}X|Y{'}'}({x_test}|Y={y_sample}) = {cond_pdf}")

    def plot_conditional_pdf(self, y_values, x_range=(-5, 5), num_points=300, true_pdf=None):
        """
        Visualize the conditional PDF f_{X|Y=y}(x) for given y values.
        
        Parameters:
            y_values: list of y points (e.g., [0, 1, -1]) for conditioning.
            x_range: tuple for X-axis (min, max).
            num_points: resolution for X-axis.
            true_pdf: optional callable true PDF f(x|y) for comparison (e.g., analytical normal pdf).
        """
        print(3)
        x_vals = np.linspace(*x_range, num_points)
        plt.figure(figsize=(8, 6))

        for y in y_values:
            pdf_vals = [self.conditional_pdf_X_given_Y(x, y) for x in x_vals]
            plt.plot(x_vals, pdf_vals, label=f"Reconstructed f(X|Y={y})")

            if true_pdf is not None:
                true_vals = [true_pdf(x, y) for x in x_vals]
                plt.plot(x_vals, true_vals, '--', label=f"True f(X|Y={y})")

        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Conditional PDFs f(X|Y=y)")
        plt.legend()
        plt.grid(True)
        plt.show()
