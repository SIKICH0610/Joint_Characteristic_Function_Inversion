import numpy as np
import time
from contextlib import contextmanager
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt

# only needed to seed L (frequency cutoff) for X
from Joint_Helper import CharacteristicFunctionInverter

class JointCharacteristicFunctionInverterQuad:
    """
    Reference implementation using scipy.quad/dblquad everywhere.
    Stable and simple; slower than the GL/GP variant.
    """
    def __init__(self, cf_x, cf_y, joint_phi, use_fejer=True):
        self.cf_x = cf_x
        self.cf_y = cf_y
        self.phi_joint = joint_phi
        self.Lx = cf_x.L
        self.Ly = cf_y.L
        self.use_fejer = use_fejer
        self.timing = False

    # ---------- utilities ----------
    def enable_timing(self, on: bool = True):
        self.timing = bool(on)

    @contextmanager
    def _timer(self, label: str):
        if not self.timing:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            print(f"[{label}] {time.perf_counter() - t0:.3f}s", flush=True)

    def _fejer_weight(self, s, L):
        return max(1 - abs(s) / L, 0.0) if self.use_fejer else 1.0

    @classmethod
    def from_conditional(cls, cf_y, conditional_cf_given_y,
                         y_support=(-10.0, 10.0), p_y=None, damping_alpha=0.01,
                         use_fejer=True):
        """
        Build φ_{X,Y}(s,t) = ∫ φ_{X|Y=y}(s) e^{i t y} f_Y(y) e^{-α y^2} dy
        Returns a ready-to-use inverter with Lx = Ly = cf_y.L
        """
        def phi_joint(s, t):
            def g(y):
                return (np.exp(1j * t * y)
                        * conditional_cf_given_y(y)(s)
                        * ((p_y(y) if p_y else 1.0))
                        * np.exp(-damping_alpha * y * y))
            re, _ = quad(lambda yy: np.real(g(yy)), *y_support, limit=300)
            im, _ = quad(lambda yy: np.imag(g(yy)), *y_support, limit=300)
            return re + 1j * im

        dummy_cf_x = CharacteristicFunctionInverter(phi=lambda s: 1/(1+s*s),
                                                    integration_limit=cf_y.L)
        return cls(cf_x=dummy_cf_x, cf_y=cf_y, joint_phi=phi_joint, use_fejer=use_fejer)

    # ---------- core inversions (quad) ----------
    def conditional_cf_slice(self, y):
        """ψ_y(s) = (1/2π) ∫ e^{-i t y} φ_{X,Y}(s,t) dt (complex)."""
        def psi(s):
            re, _ = quad(lambda t: np.real(np.exp(-1j*t*y) * self.phi_joint(s, t)),
                         -self.Ly, self.Ly, limit=300)
            im, _ = quad(lambda t: np.imag(np.exp(-1j*t*y) * self.phi_joint(s, t)),
                         -self.Ly, self.Ly, limit=300)
            return (re + 1j*im) / (2*np.pi)
        return psi

    def marginal_pdf_Y(self, y):
        """f_Y(y) via quad in t with optional Fejér taper."""
        def integrand(t):
            w_t = self._fejer_weight(t, self.Ly)
            return np.real(np.exp(-1j * t * y) * self.phi_joint(0.0, t) * w_t)
        val, _ = quad(integrand, -self.Ly, self.Ly, limit=300)
        return float(val / (2*np.pi))

    def conditional_pdf_via_1d(self, x, y, s_min=None, s_max=None, quad_limit=300):
        """
        f_{X|Y=y}(x) = (1/2π) ∫ e^{-isx} [ψ_y(s)/f_Y(y)] w_s ds
        """
        fY = self.marginal_pdf_Y(y)
        if fY < 1e-14:
            return 0.0

        psi = self.conditional_cf_slice(y)
        S = self.Lx
        s_lo = -S if s_min is None else s_min
        s_hi =  S if s_max is None else s_max

        def integrand(s):
            w_s = self._fejer_weight(s, self.Lx)
            return np.exp(-1j * s * x) * (psi(s)/fY) * w_s

        re, _ = quad(lambda s: np.real(integrand(s)), s_lo, s_hi, limit=quad_limit)
        im, _ = quad(lambda s: np.imag(integrand(s)), s_lo, s_hi, limit=quad_limit)
        return float(np.real((re + 1j*im) / (2*np.pi)))

    # ---------- probabilities ----------
    def conditional_probability_point(self, a, y, x_upper=10.0, quad_limit=300):
        """
        P(X>a | Y=y) = ∫_{a}^{x_upper} f_{X|Y=y}(x) dx  (1D via conditional_pdf_via_1d)
        """
        val, _ = quad(lambda x: self.conditional_pdf_via_1d(x, y),
                      a, x_upper, limit=quad_limit)
        return float(max(0.0, min(1.0, val)))

    def conditional_probability_region(self, a, y_range, num_points=10, x_upper=10.0, detailed=False):
        """
        Approximate P(X>a | Y∈[yL,yH]) with slices + trapezoid in y:
          ≈ (trapz_y p(y) fY(y)) / (trapz_y fY(y))
        where p(y) = ∫_a^{x_upper} f_{X|Y=y}(x) dx via 1D quad.
        """
        yL, yH = y_range
        ys = np.linspace(yL, yH, num_points)

        p_slices = np.empty_like(ys)
        fY_vals  = np.empty_like(ys)
        weighted = np.empty_like(ys)

        for i, y in enumerate(ys):
            t0 = time.perf_counter()
            p_y = self.conditional_probability_point(a, y, x_upper=x_upper)
            fY_y = self.marginal_pdf_Y(y)
            p_slices[i], fY_vals[i] = p_y, fY_y
            weighted[i] = p_y * fY_y
            dt = time.perf_counter() - t0
            if detailed:
                print(f"[{i+1}/{num_points}] y={y: .6f}: p≈{p_y:.6f}, fY≈{fY_y:.6f}, contrib≈{weighted[i]:.6f}  ({dt:.3f}s)")
            else:
                print(f"[{i+1}/{num_points}] finished slice (y={y:.6f}) in {dt:.3f}s", flush=True)

        num = np.trapz(weighted, ys)
        den = np.trapz(fY_vals, ys)
        return float(0.0 if den < 1e-14 else num/den)

    # ---------- joint pdf (quad dblquad) ----------
    def joint_pdf(self, x, y):
        """f_{X,Y}(x,y) via 2D quad with Fejér taper in both axes."""
        def integrand(t, s):
            w_s = self._fejer_weight(s, self.Lx)
            w_t = self._fejer_weight(t, self.Ly)
            return np.exp(-1j*(s*x + t*y)) * self.phi_joint(s, t) * w_s * w_t

        val_re, _ = dblquad(lambda tt, ss: np.real(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        val_im, _ = dblquad(lambda tt, ss: np.imag(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        return float(np.real((val_re + 1j*val_im) / (4*np.pi**2)))

    # ---------- helpers for debugging/plots ----------
    def show_expression(self, s=0.0, t=0.0, y_sample=0.0, x_test=0.0, a=0.0, x_upper=12.0):
        print("=== Expression Inspection (quad) ===")
        if hasattr(self.cf_y, "expression_str") and self.cf_y.expression_str:
            print(f"Marginal CF φ_Y(t): {self.cf_y.expression_str}")
        else:
            print("Marginal CF φ_Y(t): (lambda)")
        print(f"  φ_Y({t}) = {self.cf_y.phi(t)}")

        print("\nJoint CF φ_{X,Y}(s,t):")
        print("  Defined as ∫ φ_{X|Y}(s) e^{i t y} f_Y(y) dy")
        print(f"  φ_{{X,Y}}({s},{t}) = {self.phi_joint(s, t)}")

        print("\nConditional PDF at a point:")
        pdf_val = self.conditional_pdf_via_1d(x_test, y_sample)
        print(f"  f_{{X|Y}}({x_test}|Y={y_sample}) = {pdf_val}")

        print("\nTail probability via 1D pdf integral:")
        tail = self.conditional_probability_point(a=a, y=y_sample, x_upper=x_upper)
        print(f"  P(X > {a} | Y={y_sample}) ≈ {tail}")

    def plot_conditional_pdf(self, y_values, x_range=(-5,5), num_points=300, true_pdf=None):
        xs = np.linspace(*x_range, num_points)
        plt.figure(figsize=(8,6))
        for y in y_values:
            vals = [self.conditional_pdf_via_1d(x, y) for x in xs]
            plt.plot(xs, vals, label=f"Reconstructed f(X|Y={y})")
            if true_pdf is not None:
                tv = [true_pdf(x, y) for x in xs]
                plt.plot(xs, tv, '--', label=f"True f(X|Y={y})")
        plt.xlabel("x"); plt.ylabel("Density"); plt.title("Conditional PDFs (quad)"); plt.legend(); plt.grid(True); plt.show()
