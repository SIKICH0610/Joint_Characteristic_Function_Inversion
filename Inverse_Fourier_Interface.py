import numpy as np
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from scipy.integrate import quad
import matplotlib.pyplot as plt

from Joint_Helper import CharacteristicFunctionInverter  # for from_conditional

class JointCFBase(ABC):
    """
    Shared interface for joint CF inversion with two concrete backends:
      - QuadJointCFInverter  (scipy.quad/dblquad)
      - GPJointCFInverter    (Gauss–Legendre + Gil–Pelaez)
    Concrete classes must implement:
      - marginal_pdf_Y
      - conditional_pdf_via_1d
      - conditional_probability_point
      - joint_pdf
    """

    def __init__(self, cf_x, cf_y, joint_phi, use_fejer=True):
        self.cf_x = cf_x
        self.cf_y = cf_y
        self.phi_joint = joint_phi  # φ_{X,Y}(s,t)
        self.Lx = cf_x.L
        self.Ly = cf_y.L
        self.use_fejer = bool(use_fejer)
        self.timing = False

    # ---------- lifecycle / utils ----------
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
        """Triangular (Fejér/Cesàro) taper."""
        return max(1.0 - abs(s) / L, 0.0) if self.use_fejer else 1.0

    @staticmethod
    def _build_phi_joint(cf_y, conditional_cf_given_y, y_support, p_y, damping_alpha):
        """
        Shared builder for φ_{X,Y}(s,t) = ∫ φ_{X|Y=y}(s) e^{i t y} f_Y(y) e^{-α y^2} dy.
        Integrates re/im separately for numerical robustness.
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
        return phi_joint

    @classmethod
    def from_conditional(cls, cf_y, conditional_cf_given_y,
                         y_support=(-10.0, 10.0), p_y=None, damping_alpha=0.01,
                         use_fejer=True):
        """
        Factory that works for any subclass: builds φ_{X,Y} and seeds Lx=Ly=cf_y.L
        """
        phi_joint = cls._build_phi_joint(cf_y, conditional_cf_given_y, y_support, p_y, damping_alpha)
        dummy_cf_x = CharacteristicFunctionInverter(phi=lambda s: 1.0/(1.0 + s*s),
                                                    integration_limit=cf_y.L)
        return cls(cf_x=dummy_cf_x, cf_y=cf_y, joint_phi=phi_joint, use_fejer=use_fejer)

    # ---------- fallback conditional CF slice (quad) ----------
    def conditional_cf_slice(self, y):
        """
        ψ_y(s) = (1/2π) ∫ e^{-i t y} φ_{X,Y}(s,t) dt  (complex)
        Provided as a shared fallback; backends may ignore it.
        """
        def psi(s):
            re, _ = quad(lambda t: np.real(np.exp(-1j*t*y) * self.phi_joint(s, t)),
                         -self.Ly, self.Ly, limit=300)
            im, _ = quad(lambda t: np.imag(np.exp(-1j*t*y) * self.phi_joint(s, t)),
                         -self.Ly, self.Ly, limit=300)
            return (re + 1j*im) / (2*np.pi)
        return psi

    # ---------- abstract primitives (must implement) ----------
    @abstractmethod
    def marginal_pdf_Y(self, y, **kwargs) -> float:
        ...

    @abstractmethod
    def conditional_pdf_via_1d(self, x, y, **kwargs) -> float:
        ...

    @abstractmethod
    def conditional_probability_point(self, a, y, **kwargs) -> float:
        ...

    @abstractmethod
    def joint_pdf(self, x, y, **kwargs) -> float:
        ...

    # ---------- shared region slicer (uses the primitives) ----------
    def conditional_probability_region(self, a, y_range, num_points=10, detailed=False, **kwargs) -> float:
        """
        P(X>a | Y∈[yL,yH]) ≈ (trapz_y p(y) fY(y)) / (trapz_y fY(y)),
        with p(y) = conditional_probability_point(a, y).
        """
        yL, yH = y_range
        ys = np.linspace(yL, yH, num_points)

        p_slices = np.empty_like(ys, dtype=float)
        fY_vals  = np.empty_like(ys, dtype=float)
        weighted = np.empty_like(ys, dtype=float)

        for i, y in enumerate(ys):
            t0 = time.perf_counter()
            p_y  = self.conditional_probability_point(a, y, **kwargs)
            fY_y = self.marginal_pdf_Y(y, **kwargs)
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

    # ---------- shared presentation ----------
    def show_expression(self, s=0.0, t=0.0, y_sample=0.0, x_test=0.0, a=0.0, **kwargs):
        print("=== Expression Inspection ===")
        if hasattr(self.cf_y, "expression_str") and self.cf_y.expression_str:
            print(f"Marginal CF φ_Y(t): {self.cf_y.expression_str}")
        else:
            print("Marginal CF φ_Y(t): (lambda, no expression string available)")
        print(f"  φ_Y({t}) = {self.cf_y.phi(t)}")

        print("\nJoint CF φ_{X,Y}(s,t):")
        print("  Defined as ∫ φ_{X|Y=y}(s) e^{i t y} f_Y(y) dy")
        print(f"  φ_{{X,Y}}({s},{t}) = {self.phi_joint(s, t)}")

        print("\nSample Conditional PDF f_{X|Y}(x|y):")
        pdf_val = self.conditional_pdf_via_1d(x_test, y_sample, **kwargs)
        print(f"  f_{{X|Y}}({x_test}|Y={y_sample}) = {pdf_val}")

        print("\nSample Conditional Probability P(X > a | Y=y):")
        tail = self.conditional_probability_point(a=a, y=y_sample, **kwargs)
        print(f"  P(X > {a} | Y={y_sample}) ≈ {tail}")

    def plot_conditional_pdf(self, y_values, x_range=(-5, 5), num_points=300, true_pdf=None, **kwargs):
        xs = np.linspace(*x_range, num_points)
        plt.figure(figsize=(8, 6))
        for y in y_values:
            vals = [self.conditional_pdf_via_1d(x, y, **kwargs) for x in xs]
            plt.plot(xs, vals, label=f"Reconstructed f(X|Y={y})")
            if true_pdf is not None:
                tv = [true_pdf(x, y) for x in xs]
                plt.plot(xs, tv, "--", label=f"True f(X|Y={y})")
        plt.xlabel("x"); plt.ylabel("Density"); plt.title("Conditional PDFs"); plt.legend(); plt.grid(True); plt.show()
