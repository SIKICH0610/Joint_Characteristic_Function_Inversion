import numpy as np
import time
from contextlib import contextmanager
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt

# only needed to seed L (frequency cutoff) for X
from Joint_Helper import CharacteristicFunctionInverter

class JointCharacteristicFunctionInverterGP:
    """
    Fast implementation:
      - ψ_y(s) and f_Y(y) via Gauss–Legendre (fixed nodes) + Fejér taper
      - Tail P(X>a|Y=y) via Gil–Pelaez with GL nodes on (0, Lx)
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

    @staticmethod
    def _gl_nodes_sym(L, N):
        """Gauss–Legendre nodes/weights on [-L, L]."""
        z, w = np.polynomial.legendre.leggauss(N)
        return L*z, L*w

    @staticmethod
    def _gl_nodes_pos(L, N):
        """Gauss–Legendre nodes/weights on (0, L] (exclude 0 so /s is safe)."""
        z, w = np.polynomial.legendre.leggauss(N)  # z in (-1,1)
        nodes = 0.5*(z + 1.0) * L
        weights = 0.5*L*w
        return nodes, weights

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

    # ---------- fast ψ_y and f_Y ----------
    def _psi_y_gl(self, y, s_vals, Nt=96):
        """
        Vectorized ψ_y(s) using fixed t-nodes (GL on [-Ly, Ly]) with Fejér taper.
        """
        Ns = np.atleast_1d(s_vals).size
        with self._timer(f"psi_y_gl(y={y:.3g}, Nt={Nt}, Ns={Ns})"):
            t_nodes, t_w = self._gl_nodes_sym(self.Ly, Nt)
            w_taper = (np.maximum(1.0 - np.abs(t_nodes)/self.Ly, 0.0) if self.use_fejer else 1.0)
            base = np.exp(-1j * t_nodes * y) * t_w * w_taper

            s_vals = np.atleast_1d(s_vals)
            try:
                Phi = self.phi_joint(s_vals[:, None], t_nodes[None, :])
            except Exception:
                Phi = np.empty((s_vals.size, t_nodes.size), dtype=complex)
                for i, s in enumerate(s_vals):
                    try:
                        Phi[i, :] = self.phi_joint(s, t_nodes)
                    except Exception:
                        Phi[i, :] = np.array([self.phi_joint(s, tj) for tj in t_nodes], dtype=complex)

            psi = (Phi * base[None, :]).sum(axis=1) / (2*np.pi)
            return psi

    def marginal_pdf_Y(self, y, Nt=96):
        """Fast f_Y(y) via GL nodes in t with Fejér taper."""
        with self._timer(f"marginal_pdf_Y(y={y:.3g}, Nt={Nt})"):
            t_nodes, t_w = self._gl_nodes_sym(self.Ly, Nt)
            w_taper = (np.maximum(1.0 - np.abs(t_nodes)/self.Ly, 0.0) if self.use_fejer else 1.0)
            try:
                phiY = self.phi_joint(0.0, t_nodes)
            except Exception:
                phiY = np.array([self.phi_joint(0.0, tt) for tt in t_nodes], dtype=complex)
            val = np.real(np.exp(-1j*t_nodes*y) * phiY * w_taper) @ t_w
            return float(val / (2*np.pi))

    # ---------- conditional pdf (fast) ----------
    def conditional_pdf_via_1d(self, x, y, Ns=96, Nt=96):
        """
        f_{X|Y=y}(x) via fixed-node ψ_y(s) + GL in s (no nested quad).
        """
        fY = self.marginal_pdf_Y(y, Nt=Nt)
        if fY < 1e-14:
            return 0.0

        s_nodes, s_w = self._gl_nodes_sym(self.Lx, Ns)
        w_s = (np.maximum(1.0 - np.abs(s_nodes)/self.Lx, 0.0) if self.use_fejer else 1.0)

        psi = self._psi_y_gl(y, s_nodes, Nt=Nt)
        integrand = np.exp(-1j * s_nodes * x) * (psi / fY) * w_s
        val = np.real(integrand) @ s_w
        return float(val / (2*np.pi))

    # ---------- probabilities (fast GP) ----------
    def conditional_probability_point(self, a, y, Ns=256, Nt=96):
        """
        Fast Gil–Pelaez tail:
          P(X>a|Y=y) = 1/2 + (1/π) ∫_0^{Lx} Im[e^{-i s a} φ_{X|Y=y}(s)/s] ds
        with GL nodes on (0, Lx) (no singularity at s=0).
        """
        with self._timer(f"P(X>{a}|Y={y}) via GP (Ns={Ns}, Nt={Nt})"):
            fY = self.marginal_pdf_Y(y, Nt=Nt)
            if fY < 1e-14:
                return 0.0
            s_pos, w_pos = self._gl_nodes_pos(self.Lx, Ns)
            w_s = ((1.0 - s_pos/self.Lx) if self.use_fejer else 1.0)
            psi = self._psi_y_gl(y, s_pos, Nt=Nt)
            phi_cond = psi / fY
            integrand = np.imag(np.exp(-1j * s_pos * a) * (phi_cond / s_pos)) * w_s
            val = (integrand * w_pos).sum()
            return float(np.clip(0.5 + val/np.pi, 0.0, 1.0))

    def conditional_probability_region(self, a, y_range, num_points=10, Ns=256, Nt=96, detailed=False):
        """
        Region conditional via slices using fast GP per slice:
          ≈ (trapz_y p(y) fY(y)) / (trapz_y fY(y))
        """
        yL, yH = y_range
        ys = np.linspace(yL, yH, num_points)

        p_slices = np.empty_like(ys)
        fY_vals  = np.empty_like(ys)
        weighted = np.empty_like(ys)

        for i, y in enumerate(ys):
            t0 = time.perf_counter()
            p_y  = self.conditional_probability_point(a, y, Ns=Ns, Nt=Nt)
            fY_y = self.marginal_pdf_Y(y, Nt=Nt)
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

    def joint_pdf(self, x, y):
        """f_{X,Y}(x,y) via 2D quad with Fejér taper (rarely needed here)."""
        def integrand(t, s):
            w_s = self._fejer_weight(s, self.Lx)
            w_t = self._fejer_weight(t, self.Ly)
            return np.exp(-1j*(s*x + t*y)) * self.phi_joint(s, t) * w_s * w_t
        val_re, _ = dblquad(lambda tt, ss: np.real(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        val_im, _ = dblquad(lambda tt, ss: np.imag(integrand(tt, ss)),
                            -self.Ly, self.Ly, lambda _: -self.Lx, lambda _: self.Lx)
        return float(np.real((val_re + 1j*val_im) / (4*np.pi**2)))

    # ---------- debug/plots ----------
    def show_expression(self, s=0.0, t=0.0, y_sample=0.0, x_test=0.0, a=0.0):
        print("=== Expression Inspection (GP) ===")
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

        print("\nTail probability via Gil–Pelaez:")
        tail = self.conditional_probability_point(a=a, y=y_sample)
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
        plt.xlabel("x"); plt.ylabel("Density"); plt.title("Conditional PDFs (GP)"); plt.legend(); plt.grid(True); plt.show()
