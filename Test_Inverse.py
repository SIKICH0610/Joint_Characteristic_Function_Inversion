import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

# Joint CF for independent N(0,1) x N(0,1)
phi_joint = lambda s, t: np.exp(-0.5*(s**2 + t**2))

def prob_X_gt_a_and_Y_gt_0_via_cf(a, L=8.0, eps_s=1e-2, eps_t=1e-2, use_fejer=True):
    """
    Numerically compute P(X>a, Y>0) from the joint CF using damped kernels:
      ∫∫ φ(s,t) * [e^{-i a s}/(eps_s + i s)] * [1/(eps_t + i t)] ds dt / (4π^2)
    As eps_s, eps_t -> 0+, the value converges (in distribution) to the true probability.
    L is the symmetric truncation bound; Fejér taper reduces truncation ringing.
    """
    def fejer(u, L_):
        if not use_fejer:
            return 1.0
        a = np.abs(u)/L_
        return (1.0 - a) if a < 1.0 else 0.0

    def integrand(t, s):
        fac_x = np.exp(-1j*s*a) / (eps_s + 1j*s)  # indicator{x>a} via damped transform
        fac_y = 1.0 / (eps_t + 1j*t)              # indicator{y>0} via damped transform
        taper = fejer(s, L) * fejer(t, L)
        val = phi_joint(s, t) * fac_x * fac_y * taper
        return np.real(val)  # integrand is complex; probability is real

    val, err = dblquad(lambda tt, ss: integrand(tt, ss),
                       -L, L, lambda _: -L, lambda _: L,
                       epsabs=1e-8, epsrel=1e-6)
    return val / (4*np.pi**2)

def conditional_prob_X_gt_a_given_Y_gt_0(a, **kw):
    num = prob_X_gt_a_and_Y_gt_0_via_cf(a, **kw)
    den = 0.5  # P(Y>0) for mean-0 normal
    return num / den

# ---- quick check
if __name__ == "__main__":
    a = 0.7
    p_num = conditional_prob_X_gt_a_given_Y_gt_0(a, L=8.0, eps_s=1e-2, eps_t=1e-2, use_fejer=True)
    p_true = 1.0 - norm.cdf(a)  # independence implies P(X>a | Y>0) = P(X>a)
    rel_err = abs(p_num - p_true)/p_true
    print(f"Numeric  ≈ {p_num:.8f}")
    print(f"Analytic = {p_true:.8f}")
    print(f"RelErr   = {rel_err:.2e}")
