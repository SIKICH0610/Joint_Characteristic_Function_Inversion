from scipy.stats import norm
from Joint_Helper import make_cf, make_conditional_cf
from GP_Method import JointCharacteristicFunctionInverterGP as JointCharacteristicFunctionInverter

def main():
    # ---- Y ~ N(0,1) ----
    cf_y = make_cf("normal", {"mu": 0.0, "sigma": 1.0})

    # ---- X | Y=y ~ N( mu(y), 1 ), with nonlinear mean mu(y) = 1 + 0.8*y + 0.5*y^2
    # make_conditional_cf expects a string in y for the mean when using "normal"
    conditional_cf = make_conditional_cf("normal", {"mean": "1 + 0.8*y + 0.5*y**2", "var": 1.0})

    # ---- Build joint CF numerically from conditional + p_Y (with light damping on y-integral)
    J = JointCharacteristicFunctionInverter.from_conditional(
        cf_y=cf_y,
        conditional_cf_given_y=conditional_cf,
        y_support=(-8.0, 8.0),
        p_y=lambda y: norm.pdf(y, 0.0, 1.0),
        damping_alpha=0.02,
    )
    J.use_fejer = True
    # Optional: enlarge frequency box if needed
    # J.Lx = max(J.Lx, 10.0); J.Ly = max(J.Ly, 10.0)

    # ---- Case test: P(X>0 | Y=y) at several y (including y=0)
    a = 0.0
    ys = [0]

    print("=== Complex case: X|Y=y ~ N(1 + 0.8y + 0.5y^2, 1),  Y~N(0,1) ===")
    print("Target: P(X>0 | Y=y)  (numeric via CF slice)  vs  analytic Φ(μ(y))")
    for y in ys:
        # Fast CF-slice tail (requires your *_fast methods; if not present, use your adaptive method)
        p_num = J.conditional_probability_point(a=a, y=y, Ns=384, Nt=160)
        mu_y = 1.0 + 0.8*y + 0.5*(y**2)
        p_true = norm.cdf(mu_y)  # σ=1 ⇒ Φ(μ(y))
        print(f"y={y:>5.2f}  numeric={p_num:.8f}  analytic={p_true:.8f}  abs_err={abs(p_num - p_true):.3e}")

    # ---- Single-focus check at y=0 (harder than the linear case since μ(0)=1 → not 0.5)
    y0 = 0.0
    p_num0 = J.conditional_probability_point(a=0.0, y=y0, Ns=384, Nt=160)
    p_true0 = norm.cdf(1.0)  # ≈ 0.841344746...
    print("\n=== Focus: y=0 ===")
    print(f"P(X>0 | Y=0): numeric={p_num0:.8f}  analytic={p_true0:.8f}  abs_err={abs(p_num0 - p_true0):.3e}")

if __name__ == "__main__":
    main()
