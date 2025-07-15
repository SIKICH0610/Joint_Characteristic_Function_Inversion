import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad

class CharacteristicFunctionInverter:
    def __init__(self, phi, integration_limit=50, window_alpha=0.01, expression_str=None):
        self.phi = phi  # characteristic function
        self.L = integration_limit
        self.alpha = window_alpha  # damping parameter
        self.expression_str = expression_str  # optional string for display

    def pdf(self, x):
        integrand = lambda t: np.real(np.exp(-1j * t * x) * self.phi(t) * np.exp(-self.alpha * t**2))
        val, _ = quad(integrand, -self.L, self.L, limit=200)
        return val / (2 * np.pi)

    def pdf_grid(self, x_vals):
        return np.array([self.pdf(x) for x in x_vals])

    def plot_pdf(self, x_vals, true_pdf=None, label='Recovered PDF'):
        approx_pdf = self.pdf_grid(x_vals)

        plt.plot(x_vals, approx_pdf, label=label)
        if true_pdf is not None:
            plt.plot(x_vals, true_pdf, '--', label='True PDF')
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("PDF Recovered from Characteristic Function")
        plt.grid(True)
        plt.legend()
        plt.show()

    def tail_probability(self, threshold, upper=10):
        prob, _ = quad(self.pdf, threshold, upper)
        return prob

    def show(self):
        if self.expression_str:
            print(f"Characteristic function expression:\n{self.expression_str}")
        else:
            print("Characteristic function expression not provided.")

# Normal distribution
class NormalCF(CharacteristicFunctionInverter):
    def __init__(self):
        super().__init__(phi=lambda t: np.exp(-0.5 * t**2))


# Exponential distribution with rate \lambda
class ExponentialCF(CharacteristicFunctionInverter):
    def __init__(self, lam=1.0):
        self.lam = lam
        phi = lambda t: self.lam / (self.lam - 1j * t)
        super().__init__(phi=phi)

# Uniform distribution over [a, b]
class UniformCF(CharacteristicFunctionInverter):
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

        def phi(t):
            t = np.asarray(t)
            result = np.ones_like(t, dtype=np.complex128)
            nonzero = t != 0
            result[nonzero] = (np.exp(1j * t[nonzero] * self.b) - np.exp(1j * t[nonzero] * self.a)) / (1j * t[nonzero] * (self.b - self.a))
            # 1j is the complex unit
            return result

        super().__init__(phi=phi)

def make_cf(type_name: str, params: dict) -> CharacteristicFunctionInverter:
    """
    Constructs a CharacteristicFunctionInverter using structural pattern matching.
    Supported types: 'normal', 'exponential', 'uniform'
    """
    match type_name.lower():
        case "normal":
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", 1.0)
            expr = f"exp(i * t * {mu} - 0.5 * ({sigma}^2) * t^2)"
            phi = lambda t: np.exp(1j * t * mu - 0.5 * (sigma ** 2) * t ** 2)
            return CharacteristicFunctionInverter(phi=phi, integration_limit=40, expression_str=expr)

        case "exponential":
            lam = params.get("lam", 1.0)
            expr = f"{lam} / ({lam} - i * t)"
            phi = lambda t: lam / (lam - 1j * t)
            return CharacteristicFunctionInverter(phi=phi, integration_limit=40, expression_str=expr)

        case "uniform":
            a = params.get("a", 0.0)
            b = params.get("b", 1.0)
            expr = f"(exp(i * t * {b}) - exp(i * t * {a})) / (i * t * ({b} - {a}))"

            def phi(t):
                t = np.asarray(t)
                result = np.ones_like(t, dtype=np.complex128)
                nonzero = t != 0
                result[nonzero] = (np.exp(1j * t[nonzero] * b) - np.exp(1j * t[nonzero] * a)) / (
                            1j * t[nonzero] * (b - a))
                return result

            return CharacteristicFunctionInverter(phi=phi, integration_limit=40, expression_str=expr)

        case _:
            raise ValueError(f"Unsupported distribution type: {type_name}")


class ConditionalCF:
    def __init__(self, func_y_to_cf, expression_str):
        self.func = func_y_to_cf  # y ↦ φ_{X|Y=y}(s)
        self.expression_str = expression_str  # symbolic string, e.g., 'exp(i * s * (2*y) - 0.5 * s^2)'

    def __call__(self, y):
        return self.func(y)

    def show(self, y=None, full=False):
        if full or y is None:
            print(f"Conditional CF symbolic expression:\n{self.expression_str}")
        else:
            # Fill in y if it's a numeric value
            filled_expr = self.expression_str.replace('y', str(y))
            print(f"Conditional CF at y={y}:\n{filled_expr}")

def make_conditional_cf(type_name: str, params: dict):
    """
    Returns a function y ↦ φ_{X|Y=y}(s)
    Supports:
    - "normal"       with mean
    - "uniform"      with a and b
    - "exponential"  with rate λ
    """
    match type_name.lower():

        case "normal":
            mean_expr = params.get("mean", "y")
            var = float(params.get("var", 1.0))
            expr_str = f"exp(i * s * ({mean_expr}) - 0.5 * {var} * s^2)"

            def conditional_cf(y):
                mu = eval(mean_expr, {"y": y}) if isinstance(mean_expr, str) else mean_expr
                return lambda s: np.exp(1j * s * mu - 0.5 * var * s ** 2)

            return ConditionalCF(conditional_cf, expr_str)

        case "uniform":
            a_expr = params.get("a")
            b_expr = params.get("b")
            expr_str = f"(exp(i * s * ({b_expr})) - exp(i * s * ({a_expr}))) / (i * s * (({b_expr}) - ({a_expr})))"

            def conditional_cf(y):
                a = eval(a_expr, {"y": y}) if isinstance(a_expr, str) else a_expr
                b = eval(b_expr, {"y": y}) if isinstance(b_expr, str) else b_expr

                def cf(s):
                    s = np.asarray(s, dtype=np.complex128)
                    result = np.ones_like(s, dtype=np.complex128)
                    nonzero = s != 0
                    result[nonzero] = (np.exp(1j * s[nonzero] * b) - np.exp(1j * s[nonzero] * a)) / (
                                1j * s[nonzero] * (b - a))
                    return result

                return cf

            return ConditionalCF(conditional_cf, expr_str)

        case "exponential":
            lam_expr = params.get("lam", "1.0")
            expr_str = f"({lam_expr}) / ({lam_expr} - i * s)"

            def conditional_cf(y):
                lam = eval(lam_expr, {"y": y}) if isinstance(lam_expr, str) else lam_expr
                return lambda s: lam / (lam - 1j * s)

            return ConditionalCF(conditional_cf, expr_str)

        case _:
            raise NotImplementedError(f"Conditional distribution type '{type_name}' is not supported.")

