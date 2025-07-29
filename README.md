# Joint Characteristic Function Inversion

This project implements **probability density function (PDF) recovery and conditional probability computation** using **characteristic functions (CFs)** and their **Fourier inversion**. It supports:
- **Univariate CF inversion** to recover PDFs and tail probabilities.
- **Joint PDF reconstruction** from conditional CFs and marginals.
- **Conditional probability estimation** $P(X > a \mid Y=y)$ or over intervals.

---

## 🔑 Key Idea

The **characteristic function (CF)** of a random variable $X$ is defined as:

$$
\phi_X(t) = \mathbb{E}\left[ e^{i t X} \right].
$$

A PDF can be recovered by **inverse Fourier transform**:

$$
f_X(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-i t x}  \phi_X(t) dt.
$$

For **joint distributions**, if we know:
- The marginal CF of $Y$, $\phi_Y(t)$,
- The conditional CF of $X \mid Y=y$, $\phi_{X \mid Y=y}(s)$,

then the **joint CF** is:

$$
\phi_{X,Y}(s,t) = \int_{-\infty}^\infty \phi_{X|Y=y}(s)  e^{i t y}  f_Y(y)  dy,
$$

and the joint PDF follows by **double inversion**:

$$
f_{X,Y}(x,y) = \frac{1}{(2\pi)^2} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} 
e^{-i (sx + ty)}  \phi_{X,Y}(s,t)  ds  dt.
$$

We add a **Gaussian damping term** $e^{-\alpha t^2}$ to stabilize numerical integration.

---

## 📂 Project Structure

1. **CF_Inverter.py**
    1.1 `CharacteristicFunctionInverter` – Univariate CF inversion  
    1.2 `NormalCF` – CF of Normal(0,1)  
    1.3 `ExponentialCF` – CF of Exponential($\lambda$)  
    1.4 `UniformCF` – CF of Uniform(a,b)  
    1.5 `make_cf()` – Factory for univariate CFs  

2. **Joint_Helper.py**
    2.1 `ConditionalCF` – Conditional CF wrapper $y \mapsto \phi_{X|Y=y}(s)$  
    2.2 `make_conditional_cf()` – Factory for conditional CFs  

3. **JointCharacteristicFunctionInverter.py**
    3.1 `JointCharacteristicFunctionInverter` – Joint CF inversion & conditional probability  
        - `from_conditional()` – Build joint CF from marginal + conditional CF  
        - `joint_pdf()` – Joint density $f(x,y)$  
        - `marginal_pdf_Y()` – Marginal density $f_Y(y)$  
        - `conditional_pdf_X_given_Y()` – Conditional density $f(X|Y=y)$  
        - `conditional_probability()` – Compute $P(X > a \mid Y=y)$ or $P(X > a \mid Y \in [y_1,y_2])$  
        - `plot_joint_pdf()` – Contour/surface plotting  
