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

From the **Gil-Pelaez inversion theorem**, the PDF of $X$ can be recovered using the **inverse Fourier transform**:

$$
f_X(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-i t x}    \phi_X(t)    dt,
$$

where a **Gaussian damping factor** $e^{-\alpha t^2}$ is introduced numerically to stabilize oscillations in the integral.

---

### 📌 Joint and Conditional Structure

For a **joint distribution** $(X,Y)$:

- The **marginal CF of $Y$** is $\phi_Y(t) = \mathbb{E}[e^{i t Y}]$.
- The **conditional CF of $X \mid Y=y$** is $\phi_{X \mid Y=y}(s) = \mathbb{E}[e^{i s X} \mid Y=y]$.

We can build the **joint CF**:

$$
\phi_{X,Y}(s,t) = \int_{-\infty}^\infty \phi_{X|Y=y}(s)    e^{i t y}    f_Y(y)    dy,
$$

and then recover the **joint PDF**:

$$
f_{X,Y}(x,y) = \frac{1}{(2\pi)^2} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} 
e^{-i (sx + ty)}    \phi_{X,Y}(s,t)    ds    dt.
$$

The **marginal PDF** of $Y$ is:

$$
f_Y(y) = \frac{1}{2\pi} \int_{-\infty}^\infty e^{-i t y}    \phi_{X,Y}(0,t)    dt.
$$


---

### 🎯 Computing Probabilities

### 1. Univariate Probability

Once PDFs are obtained, probabilities are computed via integration.

### 2. Joint Probability

For $(X,Y)$:

$$
P(X > a, \; Y \in [y_1, y_2]) 
= \int_{y_1}^{y_2} \int_a^\infty f_{X,Y}(x,y)    dx    dy.
$$

### 3. Conditional Probability

For a single $y$:

$$
P(X > a \mid Y = y) = \frac{\int_a^\infty f_{X,Y}(x,y)    dx}{f_Y(y)}.
$$

For a range $Y \in [y_1, y_2]$:

$$
P(X > a \mid Y \in [y_1,y_2]) 
= \frac{\int_{y_1}^{y_2} \int_a^\infty f_{X,Y}(x,y)    dx    dy}
       {\int_{y_1}^{y_2} f_Y(y)    dy}.
$$


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
        - `marginal_pdf_Y_quad()` – Marginal density $f_Y(y)$  
        - `conditional_pdf_X_given_Y()` – Conditional density $f(X|Y=y)$  
        - `conditional_probability()` – Compute $P(X > a \mid Y=y)$ or $P(X > a \mid Y \in [y_1,y_2])$  
        - `plot_joint_pdf()` – Contour/surface plotting

## Remaining Problems

Currently, the **conditional probability computation** (for example, $P(X > a \mid Y=y)$ ) is not producing stable results due to numerical oscillations introduced during characteristic function inversion.

### Issue
- The inversion of conditional PDFs via truncated Fourier integrals introduces the Gibbs phenomenon:

  $$f_{X \mid Y}(x \mid y) = \frac{1}{2 \pi} \int_{-L}^{L} e^{-i s x} \phi_{X \mid Y = y}(s) ds$$

  where $L$ is the finite truncation bound.

- Truncating the infinite integral to the interval \([-L, L]\) is equivalent to convolving the true PDF with a sinc kernel, which introduces oscillations near discontinuities and in the tails.

As a result:
- The computed conditional PDFs exhibit oscillatory tails.
- Integrals for probabilities such as $\int_{a}^{\infty} f_{X \mid Y}(x \mid y) dx$ fail to converge reliably.

---

### Potential Solutions

#### 1. Windowing Functions (Spectral Tapering)
Instead of a sharp cutoff at $[-L, L]$, apply a smooth window function $w(s)$ that decays near the boundaries:

$$f_{X \mid Y}(x \mid y) \approx \frac{1}{2 \pi} \int_{-L}^{L} e^{-i s x}    \phi_{X \mid Y = y}(s)    w(s)    ds$$

Common choices include:
- **Hamming window**:

  $$w_{\text{Hamming}}(s) = 0.54 + 0.46 \cos\left( \frac{\pi s}{L} \right)$$

- **Blackman window**:

  $$w_{\text{Blackman}}(s) = 0.42 + 0.50 \cos\left( \frac{\pi s}{L} \right) + 0.08 \cos\left( \frac{2 \pi s}{L} \right)$$

These windows gradually suppress high-frequency components, reducing oscillations.

---

#### 2. Fejér Summation (Cesàro Averaging)
A more rigorous fix is Fejér summation, which explicitly removes Gibbs oscillations by averaging partial Fourier sums.

Instead of direct truncation:

$$f_{X \mid Y}(x \mid y) = \lim_{L \to \infty} \frac{1}{2 \pi} \int_{-L}^{L} e^{-i s x}    \phi_{X \mid Y = y}(s) ds$$

Fejér modifies the inversion to:

$$f^{\text{Fejér}}_{X \mid Y}(x \mid y) = \frac{1}{2 \pi} \int_{-L}^{L} \left( 1 - \frac{|s|}{L} \right) e^{-i s x} \, \phi_{X \mid Y = y}(s) \, ds$$

Here, the triangular weight $1 - |s|/L$ damps oscillations arising from the truncation edges and improves convergence. This is equivalent to convolving the PDF with the Fejér kernel, which has smoother convergence properties.

---

### Current Status
- At present, conditional probability computation is not available due to instability.
- Future work will implement Fejér-weighted inversion or optimized spectral windowing to stabilize conditional PDF reconstruction.
- For now, only joint PDF and marginal PDF computations are verified and supported.

