# Joint Characteristic Function Inversion

This project implements **probability density function (PDF) recovery and conditional probability computation** using **characteristic functions (CFs)** and their **Fourier inversion**. It supports:
- **Univariate CF inversion** to recover PDFs and tail probabilities.
- **Joint PDF reconstruction** from conditional CFs and marginals.
- **Conditional probability estimation** $P(X > a \mid Y=y)$ or over intervals.

---

## Key Idea

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

## Joint and Conditional Structure

For a joint distribution $(X,Y)$:

The **marginal CF** of $Y$ is  

  $$\phi_Y(t) = \mathbb{E}[e^{i t Y}].$$
  
The **conditional CF** of $X \mid Y=y$ is  

  $$\phi_{X \mid Y=y}(s) = \mathbb{E}[e^{i s X} \mid Y=y].$$

The **joint CF** is constructed via:

$$\phi_{X,Y}(s,t) = \int_{-\infty}^\infty \phi_{X|Y=y}(s)e^{i t y}f_Y(y)dy.$$

From this, we recover:

**Joint PDF**:  

  $$f_{X,Y}(x,y) = \frac{1}{(2\pi)^2}\int\\int e^{-i(sx+ty)}\phi_{X,Y}(s,t)dsdt,$$

**Marginal of $Y$**:  

  $$f_Y(y) = \frac{1}{2\pi}\int e^{-i t y}\phi_{X,Y}(0,t)dt.$$

---

## Progress & Methods

### Current Implementation
- **Univariate CF inversion** (Normal, Exponential, Uniform).  
- **Joint CF construction** from conditional + marginal distributions.  
- **Conditional tail probabilities** via **Gil–Pelaez inversion**.  
- **Fejér tapering** to suppress oscillations from Fourier truncation.  
- **Two backends**:  
  - **Fast Gauss–Legendre + GP backend** (recommended).  
  - **Reference SciPy quad/dblquad backend** for verification.  

### Numerical Stabilization
- **Fejér triangular weights** for smoother convergence.  
- **Gaussian damping** in $y$ integration.  
- Adjustable quadrature nodes $(N_s, N_t)$ and cutoffs $(L_x, L_y)$.  

---

##  Project Structure

- **Joint_Helper.py**  
  Factories for univariate CFs (`make_cf`) and conditional CFs (`make_conditional_cf`).  

- **Inverse_Fourier_Interface.py**  
  Abstract base + shared tools for constructing joint CFs, plotting, and inversion.  

- **GP_Method.py**  
  Gauss–Legendre quadrature + Gil–Pelaez tail integral + Fejér taper.  

- **Quad_Method.py**  
  Reference implementation using SciPy `quad`/`dblquad`.  

### Examples/scripts

- **Gil-Pelaez Case Test.py** — Nonlinear-mean sanity check for $P(X>0\mid Y=y)$ vs analytic $\Phi(\mu(y))$.
- **joint_normal_over_region.py** — Computes $P(X>0\mid Y\in[y_1,y_2])$ for a normal conditional model; compares to an analytic baseline.
- **joint_normal_slicemethod_test.py** — Minimal slice-integration demo over a $Y$-interval.
- **validate_joint_normal.py** — Builds $Y\sim N(0,1)$, $X\mid Y=y\sim N(2y,1)$; plots conditional PDFs and compares $P(X>0\mid Y=y)$ to analytic values.
- **validate_exponential.py** — Univariate exponential: PDF recovery and tail probability check.
- **validate_normal.py** — Univariate normal: PDF recovery and tail probability check. 
- **validate_uniform.py** — Univariate uniform on $[0,2]$: PDF recovery and tail probability check.


### Computing Probabilities

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

## Remaining Problems

Currently, the **conditional probability computation** (e.g., $P(X > a \mid Y=y)$) is working correctly, but can be computationally slow, even with the Gauss–Pelaez (GP) method. This slowdown is especially pronounced for **nonlinear conditional means** and **higher-dimensional random variables**.

### Causes
- **High-dimensional oscillatory integrals** grow quickly in cost as dimension increases.  
- **Nonlinear conditional structures** (e.g., polynomial mean functions) lead to stronger oscillations in the characteristic function.  
- **Large cutoffs $(L_x,L_y)$** and fine quadrature grids $(N_s,N_t)$ are required for accuracy, increasing runtime.

### Solutions
Coming soon(research on Wavelet theory and more FFTs)
