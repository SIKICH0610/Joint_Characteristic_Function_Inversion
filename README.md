# Joint Characteristic Function Inversion

This project implements **probability density function (PDF) recovery and conditional probability computation** using **characteristic functions (CFs)** and their **Fourier inversion**. It supports:

- Univariate CF inversion to recover PDFs and tail probabilities.  
- Joint PDF reconstruction from conditional CFs and marginals.  
- Conditional probability estimation \(P(X > a \mid Y=y)\) and over intervals \(P(X>a \mid Y\in[y_1,y_2])\).  
- Fejér tapering to stabilize truncated Fourier integrals.  
- Two interchangeable backends: one using quadrature, and one using Gil–Peláez (GP).

---

## Key Idea

The characteristic function (CF) of a random variable \(X\) is

\[
\phi_X(t) = \mathbb{E}[e^{i t X}].
\]

By inverse Fourier transform,

\[
f_X(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-i t x}\,\phi_X(t)\, dt .
\]

### Joint and Conditional Structure

For a joint distribution \((X,Y)\):

- Marginal CF of \(Y\): \(\phi_Y(t) = \mathbb{E}[e^{i t Y}]\).  
- Conditional CF of \(X \mid Y=y\): \(\phi_{X \mid Y=y}(s) = \mathbb{E}[e^{i s X} \mid Y=y]\).

The joint CF:

\[
\phi_{X,Y}(s,t) = \int_{-\infty}^\infty \phi_{X|Y=y}(s)\, e^{i t y}\, f_Y(y)\, dy .
\]

The joint PDF:

\[
f_{X,Y}(x,y) = \frac{1}{(2\pi)^2} \iint e^{-i (s x + t y)} \,\phi_{X,Y}(s,t)\, ds\, dt .
\]

The marginal PDF of \(Y\):

\[
f_Y(y) = \frac{1}{2\pi} \int e^{-i t y}\, \phi_{X,Y}(0,t)\, dt .
\]

A useful slice:

\[
\psi_y(s) = \frac{1}{2\pi} \int e^{-i t y}\, \phi_{X,Y}(s,t)\, dt,
\qquad
f_{X\mid Y=y}(x) = \frac{1}{2\pi} \int e^{-i s x}\, \frac{\psi_y(s)}{f_Y(y)}\, ds .
\]

---

## Computing Probabilities

**Univariate**

\[
P(X>a) = \int_a^\infty f_X(x)\,dx .
\]

**Joint**

\[
P(X>a,\; Y\in[y_1,y_2]) = \int_{y_1}^{y_2} \int_a^\infty f_{X,Y}(x,y)\,dx\,dy .
\]

**Conditional**

\[
P(X>a \mid Y=y) = \frac{\int_a^\infty f_{X,Y}(x,y)\,dx}{f_Y(y)} .
\]

\[
P(X>a \mid Y\in[y_1,y_2]) =
\frac{\int_{y_1}^{y_2} \int_a^\infty f_{X,Y}(x,y)\,dx\,dy}{\int_{y_1}^{y_2} f_Y(y)\,dy} .
\]

**Gil–Peláez tail formula** (used in GP backend):

\[
P(X>a \mid Y=y) = \tfrac12+\frac{1}{\pi}\int_0^{L_x}\operatorname{Im}\!\left(e^{-isa}\,\frac{\phi_{X|Y=y}(s)}{s}\right) ds .
\]

---

## Project Structure

├─ Inverse_Fourier_Interface.py # shared interface, plotting, region slicer
├─ GP_Method.py # GPJointCFInverter (Gil–Peláez backend)
├─ Quad_Method.py # QuadJointCFInverter (quadrature backend)
├─ Joint_Helper.py # CF factories: make_cf, make_conditional_cf
├─ "Tests".py # Example usage for different cases

### Implemented Methods

Both backends implement:

- `from_conditional(...)` – build joint CF from marginal + conditional CF  
- `marginal_pdf_Y(y)` – evaluate f_Y(y)  
- `conditional_pdf_via_1d(x, y)` – evaluate f_{X|Y=y}(x)  
- `conditional_probability_point(a, y)` – compute P(X > a | Y = y)  
- `conditional_probability_region(a, (yL, yH))` – region conditional via y-slices  
- `joint_pdf(x, y)` – reconstruct joint density  
- `show_expression(...)`, `plot_conditional_pdf(...)` – inspection and visualization  
