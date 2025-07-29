# Joint Characteristic Function Inversion

This project implements **probability density function (PDF) recovery and conditional probability computation** using **characteristic functions (CFs)** and their **Fourier inversion**. It supports:
- **Univariate CF inversion** to recover PDFs and tail probabilities.
- **Joint PDF reconstruction** from conditional CFs and marginals.
- **Conditional probability estimation** \( P(X > a \mid Y=y) \) or over intervals.

---

## 🔑 Key Idea

The **characteristic function (CF)** of a random variable $X$ is defined as:
$$
\phi_X(t) = \mathbb{E}\left[ e^{i t X} \right].
$$

A PDF can be recovered by **inverse Fourier transform**:
\[
f_X(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-i t x} \, \phi_X(t) \, dt.
\]

For **joint distributions**, if we know:
- The marginal CF of \(Y\), \(\phi_Y(t)\),
- The conditional CF of \(X \mid Y=y\), \(\phi_{X \mid Y=y}(s)\),

then the **joint CF** is:
\[
\phi_{X,Y}(s,t) = \int_{-\infty}^\infty \phi_{X|Y=y}(s) \, e^{i t y} \, f_Y(y) \, dy,
\]
and the joint PDF follows by **double inversion**:
\[
f_{X,Y}(x,y) = \frac{1}{(2\pi)^2} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} 
e^{-i (sx + ty)} \, \phi_{X,Y}(s,t) \, ds \, dt.
\]

We add a **Gaussian damping term** \(e^{-\alpha t^2}\) to stabilize numerical integration.

---

## 📂 Project Structure

