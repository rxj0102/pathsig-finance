# Mathematical Background

## 1. Rough Path Signatures

### 1.1 Definition

Let $X: [0,T] \to \mathbb{R}^d$ be a continuous path of bounded variation.
The **truncated signature** of $X$ up to depth $N$ is the element of the
truncated tensor algebra $T^N(\mathbb{R}^d) = \bigoplus_{k=0}^{N} (\mathbb{R}^d)^{\otimes k}$:

$$S(X)^{\leq N} = \left(1,\; S(X)^1,\; S(X)^2,\; \ldots,\; S(X)^N\right)$$

where the **level-$k$ signature tensor** is the $d^k$-dimensional array of iterated integrals:

$$S(X)^k_{i_1, \ldots, i_k} = \int_{0 < t_1 < \cdots < t_k < T}
  dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}, \qquad i_j \in \{1, \ldots, d\}$$

For a **piecewise-linear path** with increments $\delta_n = X_{t_n} - X_{t_{n-1}}$,
all integrals reduce to finite sums computable by the recurrence below.

### 1.2 The Chen Identity

The signature obeys the multiplicative concatenation rule. For $s < u < t$:

$$S(X|_{[s,t]}) = S(X|_{[s,u]}) \otimes S(X|_{[u,t]})$$

where $\otimes$ is the **tensor algebra product** (shuffle product):

$$(A \otimes B)^k = \sum_{j=0}^{k} A^j \otimes B^{k-j}, \qquad A^0 = B^0 = 1$$

This identity is the computational core of `pathsig/signatures.py`. The signature
of a long path is built by iterating over increments, applying the Chen product at each step.

**Incremental update rule** for a single new increment $\delta \in \mathbb{R}^d$:

$$S^k_{\text{new}} = \sum_{j=0}^{k} S^j_{\text{old}} \otimes \underbrace{\frac{\delta^{\otimes(k-j)}}{(k-j)!}}_{\text{level-}(k-j)\text{ sig of straight segment}}$$

### 1.3 Universality (Lyons, 1998)

**Theorem** (Lyons 1998, Chen 1954): The truncated signature is a *universal* feature map
for continuous paths in the following sense: any continuous function
$f: C^{p\text{-var}}([0,T], \mathbb{R}^d) \to \mathbb{R}$ of the path
can be approximated arbitrarily well by a **linear** functional of the full signature
$S(X)^{\leq N}$ as $N \to \infty$.

This is the analogue for paths of the Stone-Weierstrass theorem for functions.
It implies that the signature is an *injective* map (modulo reparameterisation and
tree-like pieces): two paths have the same signature if and only if they are equivalent
under these transformations.

**Practical implication**: For a fixed depth $N$, the truncated signature captures
all information about the path up to order $N$ interactions between the $d$ channels.
Depth $N=3$ is typically sufficient for financial applications because most
economically meaningful phenomena are captured by first- and second-order interactions.

### 1.4 Signature Dimension

The dimension of the non-constant part of the depth-$N$ truncated signature of a
$d$-dimensional path is:

$$\dim S(X)^{\leq N} = \sum_{k=1}^{N} d^k = d \cdot \frac{d^N - 1}{d - 1} \quad (d > 1)$$

For $d=3$ channels and $N=3$: $3 + 9 + 27 = 39$ features.
For $d=7$ channels after time + lead-lag augmentation ($d_{\text{orig}}=3 \to 8$)
and $N=2$: $8 + 64 = 72$ features.

---

## 2. Path Augmentations

### 2.1 Time Augmentation

Raw signatures are invariant to time reparameterisation. Prepending a linearly
increasing time channel $t \in [0,1]$ breaks this symmetry:

$$\tilde{X}_t = (t, X_t) \in \mathbb{R}^{d+1}$$

The level-2 cross-term $S^2_{\text{time}, i}$ then captures the **timing** of moves
in channel $i$ within the window — whether a channel tends to move early or late.

### 2.2 Lead-Lag Transform

For a $d$-dimensional path $X$, the **lead-lag transform** produces a $2d$-dimensional
path by interleaving each point with its one-step lag:

$$\text{LL}(X) = \big((X_{t_0}, X_{t_0}),\, (X_{t_1}, X_{t_0}),\, (X_{t_1}, X_{t_1}),\,
   (X_{t_2}, X_{t_1}),\, \ldots\big)$$

**Key result** (Flint, Hambly & Lyons 2016): The Lévy area between lead channel $i$
and lag channel $i$ of the lead-lag path equals the **quadratic variation** of $X^i$:

$$A_{\text{lead}_i, \text{lag}_i} := S^2_{\text{lead}_i, \text{lag}_i}
  - S^2_{\text{lag}_i, \text{lead}_i} = \sum_k (\Delta X^i_k)^2 = [X^i, X^i]_T$$

*Proof*: For each step $k$, the lead-lag path traces a right-angle turn in the
$({\rm lead}_i, {\rm lag}_i)$ plane with legs $|\Delta X^i_k|$. The Lévy area of
this turn is $(\Delta X^i_k)^2$. Summing over all steps gives the quadratic variation. $\square$

Note: the individual term $S^2_{\text{lead}_i, \text{lag}_i} = \frac{1}{2}(\text{CumRet}_i)^2 + \frac{1}{2}\text{QV}_i$, so the Lévy area (antisymmetric part) is required to isolate QV.

---

## 3. Financial Interpretability Correspondences

### 3.1 Level-1 → Momentum

$$S^1_r = \int_0^T dX^r_t = X^r_T - X^r_0 = \text{cumulative log-return}$$

This is **exactly** the momentum signal of Jegadeesh & Titman (1993). No approximation:
the level-1 signature term *is* the cumulative return. The cross-sectional predictive power
of momentum is therefore *entirely* captured by the level-1 signature.

### 3.2 Realised Variance via Lead-Lag

After lead-lag augmentation, the Lévy area between lead and lag channels equals
the quadratic variation (see §2.2). For the log-return channel with daily increments:

$$[r, r]_T = \sum_{t=1}^{T} r_t^2 \approx T \cdot \hat{\sigma}^2$$

where $\hat{\sigma}^2$ is the realised variance. The realised volatility estimate is
$\hat{\sigma} = \sqrt{[r,r]_T / T}$.

### 3.3 The Leverage Effect (Black, 1976)

The **leverage effect** is the empirical observation that equity returns are
negatively correlated with future volatility: $\text{corr}(\Delta r_t, \Delta \sigma_{t+1}) < 0$.

For a path with channels $r$ (log-return) and $\sigma$ (volatility), the
**level-2 Lévy area** captures this:

$$A_{r,\sigma} = S^2_{r,\sigma} - S^2_{\sigma,r}
  = \int_0^T X^r_t\, dX^\sigma_t - \int_0^T X^\sigma_t\, dX^r_t \quad \text{(Itô)}$$

When $A_{r,\sigma} < 0$: returns tend to move *before* volatility increases
— i.e., negative returns predict higher volatility. This is the leverage effect.

**Theorem** (informal): For a stochastic volatility model
$dS/S = \mu\,dt + \sigma_t\,dW^1_t$, $d\sigma_t = \kappa(\theta - \sigma_t)\,dt + \xi\,dW^2_t$,
with $\text{corr}(dW^1, dW^2) = \rho$:

$$\mathbb{E}[A_{r,\sigma}] \propto \rho$$

so the sign of the Lévy area recovers the sign of the leverage correlation $\rho$.

### 3.4 Price-Volume Lead-Lag

The Lévy area between the price/return channel $r$ and the log-volume channel $v$:

$$A_{r,v} = S^2_{r,v} - S^2_{v,r}$$

identifies the temporal ordering of price and volume movements:

- $A_{r,v} < 0$ ($S^2_{v,r} > S^2_{r,v}$): Volume moves first, price follows.
  Consistent with **informed trading** (Glosten & Milgrom 1985, Easley & O'Hara 1987).

- $A_{r,v} > 0$ ($S^2_{r,v} > S^2_{v,r}$): Price moves first, volume follows.
  Consistent with **momentum chasing** / trend-following behaviour.

### 3.5 Level-3 Terms: Higher-Order Interactions

Level-3 terms $S^3_{i,j,k}$ capture third-order temporal interactions. The diagonal terms
approximate skewness proxies:

$$S^3_{r,r,r} \approx \frac{1}{6} \sum_k (\Delta r_k)^3 \propto \text{skewness}(r)$$

and off-diagonal terms like $S^3_{r,\sigma,r}$ capture how the trend in returns
interacts with the volatility level — a form of **volatility-conditioned momentum**.

---

## 4. Truncation Level Choice

### Information Loss at Finite Depth

For financial applications, depth $N=2$ or $N=3$ is typically sufficient because:

1. **Financial signals are low-order**: momentum (level 1), volatility (level 2 via lead-lag),
   and most factor models are captured by first- and second-order statistics.

2. **Curse of dimensionality**: depth $N$ gives $O(d^N)$ features; for $d=7$ and $N=4$,
   this is $7+49+343+2401 = 2800$ features. Use PCA or the log-signature for compression.

3. **Noise amplification**: higher-order terms are products of small increments and tend to
   be noisier relative to their signal content.

### The Log-Signature

The **log-signature** $\log S(X)$ lives in the free Lie algebra and has dimension
equal to the number of Lyndon words of length $\leq N$ in $d$ letters.  For $d=3$, $N=3$:

$$\dim \log S = 3 + 3 + 8 = 14 \quad \text{(vs. } 39 \text{ for the full signature)}$$

The log-signature is more compact because it removes redundant tensor products,
keeping only the "genuinely new" information at each level.

---

## 5. References

- **Lyons (1998)**: T. J. Lyons. Differential equations driven by rough signals.
  *Revista Matemática Iberoamericana*, 14(2):215–310.

- **Chen (1954)**: K. T. Chen. Iterated integrals and exponential homomorphisms.
  *Proc. London Math. Soc.*, s3-4(1):502–512.

- **Chevyrev & Kormilitzin (2016)**: I. Chevyrev and A. Kormilitzin. A primer on the
  signature method in machine learning. *arXiv:1603.03788*.

- **Kidger & Lyons (2021)**: P. Kidger and T. Lyons. Signatory: differentiable
  computations of the signature and log-signature transforms, on both CPU and GPU.
  *ICLR 2021*. https://github.com/patrick-kidger/signatory

- **Flint, Hambly & Lyons (2016)**: G. Flint, B. Hambly and T. Lyons. Discretely
  sampled signals and the rough Hoff process. *Stochastic Processes and their Applications*,
  126(9):2593–2614.

- **Black (1976)**: F. Black. Studies of stock price volatility changes. *Proceedings of
  the 1976 Meetings of the American Statistical Association*, 177–181.

- **Jegadeesh & Titman (1993)**: N. Jegadeesh and S. Titman. Returns to buying winners
  and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1):65–91.

- **Diebold & Mariano (1995)**: F. X. Diebold and R. S. Mariano. Comparing predictive
  accuracy. *Journal of Business & Economic Statistics*, 13(3):253–263.

- **Fama & MacBeth (1973)**: E. F. Fama and J. D. MacBeth. Risk, return, and equilibrium:
  Empirical tests. *Journal of Political Economy*, 81(3):607–636.

- **Newey & West (1987)**: W. K. Newey and K. D. West. A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3):703–708.

- **Heston (1993)**: S. Heston. A closed-form solution for options with stochastic
  volatility with applications to bond and currency options. *Review of Financial Studies*,
  6(2):327–343.
