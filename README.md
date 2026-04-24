# pathsig-finance

**Rough path signatures for financial time series** — a Python library and research
toolkit for applying truncated path signatures to cross-sectional equity return
prediction, with a rigorous focus on mathematical interpretability.

---

## Core Thesis

Truncated path signatures of multivariate financial time series (price, volume,
volatility) produce features that are both **mathematically principled** and
**financially interpretable**. Specific signature terms correspond exactly to
known financial phenomena:

| Signature term | Financial phenomenon | Reference |
|---|---|---|
| $S^1_r = X^r_T - X^r_0$ | Momentum (cumulative return) | Jegadeesh & Titman (1993) |
| $A_{\text{lead}_r, \text{lag}_r}$ | Realised variance (quadratic variation) | Flint, Hambly & Lyons (2016) |
| $A_{r,\sigma} = S^2_{r,\sigma} - S^2_{\sigma,r}$ | Leverage effect | Black (1976) |
| $A_{r,v} = S^2_{r,v} - S^2_{v,r}$ | Price-volume lead-lag direction | Easley & O'Hara (1987) |

where $A_{i,j}$ denotes the **Lévy area** (antisymmetric level-2 signature term).

---

## Mathematical Background

For a path $X: [0,T] \to \mathbb{R}^d$, the **truncated signature** up to depth $N$ is:

$$S(X)^{\leq N} = \left(1,\; S^1,\; S^2,\; \ldots,\; S^N\right)$$

where the level-$k$ terms are iterated integrals:

$$S(X)^k_{i_1, \ldots, i_k} = \int_{0 < t_1 < \cdots < t_k < T} dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}$$

**Key property (Lyons 1998):** Any continuous functional of the path can be approximated
by a *linear* functional of its signature as $N \to \infty$ — the signature is a universal
feature map for sequential data.

The output dimension is $\sum_{k=1}^{N} d^k$. For 3 channels and depth 3: **39 features**.

See [`docs/math_background.md`](docs/math_background.md) for full proofs and derivations.

---

## Installation

```bash
git clone https://github.com/rxj0102/pathsig-finance.git
cd pathsig-finance
pip install -e .
```

**Optional (GPU-accelerated signatures):**
```bash
pip install -e ".[gpu]"   # requires PyTorch + signatory
```

**Development (tests, notebooks):**
```bash
pip install -e ".[dev]"
```

---

## Quickstart

```python
import numpy as np
from pathsig import compute_signature, SignatureFeatureExtractor
from pathsig.augmentations import time_augmentation, lead_lag_augmentation
from pathsig.interpretability import full_interpretability_report

# 1. Compute signature of a single path window
path = np.random.randn(21, 3) * 0.01        # 21-day window, 3 channels
path = time_augmentation(path)               # prepend time channel
path = lead_lag_augmentation(path)           # interleave with lagged copy
sig  = compute_signature(path, depth=2)     # 72-dimensional vector
print(f"Signature shape: {sig.shape}")       # (72,)

# 2. Interpretability report for a financial path
raw_path = np.random.randn(30, 3) * 0.01
report = full_interpretability_report(
    raw_path,
    channel_names=["log_return", "realized_vol", "log_volume"],
    augmentations=["lead_lag"],
    price_channel=0, vol_channel=1, volume_channel=2,
)
print("Leverage effect Lévy area:", report["leverage_effect"]["levy_area"])
print("Lead-lag direction:",        report["price_volume_leadlag"]["lead_lag_direction"])

# 3. Extract features for a full panel
import pandas as pd
from pathsig import SignatureFeatureExtractor

panel = pd.read_csv("data/sample_yahoo.csv")  # or use synthetic data
extractor = SignatureFeatureExtractor(
    channels=["log_return", "realized_vol", "log_volume"],
    window=21, depth=2,
    augmentations=["time", "lead_lag"],
)
features = extractor.fit_transform(panel)
print(f"Feature matrix: {features.shape}")
```

---

## Repository Structure

```
pathsig-finance/
├── pathsig/
│   ├── signatures.py          # Signature computation (NumPy + optional GPU)
│   ├── augmentations.py       # Time, lead-lag, invisibility, CMA augmentations
│   ├── interpretability.py    # Formal mapping: sig terms → financial phenomena
│   ├── features.py            # SignatureFeatureExtractor, BenchmarkFeatureExtractor
│   ├── models.py              # CrossSectionalPredictor (ridge, OLS, elastic net, RF)
│   └── evaluation.py          # OOS R², Diebold-Mariano, Fama-MacBeth, portfolio sorts
│
├── data/
│   ├── loader.py              # CSV/Parquet panel loading and validation
│   ├── synthetic.py           # Synthetic generators (Heston, lead-lag, momentum)
│   └── README.md              # Instructions for obtaining real data (CRSP, Yahoo)
│
├── experiments/
│   ├── config.yaml            # Central experiment configuration
│   ├── run_synthetic_validation.py   # Verify sig terms recover known parameters
│   ├── run_cross_sectional.py        # Main OOS prediction experiment
│   ├── run_interpretability.py       # Interpretability analysis
│   └── run_ablation.py               # Depth/window/augmentation ablation
│
├── notebooks/
│   ├── 01_signature_primer.ipynb           # Educational: what are signatures
│   ├── 02_financial_interpretability.ipynb # Walkthrough of all correspondences
│   └── 03_empirical_results.ipynb          # OOS prediction and portfolio sorts
│
├── tests/                     # 73 unit tests (pytest)
├── docs/math_background.md    # Formal definitions, proofs, references
└── scripts/download_sample_data.py   # Fetch data via Yahoo Finance
```

---

## Reproducing Experiments

All experiments use synthetic data by default (no data download required).

```bash
# 1. Synthetic validation — verify signature terms recover known parameters
python experiments/run_synthetic_validation.py

# 2. Main cross-sectional prediction experiment
python experiments/run_cross_sectional.py

# 3. Interpretability analysis
python experiments/run_interpretability.py

# 4. Ablation studies (depth, window, augmentations)
python experiments/run_ablation.py
```

Results are saved to `experiments/results/`. Edit `experiments/config.yaml` to
change data mode, model type, signature depth, etc.

**To use real data**, download via Yahoo Finance:
```bash
python scripts/download_sample_data.py --tickers AAPL MSFT GOOG AMZN META \
    --start 2015-01-01 --end 2023-12-31 --output data/sample_yahoo.csv
```
Then set `data.mode: real` and `data.panel_path: data/sample_yahoo.csv` in `config.yaml`.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

73 tests covering: signature correctness (Chen identity, known analytical values),
augmentation properties (lead-lag Lévy area = QV), interpretability mappings,
evaluation statistics (DM test, Fama-MacBeth recovery), and synthetic data properties.

---

## Key Design Choices

**Why depth 2–3?** Most economically meaningful phenomena are captured by first- and
second-order interactions. Depth 3 adds skewness proxies and vol-conditioned momentum.
Higher depth raises dimensionality without proportionate signal gain.

**Why lead-lag augmentation?** The raw signature is invariant to time reparameterisation.
The lead-lag transform provides signature-based access to the quadratic variation
(realised variance) and the Lévy area (lead-lag relationships) without requiring
an explicit volatility channel.

**Why Lévy area for the leverage effect?** The symmetric part of $S^2_{r,\sigma}$
captures the product of total displacements; the *anti*symmetric part (Lévy area)
isolates the temporal ordering — which channel moved first — giving a clean signal
for asymmetric effects like leverage.

---

## Citation

```bibtex
@software{pathsig_finance,
  author  = {rxj0102},
  title   = {pathsig-finance: Rough Path Signatures for Financial Time Series},
  year    = {2026},
  url     = {https://github.com/rxj0102/pathsig-finance},
  version = {0.1.0}
}
```

---

## References

- Lyons, T. (1998). Differential equations driven by rough signals.
  *Revista Matemática Iberoamericana*, 14(2):215–310.

- Chevyrev, I. & Kormilitzin, A. (2016). A primer on the signature method in
  machine learning. *arXiv:1603.03788*.

- Kidger, P. & Lyons, T. (2021). Signatory: differentiable computations of the
  signature and log-signature transforms. *ICLR 2021*.

- Flint, G., Hambly, B. & Lyons, T. (2016). Discretely sampled signals and the
  rough Hoff process. *Stochastic Processes and their Applications*, 126(9).

- Black, F. (1976). Studies of stock price volatility changes.
  *Proceedings of the ASA*, 177–181.

- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers.
  *Journal of Finance*, 48(1):65–91.
