"""
Prediction models for cross-sectional equity return forecasting.

The focus of this library is on features (signatures), not on model
complexity.  We therefore keep models simple and well-understood:

* OLS with Huber-White heteroskedasticity-robust standard errors
* Ridge regression (L2-penalised, λ cross-validated on training data)
* Elastic net (L1 + L2 penalty)
* Random forest (nonlinear benchmark)

All models share the :class:`CrossSectionalPredictor` interface, which
runs an expanding-window out-of-sample prediction and stores per-period
predictions alongside realised returns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


_MODEL_REGISTRY: dict[str, type] = {
    "ols": LinearRegression,
    "ridge": RidgeCV,
    "elasticnet": ElasticNetCV,
    "rf": RandomForestRegressor,
}


def _build_model(model_type: str, **kwargs):
    """Instantiate a sklearn estimator by name."""
    model_type = model_type.lower()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from {list(_MODEL_REGISTRY)}."
        )
    cls = _MODEL_REGISTRY[model_type]
    # Sensible defaults per model
    if model_type == "ridge":
        kwargs.setdefault("alphas", np.logspace(-4, 4, 50))
        kwargs.setdefault("cv", 5)
    elif model_type == "elasticnet":
        kwargs.setdefault("l1_ratio", [0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("max_iter", 5000)
    elif model_type == "rf":
        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("max_depth", 5)
        kwargs.setdefault("random_state", 42)
    return cls(**kwargs)


class CrossSectionalPredictor:
    """Expanding-window out-of-sample cross-sectional return predictor.

    At each date t (starting after ``min_train_periods``):
        1. Train on all available data up to (and including) t.
        2. Predict cross-sectional returns for date t+1.
        3. Record predictions and realised returns.

    Cross-sectional prediction means that at each date we regress the
    *cross-section* of stock returns on the *cross-section* of features,
    with the model trained on the pooled panel up to date t.

    Parameters
    ----------
    model_type:
        One of ``'ols'``, ``'ridge'`` (default), ``'elasticnet'``, ``'rf'``.
    standardize:
        If True, prepend a StandardScaler in the pipeline.
    **kwargs:
        Passed to the underlying sklearn estimator.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> n_dates, n_stocks = 300, 50
    >>> dates = pd.date_range('2010-01-01', periods=n_dates, freq='B')
    >>> tickers = [f'S{i:02d}' for i in range(n_stocks)]
    >>> rng = np.random.default_rng(0)
    >>> panel = pd.DataFrame({
    ...     'date': np.repeat(dates, n_stocks),
    ...     'ticker': np.tile(tickers, n_dates),
    ...     'feat1': rng.standard_normal(n_dates * n_stocks),
    ...     'ret_forward': rng.standard_normal(n_dates * n_stocks) * 0.01,
    ... })
    >>> predictor = CrossSectionalPredictor(model_type='ridge')
    >>> preds = predictor.expanding_window_predict(
    ...     panel, feature_cols=['feat1'], return_col='ret_forward',
    ...     min_train_periods=100)
    >>> 'predicted' in preds.columns
    True
    """

    def __init__(
        self,
        model_type: str = "ridge",
        standardize: bool = True,
        **kwargs,
    ) -> None:
        self.model_type = model_type
        self.standardize = standardize
        self.model_kwargs = kwargs
        self._last_model = None

    def _make_pipeline(self):
        estimator = _build_model(self.model_type, **self.model_kwargs)
        if self.standardize:
            return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        return estimator

    def expanding_window_predict(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        return_col: str = "ret_forward",
        min_train_periods: int = 252,
        id_col: str = "ticker",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Run expanding-window out-of-sample prediction.

        At each date after the minimum training period, fits the model on
        all prior observations, then predicts next-period returns.

        Parameters
        ----------
        panel:
            Long-format panel DataFrame.
        feature_cols:
            Names of predictor columns (must exist in ``panel``).
        return_col:
            Name of the target variable column.
        min_train_periods:
            Minimum number of *dates* to include before making the first
            out-of-sample prediction.
        id_col:
            Stock identifier column name.
        date_col:
            Date column name.

        Returns
        -------
        pd.DataFrame
            Columns: ``[date_col, id_col, 'predicted', 'realized']``.
        """
        panel = panel.copy().sort_values([date_col, id_col]).reset_index(drop=True)
        dates = sorted(panel[date_col].unique())

        if len(dates) <= min_train_periods:
            raise ValueError(
                f"Panel has {len(dates)} dates but min_train_periods={min_train_periods}. "
                "Reduce min_train_periods or provide more data."
            )

        records = []
        for i, t in enumerate(dates[min_train_periods:], start=min_train_periods):
            train_dates = dates[:i]
            train_mask = panel[date_col].isin(train_dates)
            test_mask = panel[date_col] == t

            train = panel[train_mask].dropna(subset=feature_cols + [return_col])
            test = panel[test_mask].dropna(subset=feature_cols)

            if len(train) < 10 or len(test) == 0:
                continue

            X_train = train[feature_cols].values
            y_train = train[return_col].values
            X_test = test[feature_cols].values

            model = self._make_pipeline()
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                self._last_model = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model fit failed at date %s: %s", t, exc)
                continue

            for j, (pred, row) in enumerate(zip(preds, test.itertuples())):
                realized = getattr(row, return_col, np.nan)
                records.append(
                    {
                        date_col: t,
                        id_col: getattr(row, id_col),
                        "predicted": float(pred),
                        "realized": float(realized),
                    }
                )

            if i % 50 == 0:
                logger.info("Expanding window: processed %d / %d dates.", i, len(dates))

        return pd.DataFrame(records)

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Return the coefficients of the last fitted model (if linear)."""
        if self._last_model is None:
            return None
        if isinstance(self._last_model, Pipeline):
            step = self._last_model.named_steps.get("model")
        else:
            step = self._last_model
        return getattr(step, "coef_", None)
