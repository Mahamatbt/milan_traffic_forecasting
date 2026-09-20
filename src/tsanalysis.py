"""Time-series characterisation of a single area's traffic.

These are the analyses that decide the modelling approach, so each returns
numbers the report can quote rather than only a picture:

``decompose``
    MSTL with daily (144) and weekly (1008) periods. A single seasonal period
    cannot represent both, and classical decomposition cannot handle two at
    once, which is itself an argument for the model line-up.
``stationarity_table``
    ADF and KPSS on the raw series, its first difference and its seasonal
    difference. The two tests are reported together deliberately: they have
    opposite null hypotheses, so agreement is informative and disagreement
    diagnoses the awkward middle cases rather than hiding them.
``autocorrelation``
    ACF and PACF far enough out to see the weekly lag, with confidence bands.
``periodogram_peaks``
    Which cycles actually carry power, as independent corroboration of the
    periods assumed by the decomposition.
``seasonal_naive_anomalies``
    Points a seasonal-naive predictor misses badly, which is where the
    forecasting models are expected to struggle too.

Strength of seasonality follows Wang, Smith and Hyndman (2006): the variance of
a component alone is misleading because components are correlated and their
variances do not sum to the total.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DecompositionResult",
    "StationarityResult",
    "decompose",
    "stationarity_table",
    "autocorrelation",
    "periodogram_peaks",
    "seasonal_naive_anomalies",
    "rolling_stats",
]


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------


@dataclass
class DecompositionResult:
    """MSTL components and the share of variation each explains."""

    periods: list[int]
    trend: np.ndarray = field(repr=False)
    seasonal: dict[str, np.ndarray] = field(repr=False)
    residual: np.ndarray = field(repr=False)
    observed: np.ndarray = field(repr=False)
    variance_share: dict[str, float] = field(default_factory=dict)
    strength: dict[str, float] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        """The quotable numbers, without the arrays."""
        return {
            "periods": self.periods,
            "variance_share": self.variance_share,
            "strength": self.strength,
            "residual_std": float(np.nanstd(self.residual)),
            "observed_std": float(np.nanstd(self.observed)),
        }

    def summary_lines(self) -> list[str]:
        lines = [f"MSTL periods: {self.periods}"]
        for name, share in self.variance_share.items():
            strength = self.strength.get(name)
            suffix = f", strength {strength:.3f}" if strength is not None else ""
            lines.append(f"  {name:<12} variance share {share:6.1%}{suffix}")
        return lines


def _seasonal_strength(component: np.ndarray, residual: np.ndarray) -> float:
    """Strength of a seasonal or trend component, in ``[0, 1]``.

    Defined as ``1 - Var(residual) / Var(component + residual)``. Near 1 means
    the component dominates what remains after the others are removed; near 0
    means it explains nothing the residual does not.
    """
    combined = np.nanvar(component + residual)
    if combined <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.nanvar(residual) / combined))


def decompose(series: np.ndarray, periods: tuple[int, ...]) -> DecompositionResult:
    """Run MSTL with the given seasonal periods.

    Args:
        series: One area's traffic, evenly spaced with no gaps.
        periods: Seasonal periods in samples, e.g. ``(144, 1008)`` for daily and
            weekly cycles at 10-minute resolution.

    Returns:
        A :class:`DecompositionResult` with components and variance shares.

    Raises:
        ValueError: If the series is shorter than two full cycles of the
            longest period, where the decomposition would be meaningless.
    """
    from statsmodels.tsa.seasonal import MSTL

    values = np.asarray(series, dtype=np.float64)
    longest = max(periods)
    if values.size < 2 * longest:
        raise ValueError(
            f"series of {values.size} points is too short for period {longest}; "
            f"at least {2 * longest} are needed"
        )

    result = MSTL(values, periods=periods).fit()
    seasonal_raw = np.asarray(result.seasonal)
    if seasonal_raw.ndim == 1:
        seasonal_raw = seasonal_raw[:, None]

    seasonal = {
        f"seasonal_{period}": seasonal_raw[:, index] for index, period in enumerate(periods)
    }
    trend = np.asarray(result.trend)
    residual = np.asarray(result.resid)

    observed_var = np.nanvar(values)
    components = {"trend": trend, **seasonal, "residual": residual}
    variance_share = {
        name: float(np.nanvar(component) / observed_var) if observed_var else 0.0
        for name, component in components.items()
    }
    strength = {
        name: _seasonal_strength(component, residual)
        for name, component in components.items()
        if name != "residual"
    }

    return DecompositionResult(
        periods=list(periods),
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        observed=values,
        variance_share=variance_share,
        strength=strength,
    )


# --------------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------------


@dataclass
class StationarityResult:
    """ADF and KPSS applied to one transformation of a series."""

    transform: str
    n_obs: int
    adf_stat: float
    adf_pvalue: float
    adf_lags: int
    adf_rejects_unit_root: bool
    kpss_stat: float
    kpss_pvalue: float
    kpss_lags: int
    kpss_rejects_stationarity: bool

    @property
    def verdict(self) -> str:
        """Plain reading of the two tests taken together.

        ADF's null is a unit root; KPSS's null is stationarity. Agreement is
        the informative case; disagreement usually means the series is close to
        the boundary or has structure neither test models.
        """
        if self.adf_rejects_unit_root and not self.kpss_rejects_stationarity:
            return "stationary (both agree)"
        if not self.adf_rejects_unit_root and self.kpss_rejects_stationarity:
            return "non-stationary (both agree)"
        if self.adf_rejects_unit_root and self.kpss_rejects_stationarity:
            return "conflicting: difference-stationary or heteroscedastic"
        return "conflicting: inconclusive, likely near-unit-root"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["verdict"] = self.verdict
        return row


def _one_stationarity(values: np.ndarray, label: str, alpha: float) -> StationarityResult:
    """Run ADF and KPSS on one already-transformed series."""
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    clean = values[np.isfinite(values)]
    adf_stat, adf_p, adf_lags, *_ = adfuller(clean, autolag="AIC")
    with warnings.catch_warnings():
        # KPSS clamps its p-value to the tabulated range and warns each time;
        # the clamping is expected and the statistic is still reported.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, kpss_lags, _ = kpss(clean, regression="c", nlags="auto")

    return StationarityResult(
        transform=label,
        n_obs=int(clean.size),
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        adf_lags=int(adf_lags),
        adf_rejects_unit_root=bool(adf_p < alpha),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        kpss_lags=int(kpss_lags),
        kpss_rejects_stationarity=bool(kpss_p < alpha),
    )


def stationarity_table(
    series: np.ndarray,
    *,
    seasonal_period: int,
    alpha: float = 0.05,
) -> list[StationarityResult]:
    """Test the raw series, its first difference and its seasonal difference.

    Args:
        series: One area's traffic.
        seasonal_period: Lag for the seasonal difference, e.g. 144 for daily.
        alpha: Significance level for both tests.

    Returns:
        Three :class:`StationarityResult` entries, forming the 3x2 table the
        report presents.
    """
    values = np.asarray(series, dtype=np.float64)
    return [
        _one_stationarity(values, "raw", alpha),
        _one_stationarity(np.diff(values, n=1), "first difference", alpha),
        _one_stationarity(
            values[seasonal_period:] - values[:-seasonal_period],
            f"seasonal difference (lag {seasonal_period})",
            alpha,
        ),
    ]


# --------------------------------------------------------------------------
# Autocorrelation
# --------------------------------------------------------------------------


def autocorrelation(
    series: np.ndarray,
    *,
    nlags: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """ACF and PACF with confidence bands, plus the most prominent lags.

    Args:
        series: One area's traffic.
        nlags: Highest lag to compute. Should exceed the weekly period so the
            weekly peak is visible rather than assumed.
        alpha: Confidence level for the bands.

    Returns:
        Arrays for plotting and a ``notable_lags`` list for the discussion.
    """
    from statsmodels.tsa.stattools import acf, pacf

    values = np.asarray(series, dtype=np.float64)
    # PACF is solved from an nlags x nlags system, so it is both slow and
    # ill-conditioned at large lags; the ACF carries the long-range structure.
    pacf_lags = min(nlags, max(1, values.size // 2 - 1), 400)

    acf_values, acf_confint = acf(values, nlags=nlags, alpha=alpha, fft=True)
    pacf_values, pacf_confint = pacf(values, nlags=pacf_lags, alpha=alpha, method="ywm")

    # Local maxima of the ACF beyond lag 1, which is where a seasonal period
    # announces itself.
    interior = acf_values[1:]
    peaks = [
        int(i + 1)
        for i in range(1, interior.size - 1)
        if interior[i] > interior[i - 1] and interior[i] > interior[i + 1] and interior[i] > 0.2
    ]
    peaks.sort(key=lambda lag: -float(acf_values[lag]))

    return {
        "acf": acf_values,
        "acf_confint": acf_confint,
        "pacf": pacf_values,
        "pacf_confint": pacf_confint,
        "nlags": int(nlags),
        "pacf_lags": int(pacf_lags),
        "lag_1": float(acf_values[1]),
        "notable_lags": peaks[:10],
        "notable_lag_values": [float(acf_values[lag]) for lag in peaks[:10]],
    }


def rolling_stats(series: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Rolling mean and standard deviation, for a visual stationarity check.

    A drifting mean or a level-dependent spread shows up here more legibly than
    in a hypothesis test, and motivates the variance-stabilising transform.
    """
    values = np.asarray(series, dtype=np.float64)
    if window < 2 or window > values.size:
        raise ValueError(f"window {window} invalid for a series of {values.size}")

    kernel = np.ones(window) / window
    mean = np.convolve(values, kernel, mode="valid")
    mean_of_squares = np.convolve(values**2, kernel, mode="valid")
    variance = np.maximum(mean_of_squares - mean**2, 0.0)
    return {
        "index": np.arange(window - 1, values.size),
        "mean": mean,
        "std": np.sqrt(variance),
    }


# --------------------------------------------------------------------------
# Spectral and anomaly views
# --------------------------------------------------------------------------


def periodogram_peaks(
    series: np.ndarray,
    *,
    interval_minutes: int,
    top_n: int = 6,
) -> dict[str, Any]:
    """Dominant cycles by spectral power, expressed in hours.

    Independent corroboration that the periods handed to the decomposition are
    the ones actually present, rather than assumed from domain knowledge.

    Args:
        series: One area's traffic.
        interval_minutes: Sampling interval, for converting frequency to hours.
        top_n: How many peaks to return.

    Returns:
        Frequencies, power, and the strongest cycles with their periods in hours.
    """
    from scipy.signal import periodogram

    values = np.asarray(series, dtype=np.float64)
    samples_per_hour = 60.0 / interval_minutes
    frequencies, power = periodogram(values - values.mean(), fs=samples_per_hour)

    order = np.argsort(power)[::-1]
    peaks = []
    for index in order:
        if frequencies[index] <= 0:
            continue
        peaks.append(
            {
                "period_hours": float(1.0 / frequencies[index]),
                "frequency_per_hour": float(frequencies[index]),
                "power": float(power[index]),
            }
        )
        if len(peaks) >= top_n:
            break

    return {"frequencies": frequencies, "power": power, "peaks": peaks}


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-derived scale, falling back to the standard deviation.

    Median and MAD rather than mean and standard deviation, because the
    anomalies being detected would otherwise inflate the very scale used to
    detect them.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(values))
    return median, scale


def seasonal_naive_anomalies(
    series: np.ndarray,
    *,
    seasonal_period: int,
    z_threshold: float = 4.0,
    per_phase_scale: bool = True,
) -> dict[str, Any]:
    """Points where a seasonal-naive predictor fails unusually badly.

    These are the intervals every model in the study is likely to miss, so
    identifying them in advance turns the later failure analysis from an
    after-the-fact excuse into a prediction that can be checked.

    Why the scale is estimated per phase
    ------------------------------------
    Forecast errors on this data are strongly heteroscedastic: on square 5161
    the seasonal-naive residual's standard deviation varies 25x between the
    quietest and busiest hour of the day. A single global scale is therefore set
    by the quiet hours and flags every busy one -- at ``z > 4`` it marked 12.7%
    of all points, which identifies nothing.

    Estimating the scale separately for each position in the seasonal cycle asks
    the question that actually matters: is this point unusual *for this time of
    day*. With 62 days of history each phase has 62 observations, enough for a
    stable median and MAD.

    Args:
        series: One area's traffic. Pass a variance-stabilised series (log1p)
            unless there is a reason not to; the residuals are far closer to
            homoscedastic.
        seasonal_period: Lag the naive predictor copies from.
        z_threshold: Robust z-score above which a point counts as anomalous.
        per_phase_scale: Estimate the scale per position in the cycle. Set
            False only to reproduce the naive global-scale behaviour.

    Returns:
        Indices into the original series, residuals, z-scores and the scales used.
    """
    values = np.asarray(series, dtype=np.float64)
    residual = values[seasonal_period:] - values[:-seasonal_period]
    if residual.size == 0:
        raise ValueError(f"series of {values.size} is shorter than period {seasonal_period}")

    z_scores = np.zeros_like(residual)
    scales = np.zeros_like(residual)

    if per_phase_scale:
        phase = (np.arange(residual.size) + seasonal_period) % seasonal_period
        for position in range(seasonal_period):
            mask = phase == position
            if not mask.any():
                continue
            median, scale = _robust_scale(residual[mask])
            scales[mask] = scale
            z_scores[mask] = np.abs(residual[mask] - median) / scale if scale else 0.0
    else:
        median, scale = _robust_scale(residual)
        scales[:] = scale
        z_scores = np.abs(residual - median) / scale if scale else z_scores

    flagged = np.flatnonzero(z_scores > z_threshold)
    return {
        "index": flagged + seasonal_period,
        "residual": residual,
        "z_scores": z_scores,
        "threshold": float(z_threshold),
        "scale": float(np.median(scales)),
        "scale_per_phase": per_phase_scale,
        "n_flagged": int(flagged.size),
        "fraction_flagged": float(flagged.size / residual.size),
    }
