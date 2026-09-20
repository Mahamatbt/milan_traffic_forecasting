# Related Work and Model Selection

> **Before submitting:** read the sources cited here rather than relying on this
> summary. Every citation below was verified to exist, with authors, venue and DOI
> checked against the publisher or arXiv record, but the characterisations are drawn
> from abstracts and summaries rather than full readings. A viva question on any of
> these papers would expect first-hand knowledge.

---

## 1. Why this problem has its own literature

Forecasting mobile network traffic matters operationally rather than academically:
operators use short-horizon predictions to allocate radio resources, balance load
between cells, and put lightly loaded base stations to sleep. Wang *et al.* [5] survey
the field and identify these as the recurring motivations, alongside the observation
that 5G deployment has made both the data volumes and the traffic patterns
substantially harder to model than in earlier generations.

The dataset used in this study occupies an unusual position. Barlacchi *et al.* [1]
released it as part of the Telecom Italia Big Data Challenge, and because it is public,
large, and spatially resolved, it has become a common benchmark. That means results on
it are comparable across papers in a way that proprietary operator data never is — and
it also means the obvious modelling approaches have already been tried. Two studies are
directly relevant here because they use the same Milan grid:

- **Zhang and Patras [3]** forecast network-wide traffic with a spatio-temporal neural
  network, treating the 100 × 100 grid as an image sequence and combining convolutional
  and recurrent components. They report up to 61% lower prediction error than
  established baselines, and target horizons up to ten hours.
- **Santos *et al.* [4]** address the short-horizon problem this study addresses. They
  compare LSTM and GRU against Random Forest and Decision Tree on two months of Milan
  data, cluster cells by activity level using k-means, and tune by grid search. They
  report that both recurrent architectures model the within-day and across-month
  seasonality effectively, evaluating with RMSE and MAE.

Santos *et al.* is the closest published analogue to this study, and the clustering step
is worth noting: they too found that cells differ enough that treating them uniformly is
unsatisfactory. This study reaches the same conclusion from a different direction — the
three highest-traffic cells here lie within 470 m of one another and behave almost
identically, which is why the areas modelled span ranks 1, 424 and 109 rather than the
top three.

---

## 2. Statistical approaches, and why the obvious one fails here

ARIMA and its seasonal extension SARIMA are the standard statistical baselines for this
problem, and much of the cellular forecasting literature positions itself against them.
Azari *et al.* [2] provide the most careful comparison: they evaluate LSTM against ARIMA
on real network traffic and find LSTM superior in general, but — importantly for model
selection — identify conditions under which ARIMA performs close to optimal at
considerably lower complexity. A study that discards statistical models on the strength
of the headline result alone would be ignoring the paper's actual conclusion.

Seasonal ARIMA is nevertheless not usable on this data, for a reason that is structural
rather than empirical. At 10-minute resolution the daily period is s = 144, and SARIMA's
seasonal differencing and parameter estimation become computationally intractable at that
seasonal length. More fundamentally, SARIMA accommodates **one** seasonal period. The
decomposition performed in this study finds two: daily seasonality carries 82.9% of the
variance and weekly a further 10.0%, with strengths of 0.959 and 0.746 respectively. A
single-period model cannot represent both.

The standard remedy is **dynamic harmonic regression** — Fourier terms at the required
seasonal frequencies entered as exogenous regressors, with ARIMA errors capturing what
remains. Hyndman and Athanasopoulos [6] set this out as the recommended approach
precisely for long seasonal periods and multiple seasonalities, noting that the number of
Fourier terms controls the smoothness of the seasonal shape and that the method scales to
seasonal periods where SARIMA does not.

The periodogram computed here supports that framing and constrains it: power is
concentrated at 24.00 h, with a clear secondary peak at **12.00 h**. The daily cycle is
therefore not a single sinusoid, and a harmonic regression with K₁ = 1 would fit the
envelope while missing the shape. At least two daily harmonics are required.

The decomposition method itself comes from this literature. Bandara *et al.* [7]
introduce MSTL as an extension of STL to multiple seasonal patterns, motivated by exactly
the high-sampling-rate data this study works with, and report competitive accuracy at
lower computational cost than alternatives. It is the tool used in the exploratory
analysis and the source of the variance shares quoted above.

---

## 3. Deep learning, and what it does and does not buy

The survey by Wang *et al.* [5] documents the field's movement toward deep learning, and
the direction of travel is not in dispute. LSTM and GRU dominate the single-cell
forecasting literature; convolutional and graph-based architectures dominate where
spatial structure is exploited.

Three qualifications matter for this study.

**The spatial architectures solve a different problem.** Zhang and Patras [3] achieve
their gains by using the whole grid — neighbouring cells are informative because users
move between them. This study is univariate by design, forecasting each area from its own
history alone. That is a real limitation rather than a neutral choice, and the correct
framing in the report is that spatial and multivariate extensions are the most promising
direction for future work, with [3] as the evidence that they help.

**Recurrent architecture choice is not settled.** Santos *et al.* [4] compare LSTM and
GRU on this dataset and find both effective. GRU is the cheaper of the two, with fewer
gates and parameters. Selecting LSTM over GRU is therefore a defensible default rather
than a finding, and should be stated as such.

**Data hunger is a recognised failure mode.** The survey literature notes that LSTM and
CNN approaches generalise poorly when data is sparse, overfitting rather than adapting.
This study trains on 5,472 observations per area — adequate, but not abundant, and enough
to justify early stopping and a dropout search rather than assuming a large model will
behave.

There is also a failure mode specific to one-step-ahead forecasting that the literature
warns about and that this data makes acute. The lag-1 autocorrelation of the highest-
traffic area measured here is **0.987**. A neural model can reach a low error simply by
learning to output its most recent input, and will look successful while having learned
nothing. This is why the evaluation protocol reports a persistence baseline in every
table and includes an explicit lag-1 copying check.

---

## 4. Gradient-boosted trees as a serious competitor

The strongest evidence against assuming a neural model will win comes from outside the
networking literature. The M5 forecasting competition, reviewed by Makridakis *et al.*
[8], concluded that machine-learning methods outperformed classical statistical ones —
and the methods that did so were overwhelmingly gradient-boosted trees, with LightGBM [9]
the most-used model among the winning entries. Neural approaches appeared mainly in
hybrid combinations rather than winning outright.

Gradient boosting on lag features is therefore not a weak third option included for
variety. It is the approach that most recently beat both alternatives at scale, and on
this data it has a concrete advantage: the features it needs are directly indicated by
the measured autocorrelation. The ACF of the study area shows local maxima at lags 144,
288, 432, 720, 864 and 1008 — every multiple of the daily period, with the weekly lag
among them — so the lag set is identified empirically rather than assumed. Feature
importances then provide an interpretability check that neither of the other two models
offers.

Santos *et al.* [4] included Random Forest and Decision Tree on this dataset, but not
gradient boosting, which leaves a genuine gap this study can occupy.

---

## 5. Evaluation practice

Two points from the forecasting literature shape the evaluation design.

Hyndman and Koehler [10] show that many commonly used accuracy measures are degenerate in
situations that occur routinely, and propose the mean absolute scaled error as a
general-purpose alternative. MASE is scale-free and interpretable: below 1 means the
model beats a naive one-step forecast on the training data. That property is necessary
here, because the three areas modelled differ by roughly 5× in mean volume, and raw MAE
or RMSE cannot be compared across them. MASE is therefore the metric used for the
cross-area comparison, with MAE, RMSE and MAPE reported per area as the brief requires.

MAPE needs a stated caveat rather than silent use: it is undefined at zero and unstable
near it. The three areas modelled here have minima of 47.3, 55.8 and 80.2, so no
near-zero denominator arises — but the check is reported rather than assumed.

---

## 6. The three models, and why

### Model 1 — Dynamic harmonic regression with ARIMA errors

**What it is.** Fourier terms at the daily (144) and weekly (1008) periods as exogenous
regressors in a SARIMAX model, with ARIMA errors capturing the residual autocorrelation.

**Why here.** It is the approach [6] recommends for exactly this situation — long
seasonal periods, more than one of them — and it is the only one of the three that
represents the seasonal structure explicitly rather than learning it. The measured
evidence sets its parameters rather than leaving them to be guessed: two seasonal periods
present (82.9% and 10.0% of variance) fix the two frequencies; the 12.00 h spectral peak
requires K₁ ≥ 2; and ADF and KPSS agreeing on stationarity across the raw series and both
differences means d = 0, with no differencing for a stochastic trend.

**Limitations.** Fourier terms impose a *fixed* seasonal shape: the model cannot express a
daily profile that changes across the period, and the absent-cell analysis shows activity
declining measurably from late November onward. It is linear in the regressors, so it
cannot capture the interaction between time-of-day and day-of-week that the weekend ratios
in the exploratory analysis suggest exists. It is also the slowest of the three at
inference, since each step appends an observation to the state-space filter.

**Criticism to state honestly.** Azari *et al.* [2] find ARIMA-family methods generally
inferior to LSTM. Including one is justified by their finding that it can approach optimal
at lower complexity, and by its interpretability — not by an expectation that it will win.

### Model 2 — LSTM

**What it is.** A stacked LSTM over a window of recent observations, optionally with
calendar features, with a linear output head, early stopping on validation MAE and
best-checkpoint restore.

**Why here.** It is the dominant architecture in this literature [5] and has been shown
effective on this dataset specifically [4]. It makes no assumption about seasonal form, so
it can in principle capture the nonlinear, state-dependent behaviour the other two cannot:
the exploratory analysis shows the residual after removing trend and both seasonalities is
still 3.6% of variance and strongly heteroscedastic, with spread varying 25× across the
day.

**Limitations.** Most expensive to train and to tune, and the only model whose result
varies with random seed — which is why the protocol requires three seeds and reports
mean ± standard deviation. Least interpretable of the three. Most exposed to the
persistence-collapse risk described above, given a lag-1 autocorrelation of 0.987.

**Criticism to state honestly.** GRU would be a reasonable and cheaper alternative [4], and
choosing LSTM is a convention rather than a result. The 5,472 training observations per
area are modest for an architecture with this capacity.

### Model 3 — LightGBM on causal lag features

**What it is.** Gradient-boosted trees [9] over lagged values, rolling statistics and
calendar features, all computed causally, with early stopping on the validation split.

**Why here.** The M5 evidence [8] makes it the strongest prior favourite, not a filler
third model. Its feature set is determined by measurement rather than convention — the ACF
peaks at 144, 288, 432, 720, 864 and 1008 give the lags directly. It handles the
interaction between time-of-day and day-of-week natively, which the harmonic model cannot.
It trains in seconds, and its feature importances provide a cross-check on the exploratory
analysis: if the model does not rely on the lags the ACF identified, one of the two
analyses is wrong.

**Limitations.** It cannot extrapolate beyond the range of its training targets, which
matters for the stress split containing Christmas and New Year. It has no notion of
sequence beyond the lags it is given explicitly, so anything outside the chosen lag set is
invisible to it. Feature engineering does the work that the LSTM does implicitly, which
makes its performance partly a measure of the engineering rather than the model.

### Why these three together

They differ in what they assume rather than only in implementation: the harmonic model
imposes seasonal structure explicitly, the LSTM learns representation from raw sequence,
and LightGBM learns from engineered features. A comparison across those three positions
says something about the problem; a comparison across three neural variants would mostly
say something about architecture search. Persistence and seasonal-naive baselines appear
in every table as the floor, since with a lag-1 autocorrelation of 0.987 persistence is a
demanding opponent rather than a formality.

---

## 7. Does the review change the line-up?

**No — but it changes what the report should claim about it.**

The three models survive, and each is now justified by a source as well as by a
measurement. Three things the review changed:

1. **LightGBM's status.** It was originally the third choice for architectural diversity.
   The M5 result [8] makes it the strongest prior favourite. The report should not present
   it as the token non-neural model.
2. **The expected result is no longer obvious.** [2] favours LSTM over ARIMA; [8] favours
   boosted trees over both. The literature does not predict a winner here, which makes the
   comparison genuinely open rather than a confirmation exercise.
3. **The univariate restriction is now a stated limitation with evidence behind it.**
   Zhang and Patras [3] show that spatial information helps materially on this exact grid.
   That belongs in Conclusion and Future Work as a known and quantified gap, not as an
   afterthought.

One alternative was considered and rejected: **GRU in place of LSTM**. Santos *et al.* [4]
found both effective on this data and GRU is cheaper. Substituting it would be defensible,
but it would not change what the comparison tests — both occupy the same "learn from raw
sequence" position — and the plan requires the three models to be meaningfully different.
It is noted in the report as a reasonable alternative rather than adopted.

---

## References

See `report/references.md` for the IEEE-formatted list.
