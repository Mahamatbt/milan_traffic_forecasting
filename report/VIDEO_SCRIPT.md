# Video script — verbatim

**Timed at 1,387 spoken words.** The section timings below assume 145 words per minute,
which is a normal pace for reading a prepared script, and end at **9:33**.

| Your pace | Full script | Without the optional paragraphs |
|---|---|---|
| 155 wpm (brisk) | 8:56 | 8:13 |
| 145 wpm (assumed) | 9:33 | 8:47 |
| 140 wpm (measured) | 9:54 | 9:06 |
| 125 wpm (slow) | 11:05 — **over** | 10:12 — **over** |

Three paragraphs are marked *[Optional]*. Each is self-contained, so dropping it leaves the
surrounding argument intact. **Time yourself reading section 1 aloud before you record**:
if it takes more than 45 seconds you are under 145 wpm, so cut the optional paragraphs. If
it takes more than 55 seconds you will overrun even trimmed, so slow the *delivery* down
and cut further rather than racing.

**Tabs to open before you start, in this order:**

1. `report/REPORT.pdf` (top of document)
2. `results/figures/eda/02_spatial_totals.png`
3. `results/figures/eda/05_mstl_decomposition.png`
4. `results/experiments.csv`
5. `results/figures/cross_correlation_test_5161.png`
6. `src/models/gbm.py` (scrolled to the top)
7. `results/figures/forecast_test_5059_lstm.png`
8. A terminal in the project root, with `python run.py test` **already finished** so the
   result is on screen

**SAY** is read as written. **SHOW** is what should be on screen while you say it.

---

## 1. Introduction — 0:00 to 0:43

**SHOW:** `report/REPORT.pdf`, title page.

**SAY:**

> Hello. This is my comparative analysis of sequential models for mobile network traffic
> forecasting, using the Telecom Italia dataset for Milan.
>
> The problem is operational. Operators commit capacity before demand arrives, and the cost
> of getting it wrong is lopsided — under-provision and service degrades, over-provision and
> you waste energy.
>
> The data covers Milan as a hundred by hundred grid, every ten minutes over two months.
> That's 8,928 observations for each of ten thousand cells.
>
> My question has two halves. How do sequential models compare for one-step-ahead
> forecasting, and does the answer change depending on the area. The second half shaped my
> whole design.

---

## 2. Data handling and memory — 0:43 to 1:42

**SHOW:** PDF, Table 1 (ingest strategies).

**SAY:**

> The raw data is 19.38 gibibytes of text. After aggregation it's a 340 megabyte matrix. So
> the problem isn't storing the result, it's getting there without holding more than a day
> in memory.
>
> I measured three strategies rather than assuming. Naive pandas peaked at 633 mebibytes,
> chunked pandas at 173, Polars at 552.
>
> But the headline is the scaling, not the peak. All 62 days the naive way needs 17.9
> gibibytes resident. Both optimised paths hold one 5.5 mebibyte block however many days you
> process — constant in dataset size, against linear.

**SHOW:** Scroll to "A measurement artefact worth reporting".

**SAY:**

> One thing changed how I measure. My first benchmark ran all three strategies in one
> process and reported 544, 184 and 384 mebibytes for identical work. Python doesn't return
> freed memory to the OS, so whichever ran first paid for growing the heap. Every
> measurement now runs in its own subprocess.

*[Optional — cut this paragraph if you are running long.]*

---

## 3. Exploratory analysis — 1:42 to 3:13

**SHOW:** `results/figures/eda/02_spatial_totals.png`.

**SAY:**

> This is total activity across the grid on a log scale.
>
> The brief asks for three areas, and the obvious choice is the three busiest cells. But
> converting their IDs to coordinates put all three within 470 metres of each other, in one
> hotspot by the Duomo.
>
> Using them would have measured the same traffic regime three times, which can't answer how
> performance varies across areas — half my research question. So the areas I model are
> ranked one, four hundred and twenty-four, and one hundred and nine.
>
> And this is more than a technicality. Those three adjacent cells have weekend-to-weekday
> ratios from 0.425 to 1.384 — a factor of three between neighbours. Traffic character
> varies at a finer scale than volume rank suggests.

*[Optional — cut this paragraph if you are running long.]*

**SHOW:** `results/figures/eda/05_mstl_decomposition.png`.

**SAY:**

> This is a multi-seasonal decomposition. The daily cycle carries 82.9 percent of the
> variance, the weekly another 10.
>
> Those numbers drove my first model choice. Two seasonal periods operate at once and a
> seasonal ARIMA represents only one — and at ten-minute resolution its daily period of 144
> steps is intractable anyway. So the decomposition ruled out the obvious statistical model
> and pointed me at dynamic harmonic regression.
>
> The periodogram also has a twelve-hour peak beside the twenty-four hour one, so the daily
> cycle isn't a single sine wave and I needed at least two harmonics.

---

## 4. Models and protocol — 3:13 to 4:10

**SHOW:** PDF, Section 5.4.

**SAY:**

> I compared three models, chosen to differ in what they assume. The harmonic regression is
> told the seasonal structure explicitly. LightGBM is told which lags to look at. The LSTM
> assumes nothing — it sees a raw window and finds the structure itself.
>
> The lags came from the data: the autocorrelation peaks at 144, 288, 432, 720, 864 and
> 1008, so those are the lags LightGBM gets. Measurement, not convention.
>
> My persistence baseline is not a straw man. The lag-one autocorrelation is 0.987, so
> repeating the last observation gives an R-squared above 0.94. That's the bar deciding
> whether any of this complexity was worth paying for.
>
> On protocol: splits are chronological, never random. Transforms fit on training data only,
> and the scaler raises an error if anything tries to refit it. All inference is walk-forward
> on real observed history.

---

## 5. Experimentation — 4:10 to 4:45

**SHOW:** `results/experiments.csv`, scrolling.

**SAY:**

> Every model fitted during tuning is logged here — 101 of them. Each row records the
> hyperparameters, validation metrics, wall time, parameter count, the git commit, and a
> written rationale. The log refuses a row where that rationale is empty.

**SHOW:** Widen the `rationale_for_next_change` column on one row and read it aloud.

**SAY:**

> I searched one axis at a time so each decision is attributable, with two exceptions.
> Harmonic order was chosen by AICc, because walk-forwarding fifteen candidates costs far
> more than an in-sample criterion. LightGBM used Optuna, because its tree parameters trade
> off against each other.

---

## 6. Results — 4:45 to 6:17

**SHOW:** PDF, Table 10 (MASE by area).

**SAY:**

> Here are the test-week results. I use MASE because the areas differ by an order of
> magnitude in volume, so raw error isn't comparable across them.
>
> The smallest model won. The harmonic regression has the best mean MASE at 0.213 and takes
> two of three areas — with 22 parameters, against twenty thousand for LightGBM and two
> hundred thousand for the LSTM.
>
> And the uncomfortable result. Averaged across areas, only two of my three models beat
> persistence. Both LSTM variants do not. Reporting model errors without that baseline would
> have shown a respectable result that's worse than doing nothing.
>
> There are two LSTM rows. The first is the mean of three seeds' errors. The second is the
> error of their averaged forecast, better because averaging can't increase absolute error —
> but costing three times the training, inference and parameters.
>
> The three-seed protocol earned its place. On one area the LSTM scores 21.16 MAE plus or
> minus 4.59 — a 22 percent relative standard deviation.

*[Optional — cut this paragraph if you are running long.]*

**SHOW:** PDF, Table 12 (timing).

**SAY:**

> And cost inverts between training and deployment. The harmonic model is cheapest to train
> at 18 seconds, and about 1,825 times the most expensive to run, because appending each
> observation still runs a Kalman filter update. At city scale that rules it out —
> forecasting ten thousand cells once takes 12.4 minutes, longer than the interval you're
> predicting.

---

## 7. A technical decision in depth — 6:17 to 7:51

**SHOW:** `results/figures/cross_correlation_test_5161.png`.

**SAY:**

> I want to go into one decision properly.
>
> With a lag-one autocorrelation of 0.987, a model can score well just by repeating its most
> recent input — looking successful while having learned nothing. So I tested for it.
>
> The obvious test is to cross-correlate forecasts against observations and see where the
> peak falls. When I did, every model peaked at lag plus one, including my best one.
>
> But that isn't copying, it's causality. A one-step forecast is built only from data up to
> the previous step, so it cannot contain the innovation at the current step. It has to
> correlate slightly more with the previous value. A forecast peaking at lag zero on a
> series this persistent would be the suspicious case, because it would imply access to the
> present value.
>
> And the data settles it. Seasonal naive is the only model that peaks at lag zero, and it's
> comfortably the worst forecaster in the study.
>
> So my verdict rests on a different measure — the distance between each forecast and the
> persistence baseline, scaled by how far the series moves. Those ratios run from 0.54 to
> 1.14, against zero for persistence. No model collapsed.
>
> My first implementation keyed the verdict on the peak, which flagged all three real models
> and cleared the worst one. The corrected reasoning is in my tests so it can't drift back.

---

## 8. Failure analysis and limitations — 7:51 to 9:06

**SHOW:** PDF, Table 16 (stress split).

**SAY:**

> My stress split is the 23rd of December to the 1st of January. It holds four public
> holidays, and I never tuned on it.
>
> Persistence wins outright on two of three areas. LightGBM degrades by 69 to 140 percent,
> the LSTM by 78 to 103. The harmonic model stays within nine percent.

**SHOW:** `src/models/gbm.py`, module docstring.

**SAY:**

> I wrote this before running that split. It says a tree ensemble can't extrapolate beyond
> the range of its training targets, and names the holiday period as where that should bite.
> The prediction held — LightGBM is the worst non-naive model on all three areas.

**SHOW:** `results/figures/forecast_test_5059_lstm.png`.

**SAY:**

> The LSTM's failure is specific, not general. Its three worst six-hour windows all start
> around ten in the morning on consecutive weekdays, and its weekday error is 24.13 against
> 12.03 at weekends. It's missing the morning ramp.
>
> And the limitation I have to be honest about. I tuned hyperparameters on the
> highest-traffic area only, so that failure is partly a transfer result — my evidence can't
> separate an architectural limitation from a transfer failure. Per-area tuning is the most
> useful thing I'd do next.

---

## 9. Reproducibility and close — 9:06 to 9:33

**SHOW:** Terminal with the finished `python run.py test` output.

**SAY:**

> Finally, reproducibility. There are 501 tests, targeting what my conclusions depend on —
> leakage, and the correctness of the inference path. A clean clone into an empty virtual
> environment reproduces every metric I've reported exactly.
>
> To summarise. The smallest model won on the test week. No model beat persistence once the
> distribution shifted. And the cheapest model to train was the most expensive to deploy.
> Thank you.

---

## Before you submit

- [ ] Length is between 7 and 10 minutes
- [ ] Listen to the whole recording once — audio matters more than video
- [ ] Every figure is legible at the recorded resolution
- [ ] No notifications, unrelated tabs or credentials visible
- [ ] Test the sharing link in a private browser window
- [ ] Paste the URL into `report/REPORT.md` and `report/references.md`, then run
      `python run.py pdf` to rebuild the PDF with the link in it
