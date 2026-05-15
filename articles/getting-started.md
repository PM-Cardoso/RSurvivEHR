# Getting started with RSurvivEHR

**RSurvivEHR** trains transformer-based survival models on longitudinal
EHR data directly from R, with no Python experience required. The full
pipeline is:

    events + static ──▶ survivehr_pretrain()  ──▶ pt_model
                                                      │
                                          survivehr_save_model()  ←── save for reuse
                                                      │
    events + targets ─▶ survivehr_finetune()  ──▶ ft_model  (CR or SR)
                                                      │
    new events ────────▶ survivehr_predict()  ──▶ predictions

------------------------------------------------------------------------

## 1 Setup (one-time)

Install the package from GitHub:

``` r

# install.packages("remotes")
remotes::install_github("PM-Cardoso/RSurvivEHR")
```

``` r

# install.packages("pak")
pak::pkg_install("PM-Cardoso/RSurvivEHR")
```

Then set up the Python backend (only needed once):

``` r

library(RSurvivEHR)

# Creates the "RSurvivEHR" Python virtualenv and installs all dependencies.
# Safe to call repeatedly — it is a no-op if the environment already exists.
survivehr_setup()
```

------------------------------------------------------------------------

## 2 Input data

### Events data frame

One row per clinical event, sorted by age within each patient.

``` r

# 10-patient pre-training cohort.
# Patients 1–3 have CVD events so the vocabulary includes CVD —
# a prerequisite for CVD fine-tuning in Section 5.
events_pop <- data.frame(
  patient_id = c(rep(1,4), rep(2,6), rep(3,6), rep(4,4), rep(5,6),
                 rep(6,4), rep(7,4), rep(8,4), rep(9,6), rep(10,4)),
  event = c(
    "HYPERTENSION","STATIN","BP_CHECK","CVD",                          # p1 — CVD at 58
    "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HYPERTENSION","CVD",  # p2 — T2D at 48, CVD at 52
    "HYPERTENSION","BP_CHECK","STATIN","T2D","BP_CHECK","CVD",         # p3 — T2D at 63.5, CVD at 65.5
    "HYPERTENSION","STATIN","T2D","METFORMIN",                         # p4 — T2D at 47
    "HYPERTENSION","BP_CHECK","T2D","HBA1C","METFORMIN","STATIN",      # p5 — T2D at 51
    "HYPERTENSION","AMLODIPINE","BP_CHECK","STATIN",                   # p6 — censored
    "STATIN","T2D","HBA1C","METFORMIN",                                # p7 — T2D at 42
    "HYPERTENSION","BP_CHECK","STATIN","T2D",                          # p8 — T2D at 62
    "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HBA1C","STATIN",      # p9 — T2D at 50
    "STATIN","BP_CHECK","HYPERTENSION","T2D"                           # p10 — T2D at 49
  ),
  age = c(
    55.0, 55.5, 56.2, 58.0,
    44.0, 45.5, 48.0, 48.3, 50.5, 52.0,
    58.0, 59.5, 62.0, 63.5, 64.0, 65.5,
    45.0, 45.5, 47.0, 48.3,
    48.0, 49.0, 51.0, 52.0, 53.0, 54.5,
    60.0, 61.0, 62.3, 63.0,
    40.0, 42.0, 43.5, 44.8,
    58.0, 59.2, 60.0, 62.0,
    46.0, 47.5, 50.0, 52.0, 52.5, 54.0,
    44.0, 46.5, 47.0, 48.5
  ),
  value = c(
    NA,  NA,  148, NA,          # p1 : BP_CHECK = 148 mmHg
    NA,  145, NA,  NA,  NA, NA, # p2 : BP_CHECK = 145 mmHg
    NA,  158, NA,  NA,  162,NA, # p3 : BP_CHECK = 158 mmHg, 162 mmHg
    NA,  NA,  NA,  NA,          # p4
    NA,  152, NA,  68,  NA, NA, # p5 : BP_CHECK = 152 mmHg, HBA1C = 68 mmol/mol
    NA,  NA,  155, NA,          # p6 : BP_CHECK = 155 mmHg
    NA,  NA,  74,  NA,          # p7 : HBA1C = 74 mmol/mol
    NA,  145, NA,  NA,          # p8 : BP_CHECK = 145 mmHg
    NA,  138, NA,  NA,  71, NA, # p9 : BP_CHECK = 138 mmHg, HBA1C = 71 mmol/mol
    NA,  138, NA,  NA           # p10: BP_CHECK = 138 mmHg
  )
)
```

Required columns: `patient_id`, `event`, `age`.

> Continuous readings (blood pressure, HbA1c, BMI, etc.) go in the
> `value` column alongside the event that recorded them. Rows without a
> reading are `NA`. Set `value_weight > 0` in
> [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
> to train an auxiliary regression head on these values; leave at `0`
> (default) to use token sequences only.

### Static covariates data frame

One row per patient. Categorical columns (\< 80 % numeric) are one-hot
encoded automatically; numeric columns pass through unchanged.

> **Column naming**: `patient_id` is the only reserved name in the
> static table. Every other column can be named freely — **lowercase is
> recommended** for consistency with the rest of the package. The column
> names become the basis for encoded feature names (e.g. `sex` →
> `sex_F`, `sex_M` after one-hot encoding). By contrast, the **events**
> and **targets** tables require fixed column names (`patient_id`,
> `event`, `age`, `value`; and `patient_id`, `target_event`,
> `target_age`, `target_value` respectively) — uppercase aliases are
> accepted for backward compatibility but lowercase is the canonical
> form.

``` r

static_pop <- data.frame(
  patient_id    = 1:10,
  sex           = c("M","F","M","F","M","F","M","F","M","F"),
  ethnicity     = c("White","Asian","White","Black","White",
                    "Asian","White","White","Black","White"),
  imd           = c(3L, 1L, 5L, 2L, 4L, 3L, 1L, 5L, 2L, 4L),
  year_of_birth = c(1960L,1970L,1952L,1975L,1963L,1958L,1978L,1960L,1968L,1975L)
)
```

> **Important**: always pass the **same columns in the same order**
> across pretrain, fine-tune, and prediction. The model records the
> exact encoded column list at training time and raises a clear error if
> prediction data does not match.

### Validate your data

Run the validators before any training call to catch schema problems
early — wrong column names, unsorted ages, non-numeric values — with a
clear R error rather than a cryptic Python traceback.

``` r

survivehr_validate_events(events_pop)
survivehr_validate_static(static_pop)
# [OK] Events: 48 rows, 10 patients. Columns present, ages numeric and time-ordered.
# [OK] Static covariates: 10 patients, 4 covariate column(s): sex, ethnicity, imd, year_of_birth.
```

------------------------------------------------------------------------

## 3 Configuration

[`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
returns a named list consumed by all training functions.

``` r

cfg <- survivehr_config(
  block_size    = 64,   # context window — sequences are padded/truncated to this
  n_layer       = 2,    # transformer blocks
  n_head        = 2,    # attention heads (n_embd must be divisible by n_head)
  n_embd        = 64,   # embedding / hidden dimension
  dropout       = 0,    # dropout rate (increase for regularisation)
  learning_rate = 3e-4,
  epochs        = 10,
  batch_size    = 4,
  surv_layer    = "competing-risk",  # or "single-risk" — see Section 5
  value_weight  = 0.1,  # weight for value regression loss (0 = disabled)
  time_scale    = 5.0   # prediction window = 5 years (ages are in years)
)
```

Key parameters:

| Parameter | Description |
|----|----|
| `block_size` | Context window length — sequences are padded/truncated to this. |
| `surv_layer` | Controls the **backbone** time-to-next-event head during pre-training (and must remain consistent at fine-tune). Leave at `"competing-risk"` (default) for most use cases. The outcome-level head is controlled by `risk_model` in [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md) — see Section 5. |
| `time_scale` | **Sets both the age normalisation divisor and the prediction window length.** Ages are divided by this value before entering the model; the survival ODE evaluates over a normalised \[0, 1\] grid mapping back to \[0, `time_scale`\] in your age units. Use `5.0` for a 5-year window with ages in years, `1.0` for 1-year. Stored in every model bundle — [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md) reads it automatically. |
| `include_unk` | If `TRUE` (default), a `<UNK>` token is reserved for unseen events at prediction time. |

------------------------------------------------------------------------

## 4 Pre-training

Pre-training builds the vocabulary and trains the backbone transformer
on the full event history.

``` r

pt_model <- survivehr_pretrain(
  events            = events_pop,
  static_covariates = static_pop,
  config            = cfg
)

# Vocabulary ordered by frequency — most-common events get the smallest token IDs.
# With this dataset: HYPERTENSION (10), BP_CHECK (9), STATIN (9), T2D (8),
# METFORMIN (5), CVD (3), HBA1C (3), AMLODIPINE (1).
pt_model$event_vocab
```

### Saving the pretrained model

Save the backbone now so you can load it later and fine-tune on
different outcome definitions without re-training from scratch.

``` r

survivehr_save_model(pt_model, "pretrain_backbone.pt")

# Reload at any time — vocabulary, time_scale, and token policy are all preserved
pt_model <- survivehr_load_model("pretrain_backbone.pt")
```

### Pre-train predictions — multi-step next events

The pretrain model autoregressively generates future events for each
patient. Set `max_new_tokens` to control how many steps ahead to
predict. Each generated step is returned as a separate row with a `step`
column, so a `max_new_tokens = 3` call gives up to 3 rows per patient.

``` r

# Generate the next 3 predicted events for every patient
pt_preds <- survivehr_predict(pt_model, events_pop, static_pop, max_new_tokens = 10)
head(pt_preds)
# Columns: patient_id, step, generated_token, generated_event, generated_age,
#          generated_value
#
# step            : 1 = next event, 2 = event after that, 3 = third predicted event
# generated_event : decoded event name (e.g. "HYPERTENSION", "<UNK>")
# generated_age   : predicted age in the same units as the input age column
# generated_value : predicted numeric value (NaN for non-measurement events)
```

### Pre-train predictions — predicted measurement value

[`survivehr_predict_value()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict_value.md)
queries the backbone’s Gaussian regression head to predict the numeric
value that would be recorded alongside a specific clinical event. For
example, `"BP_CHECK"` carries a blood-pressure reading in `value`; the
model learns to predict that reading from the patient’s history.

Remove the event you are querying from the context first (same
leakage-free logic as fine-tuning).

``` r

# Predict the blood pressure reading at the next BP_CHECK for each patient
bp_preds <- survivehr_predict_value(pt_model, events_pop, "BP_CHECK", static_pop)
print(bp_preds)
# Columns: patient_id, outcome_event, predicted_value_mean, predicted_value_sd
#
# predicted_value_mean : mean of the Gaussian distribution
# predicted_value_sd   : standard deviation of the Gaussian distribution
# NaN is returned for events that never had a numeric value at pre-training
# time (e.g. "CVD", which is a discrete diagnosis with no value).
```

> **When are predictions meaningful?** Only events that appeared with
> non-`NA` `value` entries at pre-training time will return non-NaN
> predictions. In this vignette, `BP_CHECK`, `HBA1C` are the only such
> events. All others return `NaN`.

> **Value scale**: the model stores and predicts values in exactly the
> units you supply — no internal standardisation is applied. When
> different event types carry measurements on very different scales
> (e.g. BP ~140 mmHg vs HbA1c ~70 mmol/mol), pre-normalise the `value`
> column before training and back-transform predictions afterwards. See
> the *Advanced topics* vignette for a worked example.

------------------------------------------------------------------------

## 5 Fine-tuning

Fine-tuning requires a `targets` data frame labelling, for each patient,
which outcome was observed (or the last observed non-outcome event if
censored). Two leakage conditions must both be satisfied when building
the context:

1.  **Remove outcome event codes** from context — the outcome is
    supplied only in `targets`.
2.  **Remove events after the outcome age** — events that occurred after
    the labelled outcome would not be available at prediction time.

Because condition 2 requires knowing each patient’s `target_age`, define
`targets` **before** filtering the context.

We use **patients 1–6** as the fine-tuning cohort.

``` r

ft_static <- static_pop[static_pop$patient_id %in% 1:6, ]
survivehr_validate_static(ft_static)
# [OK] Static covariates: 6 patients, 4 covariate column(s): sex, ethnicity, imd, year_of_birth.
```

### 5a Competing-risk fine-tuning

Use `risk_model = "competing-risk"` when two or more outcomes
**compete** — i.e. the occurrence of one prevents the other from being
observed. Here we model **CVD vs T2D**: patients 1–6 each experience
exactly one of the two outcomes first, or are censored if neither
occurred.

| Patient | Observed event | Why |
|----|----|----|
| p1 | CVD at 58.0 | CVD case — no prior T2D |
| p2 | T2D at 48.0 | T2D occurred first (CVD followed at 52.0 but is the competing event) |
| p3 | T2D at 63.5 | T2D occurred first (CVD followed at 65.5) |
| p4 | T2D at 47.0 | T2D case |
| p5 | T2D at 51.0 | T2D case |
| p6 | STATIN at 63.0 | Censored — neither CVD nor T2D observed |

Both outcome codes are removed from the context so neither leaks into
the input features.

``` r

# 1. Define targets first — the outcome age is needed to cut off the context
targets_cr <- data.frame(
  patient_id   = c(1L,    2L,    3L,    4L,    5L,    6L),
  target_event = c("CVD", "T2D", "T2D", "T2D", "T2D", "STATIN"),
  target_age   = c(58.0,  48.0,  63.5,  47.0,  51.0,  63.0)
)
survivehr_validate_targets(targets_cr)

# 2. Build context: patients 1–6, outcome codes removed, events after outcome age excluded
ft_events_cr <- events_pop[events_pop$patient_id %in% 1:6 &
                              !events_pop$event %in% c("CVD", "T2D"), ]
ft_events_cr <- merge(ft_events_cr, targets_cr[, c("patient_id", "target_age")],
                      by = "patient_id")
ft_events_cr <- ft_events_cr[ft_events_cr$age < ft_events_cr$target_age,
                              c("patient_id", "event", "age", "value")]
survivehr_validate_events(ft_events_cr)
# [OK] Events: 15 rows, 6 patients. ...

# time_scale is not set here — it is inherited automatically from the pretrained bundle
cfg_cr <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  dropout = 0, learning_rate = 3e-4, epochs = 10, batch_size = 4,
  surv_layer = "competing-risk"
)

ft_cr <- survivehr_finetune(
  events            = ft_events_cr,
  targets           = targets_cr,
  outcomes          = c("CVD", "T2D"),   # two competing outcomes
  risk_model        = "competing-risk",
  static_covariates = ft_static,
  config            = cfg_cr,
  pretrained_model  = pt_model           # vocabulary + weights inherited
)

cat("Fine-tune (CR) loss history:", unlist(ft_cr$history), "\n")
```

### 5b Single-risk fine-tuning

Use `risk_model = "single-risk"` when there is a single endpoint of
interest and no competing events need to be modelled explicitly. Here we
model **CVD only**: patients 1–3 are CVD cases; patients 4–6 are
right-censored controls.

``` r

# 1. Define targets first
targets_sr <- data.frame(
  patient_id   = c(1L,    2L,    3L,     4L,          5L,       6L),
  target_event = c("CVD", "CVD", "CVD",  "METFORMIN", "STATIN", "STATIN"),
  target_age   = c(58.0,  52.0,  65.5,   48.3,         54.5,     63.0)
)
survivehr_validate_targets(targets_sr)

# 2. Build context: CVD removed, events after outcome age excluded
# (T2D remains in context because it is not an outcome in this single-risk model)
ft_events_sr <- events_pop[events_pop$patient_id %in% 1:6 &
                              events_pop$event != "CVD", ]
ft_events_sr <- merge(ft_events_sr, targets_sr[, c("patient_id", "target_age")],
                      by = "patient_id")
ft_events_sr <- ft_events_sr[ft_events_sr$age < ft_events_sr$target_age,
                              c("patient_id", "event", "age", "value")]
survivehr_validate_events(ft_events_sr)
# [OK] Events: 24 rows, 6 patients. ...

# time_scale is not set here — it is inherited automatically from the pretrained bundle
cfg_sr <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  dropout = 0, learning_rate = 3e-4, epochs = 10, batch_size = 4,
  surv_layer = "competing-risk"
)

ft_sr <- survivehr_finetune(
  events            = ft_events_sr,
  targets           = targets_sr,
  outcomes          = "CVD",             # single outcome
  risk_model        = "single-risk",
  static_covariates = ft_static,
  config            = cfg_sr,
  pretrained_model  = pt_model
)

cat("Fine-tune (SR) loss history:", unlist(ft_sr$history), "\n")
```

| Setting | When to use |
|----|----|
| `"competing-risk"` | Multiple simultaneous endpoints (e.g. CVD and T2D compete to be the first event). |
| `"single-risk"` | One endpoint; controls are right-censored by any other event. |

------------------------------------------------------------------------

## 6 Save and load fine-tuned models

``` r

survivehr_save_model(ft_cr, "model_cr.pt")
survivehr_save_model(ft_sr, "model_sr.pt")

# Reload — weights, vocabulary, static column schema, time_scale and device are preserved
ft_cr2 <- survivehr_load_model("model_cr.pt")
ft_sr2 <- survivehr_load_model("model_sr.pt")
```

------------------------------------------------------------------------

## 7 Prediction

Prediction uses the **full 10-patient population** (including patients
7–10 who were held out from fine-tuning) to demonstrate population-level
risk scoring. Two preparation steps are required — the same logic as
fine-tuning:

1.  **Remove outcome event codes** from the context.
2.  **Remove events after each patient’s prediction age** — for patients
    with a known outcome (p1–p6) this is their `target_age`; for
    held-out patients (p7–p10) this is their last recorded event age.

``` r

# Prediction ages: target_age for p1-6 (from targets_cr / targets_sr),
# last recorded event age for p7-10.
pred_ages_cr <- data.frame(
  patient_id = c(1L,   2L,   3L,    4L,   5L,   6L,   7L,   8L,   9L,  10L),
  pred_age   = c(58.0, 48.0, 63.5, 47.0, 51.0, 63.0, 44.8, 62.0, 54.0, 48.5)
)
pred_ages_sr <- data.frame(
  patient_id = c(1L,   2L,   3L,    4L,   5L,   6L,   7L,   8L,   9L,  10L),
  pred_age   = c(58.0, 52.0, 65.5, 48.3, 54.5, 63.0, 44.8, 62.0, 54.0, 48.5)
)

# Competing-risk context: remove CVD + T2D, keep events before pred_age
pred_events_cr <- events_pop[!events_pop$event %in% c("CVD", "T2D"), ]
pred_events_cr <- merge(pred_events_cr, pred_ages_cr, by = "patient_id")
pred_events_cr <- pred_events_cr[pred_events_cr$age < pred_events_cr$pred_age,
                                  c("patient_id", "event", "age", "value")]
survivehr_validate_events(pred_events_cr)
# [OK] Events: 27 rows, 10 patients. ...

# Single-risk context: remove CVD only, keep events before pred_age
pred_events_sr <- events_pop[events_pop$event != "CVD", ]
pred_events_sr <- merge(pred_events_sr, pred_ages_sr, by = "patient_id")
pred_events_sr <- pred_events_sr[pred_events_sr$age < pred_events_sr$pred_age,
                                  c("patient_id", "event", "age", "value")]
survivehr_validate_events(pred_events_sr)
# [OK] Events: 38 rows, 10 patients. ...

# Competing-risk predictions — CVD and T2D risk head
preds_cr <- survivehr_predict(ft_cr2, pred_events_cr, static_pop)
head(preds_cr)
# Columns: patient_id, CVD_cdf_last, CVD_auc, T2D_cdf_last, T2D_auc

# Single-risk predictions — CVD risk head
preds_sr <- survivehr_predict(ft_sr2, pred_events_sr, static_pop)
head(preds_sr)
# Columns: patient_id, CVD_cdf_last, CVD_auc
```

### 7a Risk at specific time points

The `eval_times` argument returns the cumulative incidence at any set of
time points within the prediction window `(0, time_scale]`. With
`time_scale = 5.0` (years), valid `eval_times` are any positive values
up to and including 5.

``` r

# 1-year, 2-year, 3-year and 5-year (= full horizon) CVD risk
preds_cr_tp <- survivehr_predict(
  ft_cr2, pred_events_cr, static_pop,
  eval_times = c(1, 2, 3, 5)
)
head(preds_cr_tp)
# New columns alongside _cdf_last and _auc:
#   CVD_cdf_t1   — cumulative CVD risk at 1 year
#   CVD_cdf_t2   — cumulative CVD risk at 2 years
#   CVD_cdf_t3   — cumulative CVD risk at 3 years
#   CVD_cdf_t5   — cumulative CVD risk at 5 years (≈ CVD_cdf_last)
#   T2D_cdf_t1   — cumulative T2D risk at 1 year  (competing-risk model)
#   ... etc.
```

> **Boundary rule**: every value in `eval_times` must be in
> `(0, time_scale]`. Requesting a time point beyond the prediction
> window the model was trained on (e.g. `eval_times = 10` with
> `time_scale = 5`) raises an error because the ODE grid does not extend
> past the trained horizon. The internal survival ODE evaluates on a
> fixed normalised grid of 1 000 equally-spaced steps from 0 to 1
> (mapped back to `[0, time_scale]` in raw units); `eval_times` values
> are snapped to the nearest grid point.

### Understanding the output columns

**Fine-tuned model outputs**
([`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)):

| Column | Description |
|----|----|
| `{outcome}_cdf_last` | Cumulative incidence at the **end** of the prediction window. With `time_scale = 5.0` this is the estimated 5-year risk from the patient’s last recorded event. |
| `{outcome}_auc` | **Area under the CDF** integrated across the full window. A single scalar risk score; values closer to 1 indicate higher overall risk. |
| `{outcome}_cdf_t{X}` | *(Only when `eval_times` is supplied.)* Cumulative incidence at time `X` in the same units as `age`, e.g. `CVD_cdf_t1` for 1-year CVD risk. |

**Pre-train model outputs**
([`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)):

| Column | Description |
|----|----|
| `step` | Generation step index (1 = next event, 2 = event after that, …). |
| `generated_event` | Decoded event name; `"<UNK>"` for out-of-vocabulary events. |
| `generated_age` | Predicted event age in raw units (de-normalised by `time_scale`). |
| `generated_value` | Predicted numeric value; `NaN` for non-measurement events. |

**Value prediction outputs**
([`survivehr_predict_value()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict_value.md)):

| Column | Description |
|----|----|
| `predicted_value_mean` | Predicted mean of the Gaussian distribution for the event’s numeric measurement. |
| `predicted_value_sd` | Predicted standard deviation. |

> **ODE grid**: the survival head evaluates on 1 000 equally-spaced time
> steps from 0 to 1 in normalised time (0 to `time_scale` in raw units).
> `eval_times` values are snapped to the nearest grid point, giving a
> resolution of `time_scale / 999` (≈ 0.005 years with
> `time_scale = 5.0`).

### New patients at inference

Pass new patients with exactly the same static columns used at training
time. The model stores the exact encoded column list (including one-hot
expanded categories) and raises a clear error rather than silently
producing wrong predictions if there is a mismatch.

``` r

new_events <- data.frame(
  patient_id = c(99, 99, 99, 99),
  event      = c("HYPERTENSION", "BP_CHECK", "METFORMIN", "ASPIRIN"),  # ASPIRIN → <UNK>
  age        = c(55.0,  55.3,       55.4,         56.1),
  value      = c(NA,    158,        NA,           NA)
)

# Must include the same columns as static_pop: sex, ethnicity, imd, year_of_birth
new_static <- data.frame(
  patient_id    = 99,
  sex           = "M",
  ethnicity     = "White",
  imd           = 2L,
  year_of_birth = 1963L
)

pred_new <- survivehr_predict(ft_cr2, new_events, new_static)
print(pred_new)
```

See the [Model
architecture](https://pm-cardoso.github.io/RSurvivEHR/articles/model-architecture.md)
article for the full configuration reference and the official Gadd et
al. (2026) hyperparameter table.

------------------------------------------------------------------------

## References

Gadd, C., Gokhale, K., Acharya, A. et al. (2026). *SurvivEHR: a
competing risks, time-to-event foundation model for multiple long-term
conditions from primary care electronic health records*. npj Digital
Medicine.
[doi:10.1038/s41746-026-02709-z](https://doi.org/10.1038/s41746-026-02709-z)
