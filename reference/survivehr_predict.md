# Predict cumulative incidence with a fine-tuned RSurvivEHR model

Runs forward inference on new event sequences using a fine-tuned model
bundle. The prediction window length and age normalisation divisor
(`time_scale`) are read automatically from the bundle — no need to
supply them at inference time.

## Usage

``` r
survivehr_predict(
  model_bundle,
  events,
  static_covariates = NULL,
  max_new_tokens = 1L,
  eval_times = NULL
)
```

## Arguments

- model_bundle:

  A model bundle returned by
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  or
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).

- events:

  data.frame with columns `patient_id`, `event`, `age`, optional
  `value`. Should **not** contain the outcome event (leakage-free).

- static_covariates:

  optional data.frame with `patient_id` and the same covariate columns
  used at training time.

- max_new_tokens:

  number of autoregressive steps (pretrain models only; ignored for
  fine-tuned models).

- eval_times:

  An optional numeric vector of time points (in the same units as `age`)
  at which to read the cumulative-incidence CDF for fine-tuned models.
  Each value must be in `(0, outcome_horizon]` — the ODE prediction
  window stored in the fine-tuned bundle. For example, with
  `outcome_horizon = 5.0` (years) use `eval_times = c(1, 2, 3, 5)` to
  obtain 1-, 2-, 3- and 5-year risks. When `NULL` (default) only
  `_cdf_last` (risk at the full horizon) and `_auc` (average risk) are
  returned, preserving backward compatibility. Ignored for pretrain
  models.

## Value

For a **fine-tuned** model, a `data.frame` with columns:

- `patient_id`:

  Patient identifier.

- `{outcome}_cdf_last`:

  Cumulative incidence at the **end** of the prediction window
  (`t = outcome_horizon`).

- `{outcome}_auc`:

  Area under the CDF integrated from 0 to `time_scale`. Interpretable as
  average risk over the window.

- `{outcome}_cdf_t{X}`:

  *(Only when `eval_times` is supplied.)* Cumulative incidence at time
  `X` (same units as `age`). One column per requested time point, e.g.
  `CVD_cdf_t1`, `CVD_cdf_t2.5`.

For a **pretrain** model, a `data.frame` with one row per generated step
per patient:

- `patient_id`:

  Patient identifier.

- `step`:

  Generation step (1 = next event, 2 = event after that, …).

- `generated_token`:

  Vocabulary token ID of the generated event.

- `generated_event`:

  Decoded event name (e.g. `"HYPERTENSION"`).

- `generated_age`:

  Predicted age of the generated event in the same units as `age`
  (de-normalised by `time_scale`).

- `generated_value`:

  Predicted numeric value for that event (e.g. a lab result) in the
  original input units (automatically de-standardised per event); `NaN`
  for non-measurement events.

## Details

The `events` frame should have the outcome event codes removed (same as
the context filtering applied at fine-tune time) to ensure leakage-free
predictions.

## Examples

``` r
if (FALSE) { # \dontrun{
# ---- Prediction using ft / events_pop / static_pop from examples above -----
# Remove the outcome event from prediction context (leakage-free)
pred_events <- events_pop[events_pop$event != "CVD", ]

preds <- survivehr_predict(ft, pred_events, static_pop)
print(preds)
# Columns: patient_id, CVD_cdf_last, CVD_auc
#
# CVD_cdf_last : probability of CVD within outcome_horizon (5 years when
#                outcome_horizon = 5.0); stored in the bundle automatically
# CVD_auc      : average cumulative CVD risk over that window;
#                higher = greater overall risk
} # }
```
