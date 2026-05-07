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
  max_new_tokens = 1L
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

## Value

For a **fine-tuned** model, a `data.frame` with columns:

- `patient_id`:

  Patient identifier.

- `{outcome}_cdf_last`:

  Cumulative incidence of `outcome` at the end of the prediction window.
  The window spans `[0, time_scale]` in the same units as `age` (e.g.
  0–5 years when `time_scale = 5.0`). `time_scale` is stored in the
  model bundle and used automatically — no need to supply it at
  prediction time.

- `{outcome}_auc`:

  Area under the CDF curve integrated from 0 to `time_scale`.
  Interpretable as the average cumulative risk over the prediction
  window. Values closer to 1 indicate higher overall risk; values closer
  to 0 indicate lower risk.

For a **pretrain** model, a `data.frame` with columns `patient_id`,
`generated_token`, `generated_event`, `generated_age`,
`generated_value`.

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
# CVD_cdf_last : probability of CVD within the next 5 years (time_scale = 5.0,
#                stored in the model bundle — no need to set it at predict time)
# CVD_auc      : average cumulative CVD risk over that 5-year window;
#                higher = greater overall risk
} # }
```
