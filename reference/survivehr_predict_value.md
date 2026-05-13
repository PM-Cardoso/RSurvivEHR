# Predict the numeric value of a named event (pretrain or fine-tuned models)

Queries the backbone's Gaussian value regression head to estimate the
numeric measurement (e.g. blood pressure, HbA1c) that would be recorded
alongside a specific clinical event, given each patient's history in
`events`.

## Usage

``` r
survivehr_predict_value(
  model_bundle,
  events,
  outcome_event,
  static_covariates = NULL
)
```

## Arguments

- model_bundle:

  A model bundle returned by
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
  or
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).

- events:

  A `data.frame` with columns `patient_id`, `event`, `age`, optional
  `value`. The `outcome_event` code should **not** appear in this frame
  (same leakage-free filtering as for fine-tuning and prediction).

- outcome_event:

  Character scalar. The event code whose value should be predicted (e.g.
  `"BP_CHECK"` for blood pressure). Must exist in the model vocabulary
  built at pre-training time.

- static_covariates:

  An optional `data.frame` with `patient_id` and the same covariate
  columns used at training time. Pass `NULL` to omit.

## Value

A `data.frame` with one row per patient and columns:

- `patient_id`:

  Patient identifier.

- `outcome_event`:

  The event code that was queried.

- `predicted_value_mean`:

  Predicted mean of the Gaussian distribution for the event's numeric
  value. `NaN` if the event never appeared with a value at pre-training
  time.

- `predicted_value_sd`:

  Predicted standard deviation. `NaN` for the same reason as above.

## Details

The value head is trained during pre-training on events that carried
non-`NA` `value` entries. For events that never appeared with a value
(e.g. `"CVD"`, which is a discrete diagnosis), the function returns
`NaN` for both the mean and standard deviation. The head is preserved in
fine-tuned bundles because fine-tuning only replaces the outcome
survival head, not the backbone.

## Examples

``` r
if (FALSE) { # \dontrun{
# Using pt_model and events_pop / static_pop from survivehr_pretrain() example.
# Predict the expected blood-pressure reading at the next BP_CHECK.
# (Remove BP_CHECK from context first — same leakage-free logic.)
ctx <- events_pop[events_pop$event != "BP_CHECK", ]
bp_preds <- survivehr_predict_value(pt_model, ctx, "BP_CHECK", static_pop)
print(bp_preds)
# Columns: patient_id, outcome_event, predicted_value_mean, predicted_value_sd
} # }
```
