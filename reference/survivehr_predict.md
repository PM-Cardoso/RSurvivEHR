# Predict next events with SurvivEHR

Predict next events with SurvivEHR

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

  model object returned by pretrain/fine-tune.

- events:

  data.frame with columns patient_id, event, age, optional value.

- static_covariates:

  optional data.frame with patient_id + numeric columns.

- max_new_tokens:

  number of autoregressive steps.

## Value

`data.frame` with columns `patient_id`, `event`, `age`, `value`.

## Examples

``` r
if (FALSE) { # \dontrun{
preds <- survivehr_predict(ft, events, static)
preds
} # }
```
