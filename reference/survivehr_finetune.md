# Fine-tune SurvivEHR from R data frames

Fine-tune SurvivEHR from R data frames

## Usage

``` r
survivehr_finetune(
  events,
  targets,
  outcomes,
  risk_model = c("competing-risk", "single-risk"),
  static_covariates = NULL,
  config = survivehr_config(),
  pretrained_model = NULL,
  event_vocab = NULL
)
```

## Arguments

- events:

  data.frame with columns patient_id, event, age, optional value.

- targets:

  data.frame with columns patient_id, target_event, target_age, optional
  target_value.

- outcomes:

  character vector of outcomes for the fine-tuned head.

- risk_model:

  "competing-risk" or "single-risk".

- static_covariates:

  optional data.frame with patient_id + numeric columns.

- config:

  list from
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md).

- pretrained_model:

  optional model handle from
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md).

- event_vocab:

  optional named integer map to keep fixed tokenization.

## Value

A named list (fine-tuned model bundle). Pass to
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
or
[`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md).

## Examples

``` r
if (FALSE) { # \dontrun{
# Using events/static/cfg/pt from survivehr_pretrain() example above
targets <- data.frame(
  patient_id   = 1L,
  target_event = "CVD",
  target_age   = 54.0
)
ft <- survivehr_finetune(
  events, targets,
  outcomes     = "CVD",
  risk_model   = "single-risk",
  static_covariates = static,
  config       = cfg,
  pretrained_model  = pt
)
} # }
```
