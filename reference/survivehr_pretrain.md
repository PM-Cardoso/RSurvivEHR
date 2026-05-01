# Train a pretrain SurvivEHR model from R data frames

Train a pretrain SurvivEHR model from R data frames

## Usage

``` r
survivehr_pretrain(
  events,
  static_covariates = NULL,
  config = survivehr_config(),
  event_vocab = NULL
)
```

## Arguments

- events:

  data.frame with columns patient_id, event, age, optional value.

- static_covariates:

  optional data.frame with patient_id + numeric columns.

- config:

  list from
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md).

- event_vocab:

  optional named integer map to keep fixed tokenization.

## Value

A named list (model bundle) with elements `model`, `event_vocab`,
`inv_vocab`, `config`, `time_scale`, and `token_policy`.

## Examples

``` r
if (FALSE) { # \dontrun{
events <- data.frame(
  patient_id = c(1L,1L,1L, 2L,2L,2L),
  event  = c("HYPERTENSION","STATIN","T2D", "T2D","METFORMIN","HYPERTENSION"),
  age    = c(50, 50.5, 52,  45, 45.3, 47.5)
)
static <- data.frame(patient_id=c(1L,2L), sex=c("M","F"), imd=c(3L,1L))
cfg    <- survivehr_config(block_size=32, n_layer=2, n_head=2, n_embd=64, epochs=1)
pt     <- survivehr_pretrain(events, static, cfg)
pt$event_vocab  # <PAD>=0, <UNK>=1, HYPERTENSION=2, ...
} # }
```
