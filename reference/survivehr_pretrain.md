# Pre-train the RSurvivEHR backbone transformer

Builds the event vocabulary from the supplied event history, then
pre-trains the transformer backbone with a competing-risk or single-risk
survival head that predicts **when** the next clinical event will occur
(over all vocabulary tokens). The resulting model bundle can be passed
directly to
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
or saved with
[`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
for later reuse.

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

  A `data.frame` with columns `patient_id`, `event`, `age` (and
  optionally `value`) — lowercase is the preferred canonical form;
  uppercase aliases `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH` are
  accepted for backward compatibility. Validated with
  [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md)
  before being passed to Python.

- static_covariates:

  An optional `data.frame` with `patient_id` and covariate columns.
  Categorical columns are one-hot encoded automatically; numeric columns
  pass through unchanged. Pass `NULL` (default) to train without static
  features.

- config:

  A named list from
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  specifying architecture and training hyperparameters.

- event_vocab:

  An optional named integer vector fixing the token mapping. Useful when
  pre-training multiple models that must share the same vocabulary.
  `NULL` (default) builds the vocabulary from the supplied events,
  ordered by descending frequency.

## Value

A named list (model bundle) with elements:

- `model`:

  The trained PyTorch model object.

- `event_vocab`:

  Named integer vector mapping event codes to token IDs
  (frequency-descending order, most common = smallest ID).

- `inv_vocab`:

  Reverse mapping from token IDs to event codes.

- `config`:

  The configuration used for training.

- `time_scale`:

  The backbone age normalisation divisor stored in the bundle. Used to
  normalise context ages before they enter the transformer; inherited
  automatically at fine-tune time.

- `token_policy`:

  Token policy flags (`include_unk`, `include_cls_sep`).

- `history`:

  List of per-epoch training losses.

- `training_duration_secs`:

  Wall-clock seconds elapsed during training (a single `numeric`
  scalar). Use this to report and compare training times, e.g.
  `cat("Pretrain took", round(pt$training_duration_secs, 1), "s\\n")`.

- `device`:

  String identifying the compute device used (e.g. `"cpu"` or
  `"cuda:0"`).

## Details

Pre-training does **not** require a `targets` frame — the next-event age
in each patient's raw sequence serves as the supervision signal.

## Examples

``` r
if (FALSE) { # \dontrun{
# ---- Pre-training on the 10-patient population from the Getting Started
#      vignette.  Patients 1-3 have CVD in their history so the vocabulary
#      includes CVD — required for CVD fine-tuning later.
events_pop <- data.frame(
  patient_id = c(rep(1,4), rep(2,6), rep(3,6), rep(4,4), rep(5,6),
                 rep(6,4), rep(7,4), rep(8,4), rep(9,6), rep(10,4)),
  event = c(
    "HYPERTENSION","STATIN","BP_CHECK","CVD",
    "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HYPERTENSION","CVD",
    "HYPERTENSION","BP_CHECK","STATIN","T2D","BP_CHECK","CVD",
    "HYPERTENSION","STATIN","T2D","METFORMIN",
    "HYPERTENSION","BP_CHECK","T2D","HBA1C","METFORMIN","STATIN",
    "HYPERTENSION","AMLODIPINE","BP_CHECK","STATIN",
    "STATIN","T2D","HBA1C","METFORMIN",
    "HYPERTENSION","BP_CHECK","STATIN","T2D",
    "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HBA1C","STATIN",
    "STATIN","BP_CHECK","HYPERTENSION","T2D"
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
    NA,  NA,  148, NA,
    NA,  145, NA,  NA,  NA,  NA,
    NA,  158, NA,  NA,  162, NA,
    NA,  NA,  NA,  NA,
    NA,  152, NA,  68,  NA,  NA,
    NA,  NA,  155, NA,
    NA,  NA,  74,  NA,
    NA,  145, NA,  NA,
    NA,  138, NA,  NA,  71,  NA,
    NA,  138, NA,  NA
  )
)
static_pop <- data.frame(
  patient_id    = 1:10,
  sex           = c("M","F","M","F","M","F","M","F","M","F"),
  ethnicity     = c("White","Asian","White","Black","White",
                    "Asian","White","White","Black","White"),
  imd           = c(3L, 1L, 5L, 2L, 4L, 3L, 1L, 5L, 2L, 4L),
  year_of_birth = c(1960L,1970L,1952L,1975L,1963L,1958L,1978L,1960L,1968L,1975L)
)
# Year-by-year backbone normalisation (ages in years → model sees plain year values)
cfg <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 10, batch_size = 4, time_scale = 1.0
)
pt <- survivehr_pretrain(events_pop, static_pop, cfg)
# Vocabulary is frequency-ordered: most-common events get the smallest IDs
# HYPERTENSION=10, BP_CHECK=9, STATIN=9, T2D=8, METFORMIN=5, CVD=3, HBA1C=3, AMLODIPINE=1
pt$event_vocab
} # }
```
