# Fine-tune the RSurvivEHR backbone on labelled outcomes

Attaches a fresh outcome-level survival head on top of the pre-trained
backbone and fine-tunes the combined model on a labelled cohort. Two
head types are supported:

- **`"competing-risk"`** — models two or more outcomes that compete (the
  first to occur prevents the others from being observed). Supply all
  outcome codes in `outcomes`.

- **`"single-risk"`** — models a single endpoint; patients without the
  outcome are treated as right-censored.

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

  A `data.frame` of context events (outcome codes and post-outcome rows
  removed) with columns `patient_id`, `event`, `age`, and optionally
  `value`.

- targets:

  A `data.frame` with columns `patient_id`, `target_event`, and
  `target_age` labelling the observed outcome (cases) or last
  non-outcome event (censored patients). Build with
  [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md).

- outcomes:

  Character vector of outcome event codes that the fine-tuned head
  predicts. Must match the codes used in `targets`.

- risk_model:

  `"competing-risk"` (default) or `"single-risk"`. Controls the
  architecture of the outcome-level head — independent of the
  `surv_layer` used during pre-training.

- static_covariates:

  An optional `data.frame` with the **same columns** as those used at
  pre-training time. Pass `NULL` to omit.

- config:

  A named list from
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md).
  `time_scale` is inherited from the pre-trained bundle when
  `pretrained_model` is supplied — no need to set it here.

- pretrained_model:

  Model bundle returned by
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
  or
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).
  When supplied, the vocabulary, weights, and `time_scale` are inherited
  from the bundle.

- event_vocab:

  An optional named integer vector overriding the vocabulary. Rarely
  needed; prefer supplying `pretrained_model`.

## Value

A named list (fine-tuned model bundle) with the same structure as the
pre-trained bundle plus fine-tune-specific fields. Pass to
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
or
[`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md).

## Details

To avoid data leakage, the `events` frame passed here must:

1.  Have all outcome event codes **removed** (they are supplied only via
    `targets`).

2.  Contain only events that occurred **before** each patient's
    `target_age` (events after the outcome would not be available at
    prediction time).

## Examples

``` r
if (FALSE) { # \dontrun{
# Uses events_pop / static_pop / cfg / pt from survivehr_pretrain() example.
ft_static <- static_pop[static_pop$patient_id %in% 1:6, ]

# ---- Competing-risk: CVD vs T2D (patients 1-6) ---------------------------
targets_cr <- data.frame(
  patient_id   = c(1L,    2L,    3L,    4L,    5L,    6L),
  target_event = c("CVD", "T2D", "T2D", "T2D", "T2D", "STATIN"),
  target_age   = c(58.0,  48.0,  63.5,  47.0,  51.0,  63.0)
)
survivehr_validate_targets(targets_cr)

# Remove both outcomes; keep only events before target_age
ft_events_cr <- events_pop[events_pop$patient_id %in% 1:6 &
                              !events_pop$event %in% c("CVD","T2D"), ]
ft_events_cr <- merge(ft_events_cr,
                      targets_cr[, c("patient_id","target_age")],
                      by = "patient_id")
ft_events_cr <- ft_events_cr[ft_events_cr$age < ft_events_cr$target_age,
                              c("patient_id","event","age","value")]

cfg_cr <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 10, batch_size = 4, surv_layer = "competing-risk"
  # time_scale inherited automatically from the pretrained bundle
)
ft_cr <- survivehr_finetune(
  events = ft_events_cr, targets = targets_cr,
  outcomes = c("CVD","T2D"), risk_model = "competing-risk",
  static_covariates = ft_static, config = cfg_cr, pretrained_model = pt
)
cat("CR loss:", unlist(ft_cr$history), "\n")

# ---- Single-risk: CVD only (patients 1-6) --------------------------------
targets_sr <- data.frame(
  patient_id   = c(1L,    2L,    3L,    4L,          5L,       6L),
  target_event = c("CVD", "CVD", "CVD", "METFORMIN", "STATIN", "STATIN"),
  target_age   = c(58.0,  52.0,  65.5,  48.3,         54.5,     63.0)
)
survivehr_validate_targets(targets_sr)

# Remove CVD only; keep only events before target_age
ft_events_sr <- events_pop[events_pop$patient_id %in% 1:6 &
                              events_pop$event != "CVD", ]
ft_events_sr <- merge(ft_events_sr,
                      targets_sr[, c("patient_id","target_age")],
                      by = "patient_id")
ft_events_sr <- ft_events_sr[ft_events_sr$age < ft_events_sr$target_age,
                              c("patient_id","event","age","value")]

cfg_sr <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 10, batch_size = 4, surv_layer = "competing-risk"
)
ft_sr <- survivehr_finetune(
  events = ft_events_sr, targets = targets_sr,
  outcomes = "CVD", risk_model = "single-risk",
  static_covariates = ft_static, config = cfg_sr, pretrained_model = pt
)
cat("SR loss:", unlist(ft_sr$history), "\n")
} # }
```
