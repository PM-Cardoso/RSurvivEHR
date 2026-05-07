# Build an RSurvivEHR configuration list

Returns a named list of hyperparameters consumed by
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
and
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md).
All parameters have sensible defaults suitable for small datasets; see
the *Model architecture* vignette for recommended values for larger
cohorts and the official Gadd et al. (2025) hyperparameter table.

## Usage

``` r
survivehr_config(
  block_size = 128,
  n_layer = 4,
  n_head = 4,
  n_embd = 256,
  dropout = 0,
  learning_rate = 3e-04,
  epochs = 1,
  batch_size = 16,
  surv_layer = "competing-risk",
  surv_weight = 1,
  value_weight = 0,
  device = "auto",
  include_unk = TRUE,
  include_cls_sep = FALSE,
  time_scale = 1
)
```

## Arguments

- block_size:

  Sequence length after padding/truncation.

- n_layer:

  Number of transformer blocks.

- n_head:

  Number of attention heads.

- n_embd:

  Hidden embedding size.

- dropout:

  Dropout probability.

- learning_rate:

  Learning rate.

- epochs:

  Number of epochs.

- batch_size:

  Batch size.

- surv_layer:

  Pretrain survival head: "competing-risk" or "single-risk".

- surv_weight:

  Survival loss weight.

- value_weight:

  Value regression loss weight.

- device:

  "auto", "cpu", or "cuda".

- include_unk:

  Whether to reserve and use `<UNK>` for unseen events.

- include_cls_sep:

  Whether to add `<CLS>` and `<SEP>` around each sequence.

- time_scale:

  Controls both the age normalisation and the length of the prediction
  window. Every raw age is divided by this value before entering the
  model; the survival ODE evaluates over a normalised `[0, 1]` grid that
  maps back to `[0, time_scale]` in your original age units.

  - Use `5.0` for a 5-year prediction window with ages in years
    (recommended).

  - Use `1.0` for a 1-year window with ages in years.

  - Use `365.25` for a 1-year window with ages in days.

  - Use `1826.25` for a 5-year window with ages in days.

  Stored automatically in every model bundle; no need to supply at
  prediction time. Must be the same across pretrain, fine-tune, and
  prediction.

## Value

Named list used by training functions.

## Examples

``` r
# Minimal config for a 5-year prediction window (ages in years)
cfg <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 1, batch_size = 4, time_scale = 5.0
)
cfg$surv_layer   # "competing-risk"
#> [1] "competing-risk"
cfg$time_scale   # 5.0
#> [1] 5
```
