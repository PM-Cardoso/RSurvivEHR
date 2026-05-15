# Build an RSurvivEHR configuration list

Returns a named list of hyperparameters consumed by
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
and
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md).
All parameters have sensible defaults suitable for small datasets; see
the *Model architecture* vignette for recommended values for larger
cohorts and the official Gadd et al. (2026) hyperparameter table.

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
  time_scale = 1,
  outcome_horizon = NULL
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

  Age normalisation divisor for the backbone transformer. Every raw age
  is divided by this value before entering the model:
  `age_norm = raw_age / time_scale`. Must be consistent between
  pre-training and fine-tuning (inherited automatically from the
  pretrained bundle). Does **not** control the prediction window — use
  `outcome_horizon` for that.

  - `1.0` (default) — ages in years, backbone sees plain year values.

  - `365.25` — ages in days, backbone sees normalised fractions.

- outcome_horizon:

  Length of the ODE prediction window in the same raw age units as the
  `age` column. The survival ODE integrates over a normalised `[0, 1]`
  grid that maps back to `[0, outcome_horizon]` in raw age units. Can
  differ freely from `time_scale` — this is what makes it possible to
  use year-by-year age normalisation (`time_scale = 1.0`) while
  producing a 5-year risk (`outcome_horizon = 5.0`).

  - `NULL` (default) — inherits `time_scale` (backward compatible).

  - `5.0` — 5-year prediction window when ages are in years.

  - `1.0` — 1-year prediction window when ages are in years.

  - `1826.25` — 5-year window when ages are in days.

  Fine-tune only; ignored at pre-training. Stored in the fine-tune
  bundle and used automatically at prediction time.

## Value

Named list used by training functions.

## Examples

``` r
# Year-by-year backbone normalisation with a 5-year prediction window
cfg <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 1, batch_size = 4,
  time_scale = 1.0,        # backbone: ages enter model as plain year values
  outcome_horizon = 5.0    # ODE: cdf_last = 5-year risk
)
cfg$outcome_horizon  # 5
#> [1] 5

# Both default to 1.0 if outcome_horizon is omitted (backward compatible)
cfg2 <- survivehr_config(time_scale = 1.0)
cfg2$outcome_horizon  # NULL -> inherits time_scale in the backend
#> NULL
```
