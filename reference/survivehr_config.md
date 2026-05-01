# Build default SurvivEHR configuration

Build default SurvivEHR configuration

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

  Divisor applied to every raw age before it enters the model. Use `1.0`
  (default) when ages are in years. Use `1825.0` when ages are in days
  (`DAYS_SINCE_BIRTH`, matching FastEHR default). Must be consistent
  between pretrain, fine-tune, and prediction.

## Value

Named list used by training functions.

## Examples

``` r
# Minimal config for a quick CPU run
cfg <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 1, batch_size = 4
)
cfg$surv_layer   # "competing-risk"
#> [1] "competing-risk"
cfg$time_scale   # 1.0
#> [1] 1
```
