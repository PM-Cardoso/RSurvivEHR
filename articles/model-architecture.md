# Model architecture and parameter reference

This vignette explains every configurable parameter in **RSurvivEHR**,
why it exists, what value the original SurvivEHR authors used in their
UK primary-care study (7.6 billion events, 23 million patients), and
guidelines for choosing a value for your own data.

Upstream defaults are taken directly from the published hyperparameter
table in Gadd et al. (2025) and from the upstream configuration files
[`config_CompetingRisk11M.yaml`](https://github.com/cwlgadd/SurvivEHR/blob/main/examples/modelling/SurvivEHR/confs/config_CompetingRisk11M.yaml)
and
[`config_SingleRisk.yaml`](https://github.com/cwlgadd/SurvivEHR/blob/main/examples/modelling/SurvivEHR/confs/config_SingleRisk.yaml)
in the `cwlgadd/SurvivEHR` repository.

------------------------------------------------------------------------

## Overview

SurvivEHR is a GPT-style transformer pre-trained on sequences of
clinical events (diagnoses, prescriptions, measurements, …) via *causal
language modelling*. After pre-training the model “understands” the
temporal structure of medical histories. A neural-ODE survival head is
then fine-tuned on labelled outcome data (target event and age at event)
to produce calibrated time-to-event curves.

    EHR sequences --> Tokeniser --> Token + positional embedding
                                            |
                                n_layer x NeoGPT block  (MHA + MLP + LayerNorm)
                                            |
                                Last attended hidden state
                                            |
                    +-----------------------+----------------------+
                    |  Competing-risk ODE head (one ODE per event)|
                    |  or                                         |
                    |  Single-risk ODE head (one ODE)             |
                    +---------------------------------------------+
                                            |
                                CDF curves per time horizon

------------------------------------------------------------------------

## Upstream hyperparameters (Gadd et al., 2025)

The following table reproduces the hyperparameters used in the original
CPRD study. These are the values to target when you have a large dataset
and GPU resources.

| Component | Hyperparameter | Pre-training | Fine-tuning |
|----|----|----|----|
| **Architecture** | Layers (`n_layer`) | 6 | 6 |
|  | Attention heads (`n_head`) | 6 | 6 |
|  | Hidden size (`n_embd`) | 384 | 384 |
|  | Max sequence length (`block_size`) | 256 | 512 |
|  | Context window | Repeated measurements | All last unique |
|  | Global diagnoses | False | Append to context |
| **Optimisation** | Batch size | 64 | 512 |
|  | Epochs | 10 | 20 |
|  | Early stopping | False | True |
|  | Optimiser | AdamW | AdamW |
|  | Backbone learning rate | 3 x 10^-4 | 5 x 10^-5 |
|  | Head learning rate | 3 x 10^-4 | 5 x 10^-4 |
|  | Scheduler | Linear warmup + cosine annealing | Reduce on plateau |
|  | Warmup steps | 10,000 | 0 |
|  | Warm restarts | True | False |

> **Note on context window (fine-tuning):** during fine-tuning the block
> size is extended to 512 and each patient’s context is formed from the
> *last unique* occurrence of each event rather than allowing repeated
> measurements. Global diagnoses (conditions recorded before the context
> window) are *appended* to the context rather than being dropped.

------------------------------------------------------------------------

## `survivehr_config()` – full parameter reference

``` r

library(RSurvivEHR)

cfg <- survivehr_config(
  # -- Transformer architecture -------------------------------------------
  block_size    = 128,       # context window (tokens); upstream pre-train: 256, fine-tune: 512
  n_layer       = 4,         # transformer blocks;      upstream: 6
  n_head        = 4,         # attention heads;         upstream: 6
  n_embd        = 256,       # embedding dimension;     upstream: 384

  # -- Regularisation ------------------------------------------------------
  dropout       = 0.0,       # applied before attention and after MLP

  # -- Optimisation --------------------------------------------------------
  learning_rate = 3e-4,      # AdamW learning rate; upstream pre-train backbone: 3e-4
  epochs        = 1,         # upstream pre-train: 10;  fine-tune: 20
  batch_size    = 16,        # upstream pre-train: 64;  fine-tune: 512

  # -- Survival head -------------------------------------------------------
  surv_layer    = "competing-risk",  # or "single-risk"
  surv_weight   = 1.0,
  value_weight  = 0.0,

  # -- Tokenisation --------------------------------------------------------
  include_unk     = TRUE,
  include_cls_sep = FALSE,

  # -- Age / time handling -------------------------------------------------
  time_scale    = 1.0,       # divide all raw ages by this number

  # -- Hardware ------------------------------------------------------------
  device        = "auto"
)
```

------------------------------------------------------------------------

## Parameter details

### `block_size` – context window

| RSurvivEHR default | Upstream pre-training | Upstream fine-tuning | Notes |
|----|----|----|----|
| `128` | `256` | `512` | Fine-tuning uses a longer window to include more history. |

The **context window** is the maximum number of tokens (clinical events)
seen by the model at once. Sequences longer than `block_size` are
*right-truncated* – only the most recent events are retained. Sequences
shorter than `block_size` are *left-padded* with `<PAD>` tokens (index
0).

Upstream pre-training used `block_size = 256` with *repeated
measurements* (e.g. two BMI records both retained). Fine-tuning extended
this to `512` and switched to *last-unique* context (keeping only the
most recent record for each event type) plus *global diagnoses* appended
at the start.

> **Tip:** set `block_size` to the 95th percentile of sequence lengths
> in your dataset.

------------------------------------------------------------------------

### `n_layer` – number of transformer blocks

| RSurvivEHR default | Upstream (both stages) |
|--------------------|------------------------|
| `4`                | `6`                    |

Each block consists of multi-head self-attention (MHA), a two-layer MLP,
LayerNorm (Pre-LN), and residual connections. More layers gives greater
representational capacity but requires more data and compute.

------------------------------------------------------------------------

### `n_head` – number of attention heads

| RSurvivEHR default | Upstream (both stages) | Notes |
|----|----|----|
| `4` | `6` | Must divide `n_embd` evenly (384 / 6 = 64 per head). |

------------------------------------------------------------------------

### `n_embd` – embedding / hidden dimension

| RSurvivEHR default | Upstream (both stages) | Notes |
|----|----|----|
| `256` | `384` | Total params ~= 12 x n_layer x n_embd^2. |

Controls the width of every layer. The MLP hidden dimension inside each
block is `4 x n_embd`. Common sizes: `64` (debug), `128` (tiny), `256`
(small), `384` (medium, upstream), `512`-`768` (large).

------------------------------------------------------------------------

### `dropout`

| RSurvivEHR default | Upstream |
|--------------------|----------|
| `0.0`              | `0.0`    |

Not needed when training on millions of patients. Set `0.1`-`0.3` for
small datasets (\< 10 000 patients).

------------------------------------------------------------------------

### `learning_rate`

| Stage        | Upstream backbone LR | Upstream head LR |
|--------------|----------------------|------------------|
| Pre-training | 3 x 10^-4            | 3 x 10^-4        |
| Fine-tuning  | 5 x 10^-5            | 5 x 10^-4        |

The optimiser is **AdamW**. During fine-tuning the backbone is updated
with a much smaller rate (5e-5) to preserve pre-trained representations,
while the new survival head trains faster (5e-4).

Pass `learning_rate` to
[`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
to set the head learning rate.

------------------------------------------------------------------------

### `epochs`

| RSurvivEHR default | Upstream pre-training | Upstream fine-tuning |
|--------------------|-----------------------|----------------------|
| `1`                | `10`                  | `20`                 |

Scale up proportionally for smaller datasets.

------------------------------------------------------------------------

### `batch_size`

| RSurvivEHR default | Upstream pre-training | Upstream fine-tuning |
|--------------------|-----------------------|----------------------|
| `16`               | `64`                  | `512`                |

The upstream fine-tuning batch of 512 requires gradient accumulation on
multi-GPU hardware. On a single laptop GPU (8 GB), `batch_size = 16-32`
is safe with `block_size = 128`.

------------------------------------------------------------------------

### Optimisation scheduler

| Stage        | Scheduler                        | Warmup steps | Warm restarts |
|--------------|----------------------------------|--------------|---------------|
| Pre-training | Linear warmup + cosine annealing | 10,000       | True          |
| Fine-tuning  | ReduceLROnPlateau                | 0            | False         |

Early stopping is **disabled** during pre-training and **enabled**
(patience = 30 validation checks) during fine-tuning.

------------------------------------------------------------------------

### `surv_layer` – survival head type

| Value | Description |
|----|----|
| `"competing-risk"` (default) | Neural-ODE head with one ODE per outcome event. Models cause-specific cumulative incidence functions (CIF). |
| `"single-risk"` | Neural-ODE head with a single ODE. Used when only one endpoint is of interest. |

The upstream config uses the abbreviated codes `"cr"` and `"sr"` – both
are also accepted by RSurvivEHR.

------------------------------------------------------------------------

### `surv_weight` and `value_weight`

| Loss term | Weight | Description |
|----|----|----|
| Causal LM (next-event prediction) | 1 (fixed) | Predict the next token. |
| Survival | `surv_weight` (default `1.0`) | Predict whether / when the next event occurs. |
| Value regression | `value_weight` (default `0.0`) | Predict the numeric value of measurement events. |

Upstream used `surv_weight = 1`, `value_weight = 0.1` for models that
also learned measurement magnitudes (e.g. BMI, blood pressure).

------------------------------------------------------------------------

### `include_unk`

| RSurvivEHR default | Notes                                  |
|--------------------|----------------------------------------|
| `TRUE`             | Always recommended for production use. |

Reserves `<UNK>` (index 1). Any unseen event code at fine-tune /
prediction time is silently mapped to `<UNK>`.

------------------------------------------------------------------------

### `include_cls_sep`

| RSurvivEHR default | Notes |
|----|----|
| `FALSE` | The original SurvivEHR model does not use CLS/SEP tokens. |

Set `TRUE` for BERT / BEHRT-style sequences. Not used in the upstream
GPT-style pre-training.

------------------------------------------------------------------------

### `time_scale`

All raw ages are divided by `time_scale` before entering the model.
**Must be consistent** across pre-training, fine-tuning, and prediction.

| Age unit                                    | Recommended `time_scale` |
|---------------------------------------------|--------------------------|
| Years (e.g. `50.0`)                         | `1.0`                    |
| Days since birth (e.g. `18262` ~= 50 years) | `365.25`                 |

------------------------------------------------------------------------

### `device`

| Value              | Behaviour                          |
|--------------------|------------------------------------|
| `"auto"` (default) | GPU (CUDA) if available, else CPU. |
| `"cpu"`            | Force CPU.                         |
| `"cuda"`           | Force GPU – error if unavailable.  |

------------------------------------------------------------------------

## Architecture notes

### NeoGPT block

RSurvivEHR always uses the **NeoGPT** block (`block_type = "Neo"`):

- **Rotary Position Embeddings (RoPE)** inside attention.
- **SwiGLU** activation in the MLP.
- **Pre-LN** (LayerNorm before each sub-layer).

### Neural-ODE survival head

For the **competing-risk** head, one ODE is instantiated per outcome
event, each with an encoder MLP, a lightweight ODE right-hand-side
network, and a sigmoid CIF output. For the **single-risk** head, a
single ODE produces one survival curve.

### Vocabulary construction

Built automatically during
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md):

1.  Unique `event` codes collected from the events data frame.
2.  Special tokens prepended at fixed indices:
    - `<PAD>` -\> index 0 (always; `padding_idx = 0`)
    - `<UNK>` -\> index 1 (when `include_unk = TRUE`)
    - `<CLS>` -\> index 2 (when `include_cls_sep = TRUE`)
    - `<SEP>` -\> index 3 (when `include_cls_sep = TRUE`)
3.  Clinical tokens assigned indices \>= 2 (or \>= 4 with CLS/SEP),
    sorted **alphabetically**.

At fine-tuning, outcome codes absent from the vocab are **automatically
appended** and the embedding layer is extended using a weight-safe copy
from the pre-trained backbone.

------------------------------------------------------------------------

## Recommended configurations

### Laptop / small dataset (\< 5 000 patients)

``` r

cfg_small <- survivehr_config(
  block_size    = 64,
  n_layer       = 2,
  n_head        = 2,
  n_embd        = 64,
  dropout       = 0.1,
  learning_rate = 3e-4,
  epochs        = 20,
  batch_size    = 8,
  surv_layer    = "competing-risk"
)
```

### Workstation / medium dataset (5 000–50 000 patients)

``` r

cfg_medium <- survivehr_config(
  block_size    = 128,
  n_layer       = 4,
  n_head        = 4,
  n_embd        = 256,
  dropout       = 0.0,
  learning_rate = 3e-4,
  epochs        = 10,
  batch_size    = 32,
  surv_layer    = "competing-risk"
)
```

### HPC / large dataset – upstream CPRD settings (\> 100 000 patients)

``` r

# Pre-training
cfg_pretrain <- survivehr_config(
  block_size    = 256,
  n_layer       = 6,
  n_head        = 6,
  n_embd        = 384,
  dropout       = 0.0,
  learning_rate = 3e-4,
  epochs        = 10,
  batch_size    = 64,
  surv_layer    = "competing-risk"
)

# Fine-tuning (larger block size, adjusted lr and batch_size)
cfg_finetune <- survivehr_config(
  block_size    = 512,
  n_layer       = 6,
  n_head        = 6,
  n_embd        = 384,
  dropout       = 0.0,
  learning_rate = 5e-4,   # head lr; backbone uses 5e-5 internally
  epochs        = 20,
  batch_size    = 512,
  surv_layer    = "competing-risk"
)
```

------------------------------------------------------------------------

## Fine-tuning-specific options

| Option | Upstream fine-tune default | Description |
|----|----|----|
| Backbone learning rate | `5e-5` | Preserve pre-trained weights. |
| Head learning rate | `5e-4` | Pass as `learning_rate` in [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md). |
| Early stopping | Enabled (patience 30) | Stops when validation loss plateaus. |
| Scheduler | ReduceLROnPlateau | Reduces backbone LR on plateau. |
| `PEFT.method` | `NULL` (full fine-tune) | Options: `"fix"` (frozen backbone), `"adapter"` (LoRA-style). |
| `PEFT.adapter_dim` | `8` | Bottleneck dimension for the adapter module. |
| `compression_layer` | `FALSE` | LayerNorm -\> Linear -\> GELU -\> Dropout between backbone and head. |
| `llrd` | `NULL` (disabled) | Layer-wise learning rate decay (float 0-1). |

------------------------------------------------------------------------

## References

Gadd, C. et al. (2025). *SurvivEHR: Transformer-based survival analysis
on electronic health records*. medRxiv.
[doi:10.1101/2025.08.04.25332916](https://doi.org/10.1101/2025.08.04.25332916)
