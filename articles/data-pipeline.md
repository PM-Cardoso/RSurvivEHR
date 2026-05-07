# Data pipeline: from R data frames to the model

This vignette traces exactly what happens to your R data frames between
the moment you call
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
/
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
and the moment tensors land inside the PyTorch model. Each step links to
the specific function in the source code that performs it.

The example data below is the 10-patient population used throughout the
[Getting
started](https://pm-cardoso.github.io/RSurvivEHR/articles/getting-started.md)
vignette.

------------------------------------------------------------------------

## The data we start with

``` r

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
  SEX           = c("M","F","M","F","M","F","M","F","M","F"),
  ETHNICITY     = c("White","Asian","White","Black","White",
                    "Asian","White","White","Black","White"),
  IMD           = c(3L, 1L, 5L, 2L, 4L, 3L, 1L, 5L, 2L, 4L),
  YEAR_OF_BIRTH = c(1960L,1970L,1952L,1975L,1963L,1958L,1978L,1960L,1968L,1975L)
)
```

------------------------------------------------------------------------

## Step 1 — R validation helpers

Before any Python code runs, the R functions call the validation helpers
to give readable error messages while still in R:

- [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md)
  checks that `events` has the required columns (`patient_id`, `event`,
  `age`) and that types are sensible.
- [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md)
  checks the static covariates frame.

Source:
[`R/validate.R`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/R/validate.R)

``` r

survivehr_validate_events(events_pop)   # silent = OK
survivehr_validate_static(static_pop)  # silent = OK
```

------------------------------------------------------------------------

## Step 2 — R calls the Python backend

[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
(source:
[`R/train.R`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/R/train.R))
passes the raw data frames directly to Python via **reticulate**:

``` r

backend$train_pretrain_model(
  events_df  = events,
  static_df  = static_covariates,
  config     = .to_py_dict(config),
  event_vocab = event_vocab
)
```

reticulate converts each R `data.frame` to a `pandas.DataFrame`
automatically. No manual serialisation is needed.

The backend module itself is loaded once and cached by the
`.survivehr_backend()` closure in
[`R/backend.R`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/R/backend.R).

------------------------------------------------------------------------

## Step 3 — Event cleaning (`_clean_events`)

Source: [`inst/python/survivehr_backend.py`,
`_clean_events()`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py)

The first thing the Python backend does is normalise the events frame:

| Operation | Detail |
|----|----|
| Column aliases | `DAYS_SINCE_BIRTH` → `age`, `PATIENT_ID` → `patient_id`, etc. |
| Sort | By `patient_id` then `age` (ascending) |
| Type coercion | `event` → `str`, `age` → `float64` |
| Missing `value` | Column added and filled with `NaN` if not supplied |

After this step the 48-row `events_pop` frame looks like:

       patient_id          event   age  value
    0           1   HYPERTENSION  55.0    NaN
    1           1         STATIN  55.5    NaN
    2           1       BP_CHECK  56.2  148.0
    3           1            CVD  58.0    NaN
    4           2   HYPERTENSION  44.0    NaN
    ...

------------------------------------------------------------------------

## Step 4 — Vocabulary construction (`_build_vocab_with_policy`)

Source: [`inst/python/survivehr_backend.py`,
`_build_vocab_with_policy()`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py)

A token vocabulary is built by **descending event frequency**, so the
most common events receive the smallest integer IDs. Special tokens are
reserved first:

| Token   | ID  | Reason                                         |
|---------|-----|------------------------------------------------|
| `<PAD>` | 0   | Always index 0; used to fill shorter sequences |
| `<UNK>` | 1   | Maps events not seen at training time          |

Clinical event tokens are then assigned in frequency order:

| Event        | Count | Token ID |
|--------------|-------|----------|
| HYPERTENSION | 10    | 2        |
| BP_CHECK     | 9     | 3        |
| STATIN       | 9     | 4        |
| T2D          | 8     | 5        |
| METFORMIN    | 5     | 6        |
| CVD          | 3     | 7        |
| HBA1C        | 3     | 8        |
| AMLODIPINE   | 1     | 9        |

Ties in frequency (e.g. BP_CHECK and STATIN both appear 9 times) are
broken alphabetically for reproducibility. This vocabulary is stored in
every model bundle as `event_vocab` and used identically at fine-tuning
and prediction time.

------------------------------------------------------------------------

## Step 5 — Tokenisation, age normalisation & padding (`_build_context_data`)

Source: [`inst/python/survivehr_backend.py`,
`_build_context_data()`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py)

For each patient the function:

1.  **Tokenises** each event string to its integer ID (unknown events →
    `<UNK>`).
2.  **Normalises ages** by dividing by `time_scale`:

``` math
\text{age\_norm} = \frac{\text{raw\_age}}{\text{time\_scale}}
```

With `time_scale = 5.0` (a 5-year prediction window), patient 1’s first
event at age 55.0 becomes `55.0 / 5.0 = 11.0`.

3.  **Pads** each sequence to `block_size` (e.g. 64). Real events occupy
    the first positions; remaining positions are filled with token ID
    `0` (`<PAD>`) and age `0.0`.

4.  **Builds an attention mask** — `True` for real event positions,
    `False` for padding — so the transformer ignores padded slots.

For patient 1 (4 events, `block_size = 64`):

    tokens:          [2, 4, 3, 7, 0, 0, ..., 0]   # HYPERTENSION STATIN BP_CHECK CVD <PAD>...
    ages (normed):   [11.0, 11.1, 11.24, 11.6, 0.0, ..., 0.0]
    values:          [NaN, NaN, 148.0, NaN, NaN, ..., NaN]
    attention_mask:  [T, T, T, T, F, F, ..., F]

All patients are stacked into arrays of shape `(n_patients, block_size)`
inside the `BuiltData` dataclass.

------------------------------------------------------------------------

## Step 6 — Static covariate encoding (`_encode_static`)

Source: [`inst/python/survivehr_backend.py`,
`_encode_static()`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py)

Static covariates are encoded into a float32 matrix:

- **Numeric columns** (`IMD`, `YEAR_OF_BIRTH`): passed through as-is;
  `NaN` filled with `0.0`.
- **Categorical columns** (`SEX`, `ETHNICITY`): one-hot encoded with
  `pd.get_dummies`.

For `static_pop`:

| Column | Input | Encoded columns |
|----|----|----|
| SEX | “M” / “F” | `SEX_F`, `SEX_M` |
| ETHNICITY | “White” / “Asian” / “Black” | `ETHNICITY_Asian`, `ETHNICITY_Black`, `ETHNICITY_White` |
| IMD | 1–5 (integer) | `IMD` (numeric, kept as-is) |
| YEAR_OF_BIRTH | 1952–1978 | `YEAR_OF_BIRTH` (numeric) |

This produces a matrix of shape `(n_patients, 7)`. The encoded column
list is stored in the bundle (`static_col_names`) and used at prediction
time to **reindex** unseen batches — any one-hot column absent from a
small prediction batch is filled with `0.0` rather than raising an
error.

------------------------------------------------------------------------

## Step 7 — Fine-tuning: target delta computation (`FineTuneDataset`)

Source: [`inst/python/survivehr_backend.py`,
`FineTuneDataset`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py)

For fine-tuning, each patient’s target label from `targets_cr` is
converted into a **normalised time-delta** — the time from the last
observed context event to the target outcome:

``` math
\delta_\text{norm} = \frac{\text{target\_age} - \text{last\_context\_age}}{\text{time\_scale}}
```

For patient 1 (context ends at `BP_CHECK @ 56.2`, target `CVD @ 58.0`,
`time_scale = 5.0`):

``` math
\delta_\text{norm} = \frac{58.0 - 56.2}{5.0} = \frac{1.8}{5.0} = 0.36
```

This normalised delta maps onto the ODE solver’s integration grid
$`[0, 1]`$, where $`1.0`$ represents the full `time_scale` horizon.

Patients whose `target_event` is not in the vocabulary, or who have no
matching row in `events`, are silently dropped from the training batch.

------------------------------------------------------------------------

## Step 8 — PyTorch `DataLoader` → model forward pass

Source: [`inst/python/survivehr_backend.py`, `PretrainDataset` /
`FineTuneDataset`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/survivehr_backend.py),
[`inst/python/SurvivEHR/experiments.py`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/SurvivEHR/experiments.py)

Each batch dictionary fed into the transformer contains:

| Key | Shape | dtype | Description |
|----|----|----|----|
| `tokens` | `[B, block_size]` | `int64` | Event token IDs |
| `ages` | `[B, block_size]` | `float32` | Normalised event ages |
| `values` | `[B, block_size]` | `float32` | Numeric event values (lab results etc.) |
| `attention_mask` | `[B, block_size]` | `bool` | `True` = real event, `False` = padding |
| `static_covariates` | `[B, n_static]` | `float32` | One-hot + numeric static features |
| `target_token` *(fine-tune)* | `[B]` | `int64` | Outcome event token ID |
| `target_age_delta` *(fine-tune)* | `[B]` | `float32` | Normalised time-to-event $`\delta`$ |

The transformer backbone (source:
[`inst/python/SurvivEHR/src/models/transformer/base.py`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/SurvivEHR/src/models/transformer/base.py))
processes `tokens`, `ages`, `values`, `attention_mask`, and
`static_covariates` to produce contextual embeddings. The survival ODE
head (source:
[`inst/python/SurvivEHR/src/modules/head_layers/survival/`](https://github.com/PM-Cardoso/RSurvivEHR/blob/main/inst/python/SurvivEHR/src/modules/head_layers/survival/))
then integrates those embeddings over the prediction window to produce
the survival CDF returned to R as `_cdf_last` and `_auc` columns.

------------------------------------------------------------------------

## Summary diagram

    R data.frame (events_pop / static_pop)
           │
           ▼  R/validate.R
      survivehr_validate_events()       ← column checks, type checks
           │
           ▼  R/train.R → reticulate
      pandas.DataFrame                  ← automatic R→Python conversion
           │
           ▼  survivehr_backend.py  _clean_events()
      sorted, typed, value-imputed      ← aliases resolved, sorted by age
           │
           ▼  survivehr_backend.py  _build_vocab_with_policy()
      event → integer token ID          ← frequency-descending; <PAD>=0, <UNK>=1
           │
           ▼  survivehr_backend.py  _build_context_data()
      tokens [n, block_size]  int64     ← padded token sequences
      ages   [n, block_size]  float32   ← age / time_scale, padded with 0
      values [n, block_size]  float32   ← lab values, padded with NaN
      mask   [n, block_size]  bool      ← True = real event
           │
           ▼  survivehr_backend.py  _encode_static()
      static [n, n_features]  float32   ← one-hot + numeric matrix
           │
           ▼  survivehr_backend.py  FineTuneDataset  (fine-tune only)
      target_token      [n]   int64     ← vocab ID of the outcome event
      target_age_delta  [n]   float32   ← (target_age − last_age) / time_scale
           │
           ▼  torch DataLoader
      batched tensors → transformer backbone → survival ODE head → loss / CDF
