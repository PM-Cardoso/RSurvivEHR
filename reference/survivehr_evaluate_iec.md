# High-Level Inter-Event Concordance Evaluation

Performs comprehensive IEC evaluation on a pretrained model across a
dataset. Extracts risk scores and observed next-event IDs in batches to
avoid memory overhead from storing one massive risk matrix.

## Usage

``` r
survivehr_evaluate_iec(
  model,
  events,
  static = NULL,
  batch_size = 32,
  stratify_by_event = TRUE,
  aggregate_only = TRUE
)
```

## Arguments

- model:

  Fitted pretrain model bundle from
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md).

- events:

  Data frame with patient event history.

- static:

  Optional static covariates data frame.

- batch_size:

  Integer. Number of patients to process per batch.

- stratify_by_event:

  Logical. If TRUE, compute IEC separately for each true next-event
  type.

- aggregate_only:

  Logical. If TRUE, return only summary statistics.

## Value

S3 object of class `"survivehr_iec_eval"`.
