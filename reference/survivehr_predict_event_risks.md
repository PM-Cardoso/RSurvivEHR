# Extract Event Risk Scores for IEC Evaluation

Extracts the full risk score matrix from a pretrained SurvivEHR model.
This function exposes the model's predicted risk for each possible next
event at each observed transition, together with the observed next-event
IDs used by the Python backend.

## Usage

``` r
survivehr_predict_event_risks(
  model,
  events,
  static = NULL,
  max_patients = NULL
)
```

## Arguments

- model:

  Fitted pretrain model bundle returned from
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
  or
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).

- events:

  Data frame with patient event history. Required columns: `patient_id`,
  `event`, `age`; optional column: `value`.

- static:

  Optional data frame with static covariates. Must contain `patient_id`
  and the same static covariate columns used during training.

- max_patients:

  Optional integer. Maximum number of patients to process. Useful for
  debugging or memory-limited runs. If `NULL`, all patients are
  processed.

## Value

List with elements:

- risk_matrix:

  Numeric matrix of shape `total_transitions x n_events`.

- risk_scores:

  List of per-patient risk matrices.

- observed_events:

  Integer vector of observed next-event IDs aligned to rows of
  `risk_matrix`.

- patient_ids:

  Character vector mapping each transition row to a patient ID.

- event_vocab:

  Named integer vector mapping event names to 1-indexed risk-matrix
  column IDs.

- event_names:

  Character vector of event names ordered by risk-matrix columns.

- event_vocab_table:

  Data frame with columns `event` and `event_id`.

- n_events:

  Integer. Number of predictable event types.

## Details

This function is mainly used internally by
[`survivehr_evaluate_iec()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_evaluate_iec.md),
but is exported because it is useful for debugging and manual IEC
calculations.

Observed events are returned from the Python backend from the same
model-built sequence as the risk scores. This avoids dimension
mismatches that can occur if observed transitions are reconstructed
independently in R.
