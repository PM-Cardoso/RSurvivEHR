# Package index

## Setup

Install and configure the Python backend.

- [`survivehr_setup()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_setup.md)
  : Set up the SurvivEHR Python environment

## Configuration

Build and inspect model configuration.

- [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  : Build an RSurvivEHR configuration list

## Training

Pre-train, fine-tune, save, and load SurvivEHR models.

- [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
  : Pre-train the RSurvivEHR backbone transformer
- [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  : Fine-tune the RSurvivEHR backbone on labelled outcomes
- [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  : Predict cumulative incidence with a fine-tuned RSurvivEHR model
- [`survivehr_predict_value()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict_value.md)
  : Predict the numeric value of a named event (pretrain or fine-tuned
  models)
- [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
  : Save an RSurvivEHR model bundle to disk
- [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md)
  : Load an RSurvivEHR model bundle from disk

## Validation helpers

Validate input data frames before passing to training functions.

- [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md)
  : Validate the events data frame for RSurvivEHR
- [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md)
  : Validate the static covariates data frame for RSurvivEHR
- [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md)
  : Validate the fine-tune targets data frame for RSurvivEHR

## Evaluation

Evaluate how well model risk scores rank competing next events with
Inter-Event Concordance (IEC).

- [`survivehr_compute_iec()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_compute_iec.md)
  : Compute Inter-Event Concordance (IEC) from Risk Scores
- [`survivehr_predict_event_risks()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict_event_risks.md)
  : Extract Event Risk Scores for IEC Evaluation
- [`survivehr_evaluate_iec()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_evaluate_iec.md)
  : High-Level Inter-Event Concordance Evaluation
