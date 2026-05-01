# Package index

## Setup

Install and configure the Python backend.

- [`survivehr_setup()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_setup.md)
  : Set up the SurvivEHR Python environment

## Configuration

Build and inspect model configuration.

- [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  : Build default SurvivEHR configuration

## Training

Pre-train, fine-tune, save, and load SurvivEHR models.

- [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
  : Train a pretrain SurvivEHR model from R data frames
- [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  : Fine-tune SurvivEHR from R data frames
- [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  : Predict next events with SurvivEHR
- [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
  : Save a SurvivEHR model bundle
- [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md)
  : Load a SurvivEHR model bundle

## Validation helpers

Validate input data frames before passing to training functions.

- [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md)
  : Validate events schema for SurvivEHR
- [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md)
  : Validate static covariates schema for SurvivEHR
- [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md)
  : Validate fine-tune targets schema for SurvivEHR
