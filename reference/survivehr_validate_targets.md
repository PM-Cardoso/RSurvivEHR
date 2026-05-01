# Validate fine-tune targets schema for SurvivEHR

Validate fine-tune targets schema for SurvivEHR

## Usage

``` r
survivehr_validate_targets(targets)
```

## Arguments

- targets:

  data.frame with columns `patient_id`, `target_event`, `target_age` or
  FastEHR aliases (`PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`). Optional
  `target_value` / `VALUE`.

## Value

Invisibly returns TRUE.

## Examples

``` r
targets <- data.frame(
  patient_id   = c(5L, 6L),
  target_event = c("CVD", "DEATH"),
  target_age   = c(51.2, 64.1)
)
survivehr_validate_targets(targets)
#> [OK] Targets: 2 labelled patients, outcome(s): CVD, DEATH.
```
