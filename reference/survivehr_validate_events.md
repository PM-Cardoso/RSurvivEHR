# Validate events schema for SurvivEHR

Validate events schema for SurvivEHR

## Usage

``` r
survivehr_validate_events(events)
```

## Arguments

- events:

  data.frame with columns `patient_id`, `event`, `age` (or FastEHR
  aliases `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`), optional
  `value`/`VALUE`.

## Value

Invisibly returns TRUE.

## Examples

``` r
events <- data.frame(
  patient_id = c(1, 1, 2, 2),
  event      = c("HYPERTENSION", "STATIN", "T2D", "METFORMIN"),
  age        = c(50.0, 50.5, 45.0, 45.3)
)
survivehr_validate_events(events)
#> [OK] Events: 4 rows, 2 patients. Columns present, ages numeric and time-ordered.
```
