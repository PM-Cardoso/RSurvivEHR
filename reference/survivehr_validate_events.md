# Validate the events data frame for RSurvivEHR

Checks that the events data frame supplied to
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
or
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
is correctly formatted before any Python call is made, giving an
informative R error instead of a cryptic Python traceback. Specifically,
it verifies that the required columns are present, that ages are numeric
and non-negative, and that rows are time-ordered within each patient.

## Usage

``` r
survivehr_validate_events(events)
```

## Arguments

- events:

  A `data.frame` with columns:

  `patient_id`

  :   Patient identifier (numeric or character).

  `event`

  :   Clinical event code (character).

  `age`

  :   Patient age at the event in consistent units (numeric,
      non-negative, time-ordered within patient). The uppercase alias
      `DAYS_SINCE_BIRTH` is also accepted.

  `value`

  :   (Optional) Continuous measurement recorded at the event (e.g.
      blood pressure or HbA1c). `NA` for discrete events.

  Uppercase column-name aliases `PATIENT_ID` and `EVENT` are also
  accepted for compatibility.

## Value

Invisibly returns `TRUE`. Prints a confirmation message on success.

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
