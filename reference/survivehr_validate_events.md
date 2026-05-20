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

  A `data.frame` with columns (lowercase is the preferred canonical
  form; uppercase aliases are accepted for backward compatibility):

  `patient_id`

  :   Patient identifier (numeric or character). Alias: `PATIENT_ID`.

  `event`

  :   Clinical event code (character). Alias: `EVENT`.

  `age`

  :   Patient age at the event in consistent units (numeric,
      non-negative, time-ordered within patient). Alias:
      `DAYS_SINCE_BIRTH`.

  `value`

  :   (Optional) Continuous measurement recorded at the event (e.g.
      blood pressure or HbA1c). `NA` for discrete events. Alias:
      `VALUE`.

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
