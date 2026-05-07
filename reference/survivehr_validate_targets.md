# Validate the fine-tune targets data frame for RSurvivEHR

Checks that the targets data frame supplied to
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
is correctly formatted. Each row labels one patient: cases supply the
observed outcome event and its age; censored patients supply their last
observed non-outcome event and its age.

## Usage

``` r
survivehr_validate_targets(targets)
```

## Arguments

- targets:

  A `data.frame` with columns:

  `patient_id`

  :   Patient identifier matching the events frame.

  `target_event`

  :   Event code for the labelled outcome (cases) or the last observed
      non-outcome event (censored).

  `target_age`

  :   Age at the target event. Must be numeric and non-negative.

  `target_value`

  :   (Optional) Continuous measurement at the target event. `NA` for
      discrete events.

  Uppercase column-name aliases `PATIENT_ID`, `EVENT`, and
  `DAYS_SINCE_BIRTH` are also accepted for compatibility.

## Value

Invisibly returns `TRUE`. Prints a confirmation message listing the
number of labelled patients and all observed outcome values.

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
