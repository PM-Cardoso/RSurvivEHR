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

  A `data.frame` with columns (lowercase is the preferred canonical
  form; uppercase aliases are accepted for backward compatibility):

  `patient_id`

  :   Patient identifier matching the events frame. Alias: `PATIENT_ID`.

  `target_event`

  :   Event code for the labelled outcome (cases) or the last observed
      non-outcome event (censored). Aliases: `TARGET_EVENT`, `EVENT`.

  `target_age`

  :   Age at the target event. Must be numeric and non-negative.
      Aliases: `TARGET_AGE`, `DAYS_SINCE_BIRTH`.

  `target_value`

  :   (Optional) Continuous measurement at the target event. `NA` for
      discrete events. Aliases: `TARGET_VALUE`, `VALUE`.

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
