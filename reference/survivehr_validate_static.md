# Validate static covariates schema for SurvivEHR

Validate static covariates schema for SurvivEHR

## Usage

``` r
survivehr_validate_static(static_covariates)
```

## Arguments

- static_covariates:

  data.frame with patient id and covariates. Accepts `patient_id` or
  `PATIENT_ID`.

## Value

Invisibly returns TRUE.

## Examples

``` r
static <- data.frame(
  patient_id = c(1, 2),
  sex = c("M", "F"),
  imd = c(3L, 1L)
)
survivehr_validate_static(static)
#> [OK] Static covariates: 2 patients, 2 covariate column(s): sex, imd.
```
