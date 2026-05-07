# Validate the static covariates data frame for RSurvivEHR

Checks that the static covariates data frame has a `patient_id` column
and at least one covariate column. Issues a warning (not an error) when
no covariate columns are found, because the backend will substitute an
intercept column in that case.

## Usage

``` r
survivehr_validate_static(static_covariates)
```

## Arguments

- static_covariates:

  A `data.frame` with a `patient_id` column (or `PATIENT_ID`) plus any
  number of covariate columns with freely chosen names. Pass the **same
  columns in the same order** across pretrain, fine-tune, and
  prediction.

## Value

Invisibly returns `TRUE`. Prints a confirmation message on success.

## Details

Categorical columns (those that are less than 80\\ one-hot encoded
automatically by the Python backend. Numeric columns are passed through
unchanged. The exact encoded column list is stored inside every model
bundle; passing different columns at prediction time raises a
descriptive error.

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
