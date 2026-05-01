# Load a SurvivEHR model bundle

Load a SurvivEHR model bundle

## Usage

``` r
survivehr_load_model(path)
```

## Arguments

- path:

  file path created by
  [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md).

## Value

Named list (model bundle) identical in structure to the original bundle
returned by
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md).

## Examples

``` r
if (FALSE) { # \dontrun{
ft2 <- survivehr_load_model("my_model.pt")
} # }
```
