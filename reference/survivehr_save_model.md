# Save a SurvivEHR model bundle

Save a SurvivEHR model bundle

## Usage

``` r
survivehr_save_model(model_bundle, path)
```

## Arguments

- model_bundle:

  object returned by training functions.

- path:

  file path ending in `.pt`.

## Value

Invisibly returns `path`.

## Examples

``` r
if (FALSE) { # \dontrun{
tmp <- tempfile(fileext = ".pt")
survivehr_save_model(ft, tmp)
ft2 <- survivehr_load_model(tmp)
unlink(tmp)
} # }
```
