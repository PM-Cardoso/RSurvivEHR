# Save an RSurvivEHR model bundle to disk

Serialises a model bundle (returned by
[`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
or
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md))
to a `.pt` file. The bundle includes the model weights, vocabulary,
static column schema, `time_scale`, `value_standardization`, token
policy, and training history. Reload with
[`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).

## Usage

``` r
survivehr_save_model(model_bundle, path)
```

## Arguments

- model_bundle:

  A model bundle returned by a training function.

- path:

  File path for the output file. Should end in `.pt`.

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
