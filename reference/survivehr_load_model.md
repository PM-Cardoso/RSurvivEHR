# Load an RSurvivEHR model bundle from disk

Restores a model bundle previously saved with
[`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md).
The returned object is identical in structure to the original bundle and
can be passed directly to
[`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
or used as `pretrained_model` in a further
[`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
call.

## Usage

``` r
survivehr_load_model(path)
```

## Arguments

- path:

  File path to a `.pt` bundle created by
  [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md).

## Value

A named list (model bundle) with elements `model`, `event_vocab`,
`inv_vocab`, `config`, `time_scale`, `token_policy`, `history`, and
`device`.

## Examples

``` r
if (FALSE) { # \dontrun{
ft2 <- survivehr_load_model("my_model.pt")
} # }
```
