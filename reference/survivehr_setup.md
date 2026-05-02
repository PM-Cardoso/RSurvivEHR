# Set up the SurvivEHR Python environment

Creates a dedicated Python virtual environment named `"RSurvivEHR"` and
installs all required packages. Only needs to be called **once** after
installing the R package. Subsequent calls skip reinstallation unless
`force = TRUE`.

## Usage

``` r
survivehr_setup(envname = "RSurvivEHR", force = FALSE, python = NULL)
```

## Arguments

- envname:

  Name of the virtualenv to create/use. Default `"RSurvivEHR"`.

- force:

  If `TRUE`, reinstall all packages even when the environment already
  exists.

- python:

  Path to the Python executable. `NULL` (default) lets reticulate choose
  Python 3.9+ automatically.

## Value

Invisibly returns the path to the Python executable in the new
environment.

## Details

Works on macOS, Windows, and Linux. An internet connection is required
on first run so that pip can download the packages.

## Examples

``` r
if (FALSE) { # \dontrun{
# Run once after installing the package:
survivehr_setup()

# Force a clean reinstall:
survivehr_setup(force = TRUE)
} # }
```
