# Convert Python IEC Result to R List

Convert Python IEC Result to R List

## Usage

``` r
convert_iec_result(
  py_result,
  stratified,
  event_vocabulary,
  n_transitions,
  n_events
)
```

## Arguments

- py_result:

  List returned from Python backend

- stratified:

  Logical. Whether stratification was requested

- event_vocabulary:

  Character vector of event names

- n_transitions:

  Integer. Total number of transitions

- n_events:

  Integer. Vocabulary size

## Value

Structured list with IEC results
