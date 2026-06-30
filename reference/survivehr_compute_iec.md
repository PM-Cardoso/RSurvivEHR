# Compute Inter-Event Concordance (IEC) from Risk Scores

Calculates Inter-Event Concordance (IEC) metrics from model risk scores
and observed events. This is a pure metric calculation function that
works with pre-computed risk score matrices, typically obtained from
model predictions.

## Usage

``` r
survivehr_compute_iec(
  risk_scores,
  observed_events,
  stratify_by_event = FALSE,
  event_vocabulary = NULL
)
```

## Arguments

- risk_scores:

  Matrix of shape (n_transitions, n_events) where each row contains risk
  scores for all possible events at that transition. Can also be a
  data.frame.

- observed_events:

  Integer vector of length n_transitions containing the 1-indexed ID of
  the actually observed event at each transition. Must match
  nrow(risk_scores).

- stratify_by_event:

  Logical. If TRUE, compute IEC separately for each event type to
  identify if concordance is biased by event prevalence.

- event_vocabulary:

  Optional character vector of event names/IDs, length n_events. Used
  for labeling stratified results. If NULL, events are labeled as "1",
  "2", etc.

## Value

An S3 object of class "survivehr_iec" (list) with elements:

- mean_iec:

  Numeric. Global mean IEC across all valid transitions.

- n_valid:

  Integer. Number of successfully processed transitions.

- n_total:

  Integer. Total number of transitions.

- iec_values:

  Numeric vector. Per-transition IEC values.

- observed_ranks:

  Integer vector. Per-transition rank positions (1-indexed).

- observed_ranks_from_top:

  Integer vector. Per-transition top-rank interpretation.

- by_event:

  Data frame (if stratified). Columns: event_id, mean_iec, n_obs.

- errors:

  Character vector. Error messages for invalid transitions.

- stratified:

  Logical. Whether stratification was performed.

- event_vocabulary:

  Character vector. Event names (if provided).

## Details

This implements the ranking-based concordance calculation from the
original SurvivEHR model: for each transition, ranks all possible next
events by their predicted risk, finds the rank position of the true
observed event, and normalizes to 0,1.

IEC Interpretation:

- IEC ≈ 1.0: Observed event ranked among highest-risk (good prediction)

- IEC ≈ 0.5: Observed event ranked in middle of risk distribution

- IEC ≈ 0.0: Observed event ranked among lowest-risk (poor prediction)

- mean_rank_from_top ≈ n_events - mean(observed_rank_from_top)

The metric is designed for use during model evaluation to assess how
well predicted risk scores rank actual next events. Stratification
reveals if the model performs better/worse on common vs. rare events.

## Examples

``` r
if (FALSE) { # \dontrun{
# Synthetic example: 10 transitions, 4 possible events
risk_matrix <- matrix(runif(40), nrow = 10, ncol = 4)
observed_events <- c(1, 2, 1, 3, 4, 2, 1, 3, 2, 1)

# Basic IEC
iec_result <- survivehr_compute_iec(
  risk_scores = risk_matrix,
  observed_events = observed_events
)
print(iec_result)

# With stratification and event names
iec_result <- survivehr_compute_iec(
  risk_scores = risk_matrix,
  observed_events = observed_events,
  stratify_by_event = TRUE,
  event_vocabulary = c("HTN", "DM", "MI", "CKD")
)
print(iec_result)
} # }
```
