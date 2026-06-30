#' Compute Inter-Event Concordance (IEC) from Risk Scores
#'
#' Calculates Inter-Event Concordance (IEC) metrics from model risk scores and observed
#' events. This is a pure metric calculation function that works with pre-computed risk
#' score matrices, typically obtained from model predictions.
#'
#' This implements the ranking-based concordance calculation from the original SurvivEHR
#' model: for each transition, ranks all possible next events by their predicted risk,
#' finds the rank position of the true observed event, and normalizes to 0,1.
#'
#' @param risk_scores Matrix of shape (n_transitions, n_events) where each row contains
#'   risk scores for all possible events at that transition. Can also be a data.frame.
#'
#' @param observed_events Integer vector of length n_transitions containing the 1-indexed
#'   ID of the actually observed event at each transition. Must match nrow(risk_scores).
#'
#' @param stratify_by_event Logical. If TRUE, compute IEC separately for each event type
#'   to identify if concordance is biased by event prevalence.
#'
#' @param event_vocabulary Optional character vector of event names/IDs, length n_events.
#'   Used for labeling stratified results. If NULL, events are labeled as "1", "2", etc.
#'
#' @return An S3 object of class "survivehr_iec" (list) with elements:
#'   \item{mean_iec}{Numeric. Global mean IEC across all valid transitions.}
#'   \item{n_valid}{Integer. Number of successfully processed transitions.}
#'   \item{n_total}{Integer. Total number of transitions.}
#'   \item{iec_values}{Numeric vector. Per-transition IEC values.}
#'   \item{observed_ranks}{Integer vector. Per-transition rank positions (1-indexed).}
#'   \item{observed_ranks_from_top}{Integer vector. Per-transition top-rank interpretation.}
#'   \item{by_event}{Data frame (if stratified). Columns: event_id, mean_iec, n_obs.}
#'   \item{errors}{Character vector. Error messages for invalid transitions.}
#'   \item{stratified}{Logical. Whether stratification was performed.}
#'   \item{event_vocabulary}{Character vector. Event names (if provided).}
#'
#' @details
#' IEC Interpretation:
#' \itemize{
#'   \item IEC ≈ 1.0: Observed event ranked among highest-risk (good prediction)
#'   \item IEC ≈ 0.5: Observed event ranked in middle of risk distribution
#'   \item IEC ≈ 0.0: Observed event ranked among lowest-risk (poor prediction)
#'   \item mean_rank_from_top ≈ n_events - mean(observed_rank_from_top)
#' }
#'
#' The metric is designed for use during model evaluation to assess how well predicted
#' risk scores rank actual next events. Stratification reveals if the model performs
#' better/worse on common vs. rare events.
#'
#' @examples
#' \dontrun{
#' # Synthetic example: 10 transitions, 4 possible events
#' risk_matrix <- matrix(runif(40), nrow = 10, ncol = 4)
#' observed_events <- c(1, 2, 1, 3, 4, 2, 1, 3, 2, 1)
#'
#' # Basic IEC
#' iec_result <- survivehr_compute_iec(
#'   risk_scores = risk_matrix,
#'   observed_events = observed_events
#' )
#' print(iec_result)
#'
#' # With stratification and event names
#' iec_result <- survivehr_compute_iec(
#'   risk_scores = risk_matrix,
#'   observed_events = observed_events,
#'   stratify_by_event = TRUE,
#'   event_vocabulary = c("HTN", "DM", "MI", "CKD")
#' )
#' print(iec_result)
#' }
#'
#' @export
survivehr_compute_iec <- function(
  risk_scores,
  observed_events,
  stratify_by_event = FALSE,
  event_vocabulary = NULL
) {
  # Validate inputs
  if (!is.matrix(risk_scores) && !is.data.frame(risk_scores)) {
    stop("risk_scores must be a matrix or data.frame")
  }

  if (is.data.frame(risk_scores)) {
    risk_scores <- as.matrix(risk_scores)
  }

  if (!is.vector(observed_events) || !is.numeric(observed_events)) {
    stop("observed_events must be a numeric vector")
  }

  if (nrow(risk_scores) != length(observed_events)) {
    stop(
      "Dimension mismatch: nrow(risk_scores)=", nrow(risk_scores),
      " but length(observed_events)=", length(observed_events)
    )
  }

  n_transitions <- nrow(risk_scores)
  n_events <- ncol(risk_scores)

  if (is.null(event_vocabulary)) {
    event_vocabulary <- as.character(1:n_events)
  } else if (length(event_vocabulary) != n_events) {
    stop(
      "event_vocabulary length (", length(event_vocabulary),
      ") does not match n_events (", n_events, ")"
    )
  }

  # Ensure integer indexing (1-indexed)
  observed_events <- as.integer(observed_events)

  # Convert risk matrix to row-wise list (one vector per transition)
  risk_scores_list <- lapply(seq_len(nrow(risk_scores)), function(i) {
    as.numeric(risk_scores[i, ])
  })

  # Call Python backend
  backend <- .survivehr_backend()

  if (stratify_by_event) {
    # Call stratified computation
    py_result <- backend$compute_iec_stratified(
      risk_scores_list = risk_scores_list,
      observed_events = as.list(observed_events),
      event_vocabulary = as.list(event_vocabulary),
      vocab_size = as.integer(n_events),
      suppress_errors = TRUE
    )
  } else {
    # Call batch computation
    py_result <- backend$compute_iec_batch(
      risk_scores_list = risk_scores_list,
      observed_events = as.list(observed_events),
      vocab_size = as.integer(n_events),
      suppress_errors = TRUE
    )
  }

  # Convert Python result to R list
  result <- convert_iec_result(
    py_result = py_result,
    stratified = stratify_by_event,
    event_vocabulary = event_vocabulary,
    n_transitions = n_transitions,
    n_events = n_events
  )

  class(result) <- c("survivehr_iec", "list")
  return(result)
}


#' Print Method for IEC Results
#'
#' @param x Object of class "survivehr_iec"
#' @param ... Additional arguments (ignored)
#'
#' @export
print.survivehr_iec <- function(x, ...) {
  cat("=== SurvivEHR Inter-Event Concordance (IEC) ===\n")
  cat("Mean IEC: ", sprintf("%.4f", x$mean_iec), " (", x$n_valid, " / ", x$n_total,
      " valid transitions)\n", sep = "")

  if (x$n_total > x$n_valid) {
    cat("Errors: ", x$n_total - x$n_valid, " invalid transitions\n", sep = "")
  }

  if (x$stratified && !is.null(x$by_event)) {
    cat("\nIEC by event type:\n")
    print(x$by_event, row.names = FALSE)
  }

  cat("\n")
  invisible(x)
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================




#' Convert Python IEC Result to R List
#'
#' @param py_result List returned from Python backend
#' @param stratified Logical. Whether stratification was requested
#' @param event_vocabulary Character vector of event names
#' @param n_transitions Integer. Total number of transitions
#' @param n_events Integer. Vocabulary size
#'
#' @return Structured list with IEC results
#'
#' @keywords internal
convert_iec_result <- function(py_result, stratified, event_vocabulary, n_transitions, n_events) {
  if (stratified) {
    # Stratified result
    by_event <- NULL

    if (!is.null(py_result$by_event)) {
      # Convert dict to data frame
      by_event_list <- py_result$by_event
      by_event <- data.frame(
        event = names(by_event_list),
        mean_iec = sapply(by_event_list, function(x) x$mean_iec),
        n_obs = sapply(by_event_list, function(x) x$n_obs),
        stringsAsFactors = FALSE
      )
      rownames(by_event) <- NULL
    }

    return(list(
      mean_iec = py_result$mean_iec,
      n_valid = py_result$n_valid,
      n_total = n_transitions,
      iec_values = unlist(py_result$iec_values),
      observed_ranks = unlist(py_result$observed_ranks),
      observed_ranks_from_top = unlist(py_result$observed_ranks_from_top),
      by_event = by_event,
      errors = py_result$errors,
      stratified = TRUE,
      event_vocabulary = event_vocabulary
    ))
  } else {
    # Non-stratified result
    return(list(
      mean_iec = mean(unlist(py_result$iec_values), na.rm = TRUE),
      n_valid = py_result$n_valid,
      n_total = n_transitions,
      iec_values = unlist(py_result$iec_values),
      observed_ranks = unlist(py_result$observed_ranks),
      observed_ranks_from_top = unlist(py_result$observed_ranks_from_top),
      by_event = NULL,
      errors = py_result$errors,
      stratified = FALSE,
      event_vocabulary = event_vocabulary
    ))
  }
}
