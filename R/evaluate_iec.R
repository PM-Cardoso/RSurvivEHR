#' High-Level Inter-Event Concordance Evaluation
#'
#' Performs comprehensive IEC evaluation on a pretrained model across a dataset.
#' Automatically extracts observed events and computes IEC in batches to avoid
#' memory overhead of storing massive risk matrices.
#'
#' @param model Fitted pretrain model bundle from `survivehr_pretrain()`.
#'
#' @param events Data frame with patient event history.
#'
#' @param static Optional static covariates data frame.
#'
#' @param batch_size Integer. Number of patients to process per batch (default: 32).
#'   Increase for speed, decrease for memory safety.
#'
#' @param stratify_by_event Logical. If TRUE, compute IEC separately for each
#'   event type to identify if model has bias by event prevalence (default: TRUE).
#'
#' @param aggregate_only Logical. If TRUE (default), return only summary statistics
#'   (mean_iec, by_event). If FALSE, return full per-transition IEC values
#'   (can be memory-intensive for large datasets).
#'
#' @return S3 object of class "survivehr_iec_eval" (list) with elements:
#'   \item{mean_iec}{Numeric. Global mean IEC across all transitions.}
#'   \item{n_valid}{Integer. Number of successfully processed transitions.}
#'   \item{n_total}{Integer. Total number of transitions.}
#'   \item{by_event}{Data frame (if stratified). Columns: event_id, mean_iec, n_obs.}
#'   \item{by_batch}{List of per-batch statistics (if aggregate_only=FALSE).}
#'   \item{errors}{Character vector. Error messages from failed transitions.}
#'   \item{stratified}{Logical. Whether stratification was requested.}
#'   \item{aggregate_only}{Logical. Whether only aggregates are stored.}
#'
#' @details
#' Processing:
#' 1. Extracts observed events from input data (transitions: i → i+1)
#' 2. Processes patients in batches to manage memory
#' 3. For each batch:
#'    - Extracts risk scores via `survivehr_predict_event_risks()`
#'    - Computes IEC via `survivehr_compute_iec()`
#'    - Aggregates statistics
#' 4. Returns summary or full results depending on `aggregate_only`
#'
#' @examples
#' \dontrun{
#' # Evaluate on test set with stratification
#' eval_result <- survivehr_evaluate_iec(
#'   model = pretrained_model,
#'   events = test_events,
#'   static = test_static,
#'   batch_size = 32,
#'   stratify_by_event = TRUE,
#'   aggregate_only = TRUE
#' )
#' print(eval_result)
#' }
#'
#' @export
survivehr_evaluate_iec <- function(
  model,
  events,
  static = NULL,
  batch_size = 32,
  stratify_by_event = TRUE,
  aggregate_only = TRUE
) {
  # Validate model
  if (!is.list(model) || is.null(model$model) || is.null(model$event_vocab)) {
    stop("model must be a RSurvivEHR model bundle with model and event_vocab")
  }

  if (!is.data.frame(events)) {
    stop("events must be a data frame")
  }

  # Note: Observed events are extracted per batch using the vocabulary from risk extraction.
  # This ensures proper alignment with the model's actual event mappings.

  # Split data by patient for batch processing
  unique_patients <- unique(events$patient_id)
  n_patients <- length(unique_patients)
  n_batches <- ceiling(n_patients / batch_size)

  # Initialize aggregation
  all_iec_values <- numeric()
  all_observed_events <- integer()  # Track all observed events for final count
  all_observed_ranks <- integer()
  all_observed_ranks_from_top <- integer()
  all_errors <- character()
  by_event_aggregates <- list()
  batch_results <- list()

  # Process batches
  for (batch_idx in 1:n_batches) {
    batch_start <- (batch_idx - 1) * batch_size + 1
    batch_end <- min(batch_idx * batch_size, n_patients)
    batch_patients <- unique_patients[batch_start:batch_end]

    # Subset data
    events_batch <- events[events$patient_id %in% batch_patients, ]
    static_batch <- if (!is.null(static)) static[static$patient_id %in% batch_patients, ] else NULL

    # Extract risk scores for this batch
    tryCatch(
      {
        risks <- survivehr_predict_event_risks(
          model = model,
          events = events_batch,
          static = static_batch,
          max_patients = NULL
        )

        # Flatten risk_scores (list of per-patient matrices) into single matrix
        if (is.null(risks$risk_scores) || length(risks$risk_scores) == 0) {
          stop("No risk scores returned from model prediction")
        }
        
        risk_matrix <- do.call(rbind, risks$risk_scores)
        
        if (!is.matrix(risk_matrix) && !is.data.frame(risk_matrix)) {
          stop(
            "do.call(rbind, ...) failed to create matrix. ",
            "Got class: ", paste(class(risk_matrix), collapse=", "),
            ". risk_scores has ", length(risks$risk_scores), " elements. ",
            "First element class: ", if(length(risks$risk_scores) > 0) 
              paste(class(risks$risk_scores[[1]]), collapse=", ") else "N/A"
          )
        }

        # Get observed events for this batch using the vocab from risk extraction
        obs_batch <- extract_observed_events(events_batch, risks$event_vocab)

        # Sanity check: dimensions must match
        if (nrow(risk_matrix) != length(obs_batch$observed_events)) {
          stop(
            "Risk/observed transition mismatch in batch ", batch_idx, ": ",
            "nrow(risk_matrix) = ", nrow(risk_matrix),
            ", length(observed_events) = ", length(obs_batch$observed_events)
          )
        }

        # Compute IEC for this batch
        iec_batch <- survivehr_compute_iec(
          risk_scores = risk_matrix,
          observed_events = obs_batch$observed_events,
          stratify_by_event = stratify_by_event,
          event_vocabulary = risks$event_names
        )

        # Accumulate IEC values and observed events
        all_iec_values <- c(all_iec_values, iec_batch$iec_values)
        all_observed_events <- c(all_observed_events, obs_batch$observed_events)
        all_observed_ranks <- c(all_observed_ranks, iec_batch$observed_ranks)
        all_observed_ranks_from_top <- c(all_observed_ranks_from_top, iec_batch$observed_ranks_from_top)
        all_errors <- c(all_errors, iec_batch$errors)

        # Accumulate stratified results
        if (stratify_by_event && !is.null(iec_batch$by_event)) {
          for (idx in seq_len(nrow(iec_batch$by_event))) {
            event_id <- iec_batch$by_event$event[idx]
            if (!(event_id %in% names(by_event_aggregates))) {
              by_event_aggregates[[event_id]] <- list(iec_sum = 0, n_obs = 0)
            }
            by_event_aggregates[[event_id]]$iec_sum <- by_event_aggregates[[event_id]]$iec_sum +
              iec_batch$by_event$mean_iec[idx] * iec_batch$by_event$n_obs[idx]
            by_event_aggregates[[event_id]]$n_obs <- by_event_aggregates[[event_id]]$n_obs +
              iec_batch$by_event$n_obs[idx]
          }
        }

        if (!aggregate_only) {
          batch_results[[batch_idx]] <- list(
            batch_patients = batch_patients,
            iec_values = iec_batch$iec_values,
            n_valid = iec_batch$n_valid,
            errors = iec_batch$errors
          )
        }
      },
      error = function(e) {
        warning("Batch ", batch_idx, " failed: ", e$message)
        all_errors <<- c(all_errors, paste("Batch error:", e$message))
      }
    )
  }

  # Finalize stratified aggregates
  by_event_df <- NULL
  if (stratify_by_event && length(by_event_aggregates) > 0) {
    by_event_df <- data.frame(
      event = names(by_event_aggregates),
      mean_iec = sapply(by_event_aggregates, function(x) if (x$n_obs > 0) x$iec_sum / x$n_obs else 0),
      n_obs = sapply(by_event_aggregates, function(x) x$n_obs),
      stringsAsFactors = FALSE
    )
    rownames(by_event_df) <- NULL
  }

  # Final overall summary
  # If iec_values were not stored (aggregate_only=TRUE), compute overall from stratified table
  if (length(all_iec_values) > 0) {
    final_mean_iec <- mean(all_iec_values, na.rm = TRUE)
    final_n_valid <- sum(!is.na(all_iec_values))
  } else if (!is.null(by_event_df) && nrow(by_event_df) > 0) {
    # Fallback: compute weighted mean from stratified results
    final_n_valid <- sum(by_event_df$n_obs, na.rm = TRUE)
    final_mean_iec <- if (final_n_valid > 0) {
      sum(by_event_df$mean_iec * by_event_df$n_obs, na.rm = TRUE) / final_n_valid
    } else {
      0
    }
  } else {
    final_mean_iec <- 0
    final_n_valid <- 0
  }

  # Create result object
  result <- list(
    mean_iec = final_mean_iec,
    n_valid = final_n_valid,
    n_total = length(all_observed_events),
    iec_values = if (aggregate_only) NULL else all_iec_values,
    observed_ranks = if (aggregate_only) NULL else all_observed_ranks,
    observed_ranks_from_top = if (aggregate_only) NULL else all_observed_ranks_from_top,
    by_event = by_event_df,
    by_batch = if (aggregate_only) NULL else batch_results,
    errors = all_errors,
    stratified = stratify_by_event,
    aggregate_only = aggregate_only
  )

  class(result) <- c("survivehr_iec_eval", "list")
  return(result)
}


#' Print Method for IEC Evaluation Results
#'
#' @param x Object of class "survivehr_iec_eval"
#' @param ... Additional arguments (ignored)
#'
#' @export
print.survivehr_iec_eval <- function(x, ...) {
  cat("=== SurvivEHR IEC Evaluation ===\n")
  cat("Mean IEC: ", sprintf("%.4f", x$mean_iec), " (", x$n_valid, " / ", x$n_total,
      " valid transitions)\n", sep = "")

  if (x$n_total > x$n_valid) {
    n_errors <- x$n_total - x$n_valid
    cat("Skipped: ", n_errors, " transitions\n", sep = "")
  }

  if (x$stratified && !is.null(x$by_event)) {
    cat("\nIEC by event type:\n")
    print(x$by_event, row.names = FALSE)
  }

  if (!x$aggregate_only && !is.null(x$by_batch) && length(x$by_batch) > 0) {
    cat("\nBatch processing: ", length(x$by_batch), " batches\n", sep = "")
  }

  cat("\n")
  invisible(x)
}


# ============================================================================
# HELPERS
# ============================================================================

#' Extract Observed Events from Transitions
#'
#' From sequential events per patient, extracts the observed next event
#' at each transition using the model's event vocabulary (frequency-ordered).
#'
#' @param events Data frame with columns: patient_id, event, age, ...
#' @param event_vocab List mapping event names to 1-indexed vocabulary IDs
#'
#' @return List with:
#'   - observed_events: Integer vector of 1-indexed event IDs (from event_vocab)
#'   - patient_mapping: Character vector of patient_id per transition
#'
#' @keywords internal
extract_observed_events <- function(events, event_vocab) {
  if (!all(c("patient_id", "event", "age") %in% names(events))) {
    stop("events must have columns: patient_id, event, age")
  }

  # Sort by patient and time
  events <- events[order(events$patient_id, events$age), ]

  observed_list <- list()
  patient_list <- list()

  # For each patient, get transitions
  for (pid in unique(events$patient_id)) {
    patient_events <- events[events$patient_id == pid, ]

    if (nrow(patient_events) < 2) {
      next  # Need at least 2 events for a transition
    }

    # Each row after the first is an observed event in the sequence
    for (i in 2:nrow(patient_events)) {
      observed_event_name <- patient_events$event[i]

      # Map to vocabulary ID using model's event_vocab (frequency-ordered)
      if (!(observed_event_name %in% names(event_vocab))) {
        next  # Skip unknown events
      }

      vocab_id <- as.integer(event_vocab[[observed_event_name]])
      observed_list[[length(observed_list) + 1]] <- vocab_id
      patient_list[[length(patient_list) + 1]] <- pid
    }
  }

  return(list(
    observed_events = unlist(observed_list),
    patient_mapping = unlist(patient_list)
  ))
}
