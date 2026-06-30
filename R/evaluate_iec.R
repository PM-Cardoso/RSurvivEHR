#' High-Level Inter-Event Concordance Evaluation
#'
#' Performs comprehensive IEC evaluation on a pretrained model across a dataset.
#' Extracts risk scores and observed next-event IDs in batches to avoid memory
#' overhead from storing one massive risk matrix.
#'
#' @param model Fitted pretrain model bundle from `survivehr_pretrain()`.
#' @param events Data frame with patient event history.
#' @param static Optional static covariates data frame.
#' @param batch_size Integer. Number of patients to process per batch.
#' @param stratify_by_event Logical. If TRUE, compute IEC separately for each
#'   true next-event type.
#' @param aggregate_only Logical. If TRUE, return only summary statistics.
#'
#' @return S3 object of class `"survivehr_iec_eval"`.
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

  # Validate input data
  survivehr_validate_events(events)

  if (!is.null(static)) {
    survivehr_validate_static(static)
  }

  if (!is.numeric(batch_size) || length(batch_size) != 1 || batch_size < 1) {
    stop("batch_size must be a positive integer")
  }

  batch_size <- as.integer(batch_size)

  # Split patients into batches
  unique_patients <- unique(events$patient_id)
  n_patients <- length(unique_patients)
  n_batches <- ceiling(n_patients / batch_size)

  # Initialise aggregation containers
  all_iec_values <- numeric()
  all_observed_events <- integer()
  all_observed_ranks <- integer()
  all_observed_ranks_from_top <- integer()
  all_errors <- character()
  by_event_aggregates <- list()
  batch_results <- list()

  # Process batches
  for (batch_idx in seq_len(n_batches)) {
    batch_start <- (batch_idx - 1) * batch_size + 1
    batch_end <- min(batch_idx * batch_size, n_patients)
    batch_patients <- unique_patients[batch_start:batch_end]

    events_batch <- events[events$patient_id %in% batch_patients, , drop = FALSE]

    static_batch <- NULL
    if (!is.null(static)) {
      static_batch <- static[static$patient_id %in% batch_patients, , drop = FALSE]
    }

    tryCatch(
      {
        risks <- survivehr_predict_event_risks(
          model = model,
          events = events_batch,
          static = static_batch,
          max_patients = NULL
        )

        if (is.null(risks$risk_matrix)) {
          stop("No risk_matrix returned from survivehr_predict_event_risks().")
        }

        if (is.null(risks$observed_events)) {
          stop(
            "No observed_events returned from survivehr_predict_event_risks(). ",
            "The Python backend should return observed next-event IDs from outputs['surv']['k']."
          )
        }

        if (!is.matrix(risks$risk_matrix) && !is.data.frame(risks$risk_matrix)) {
          stop(
            "risk_matrix must be a matrix or data.frame. Got class: ",
            paste(class(risks$risk_matrix), collapse = ", ")
          )
        }

        if (nrow(risks$risk_matrix) != length(risks$observed_events)) {
          stop(
            "Risk/observed transition mismatch in batch ", batch_idx, ": ",
            "nrow(risk_matrix) = ", nrow(risks$risk_matrix),
            ", length(observed_events) = ", length(risks$observed_events),
            ". These should match because both are returned from the same Python-built model sequence."
          )
        }

        iec_batch <- survivehr_compute_iec(
          risk_scores = risks$risk_matrix,
          observed_events = risks$observed_events,
          stratify_by_event = stratify_by_event,
          event_vocabulary = risks$event_names
        )

        # Accumulate transition-level values when available
        all_iec_values <- c(all_iec_values, iec_batch$iec_values)
        all_observed_events <- c(all_observed_events, risks$observed_events)
        all_observed_ranks <- c(all_observed_ranks, iec_batch$observed_ranks)
        all_observed_ranks_from_top <- c(
          all_observed_ranks_from_top,
          iec_batch$observed_ranks_from_top
        )
        all_errors <- c(all_errors, iec_batch$errors)

        # Accumulate stratified event-level results
        if (stratify_by_event && !is.null(iec_batch$by_event) && nrow(iec_batch$by_event) > 0) {
          for (idx in seq_len(nrow(iec_batch$by_event))) {
            event_name <- as.character(iec_batch$by_event$event[idx])

            if (!(event_name %in% names(by_event_aggregates))) {
              by_event_aggregates[[event_name]] <- list(
                iec_sum = 0,
                n_obs = 0
              )
            }

            by_event_aggregates[[event_name]]$iec_sum <-
              by_event_aggregates[[event_name]]$iec_sum +
              iec_batch$by_event$mean_iec[idx] * iec_batch$by_event$n_obs[idx]

            by_event_aggregates[[event_name]]$n_obs <-
              by_event_aggregates[[event_name]]$n_obs +
              iec_batch$by_event$n_obs[idx]
          }
        }

        if (!aggregate_only) {
          batch_results[[batch_idx]] <- list(
            batch_patients = batch_patients,
            mean_iec = iec_batch$mean_iec,
            n_valid = iec_batch$n_valid,
            n_total = iec_batch$n_total,
            iec_values = iec_batch$iec_values,
            observed_events = risks$observed_events,
            observed_ranks = iec_batch$observed_ranks,
            observed_ranks_from_top = iec_batch$observed_ranks_from_top,
            by_event = iec_batch$by_event,
            errors = iec_batch$errors
          )
        }
      },
      error = function(e) {
        warning("Batch ", batch_idx, " failed: ", e$message, call. = FALSE)
        all_errors <<- c(
          all_errors,
          paste0("Batch ", batch_idx, " failed: ", e$message)
        )
      }
    )
  }

  # Finalise stratified aggregates
  by_event_df <- NULL

  if (stratify_by_event && length(by_event_aggregates) > 0) {
    by_event_df <- data.frame(
      event = names(by_event_aggregates),
      mean_iec = vapply(
        by_event_aggregates,
        function(x) {
          if (x$n_obs > 0) x$iec_sum / x$n_obs else NA_real_
        },
        numeric(1)
      ),
      n_obs = vapply(
        by_event_aggregates,
        function(x) x$n_obs,
        numeric(1)
      ),
      stringsAsFactors = FALSE
    )

    rownames(by_event_df) <- NULL

    by_event_df <- by_event_df[order(by_event_df$event), , drop = FALSE]
  }

  # Final overall summary
  #
  # Usually all_iec_values will be available. If not, fall back to the weighted
  # event-level mean from by_event_df.
  if (length(all_iec_values) > 0) {
    final_mean_iec <- mean(all_iec_values, na.rm = TRUE)
    final_n_valid <- sum(!is.na(all_iec_values))
  } else if (!is.null(by_event_df) && nrow(by_event_df) > 0) {
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

  # n_total should be the number of model-used observed transitions, not the raw
  # number of rows in the input event table.
  final_n_total <- length(all_observed_events)

  result <- list(
    mean_iec = final_mean_iec,
    n_valid = final_n_valid,
    n_total = final_n_total,
    iec_values = if (aggregate_only) NULL else all_iec_values,
    observed_events = if (aggregate_only) NULL else all_observed_events,
    observed_ranks = if (aggregate_only) NULL else all_observed_ranks,
    observed_ranks_from_top = if (aggregate_only) NULL else all_observed_ranks_from_top,
    by_event = by_event_df,
    by_batch = if (aggregate_only) NULL else batch_results,
    errors = all_errors,
    stratified = stratify_by_event,
    aggregate_only = aggregate_only
  )

  class(result) <- c("survivehr_iec_eval", "list")
  result
}


#' Print Method for IEC Evaluation Results
#'
#' @param x Object of class `"survivehr_iec_eval"`.
#' @param ... Additional arguments ignored.
#'
#' @export
print.survivehr_iec_eval <- function(x, ...) {
  cat("=== SurvivEHR IEC Evaluation ===\n")
  cat(
    "Mean IEC: ",
    sprintf("%.4f", x$mean_iec),
    " (", x$n_valid, " / ", x$n_total, " valid transitions)\n",
    sep = ""
  )

  if (length(x$errors) > 0) {
    cat("Batch/transition errors: ", length(x$errors), "\n", sep = "")
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