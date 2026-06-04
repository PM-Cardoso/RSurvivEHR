#' Extract Event Risk Scores for IEC Evaluation
#'
#' Extracts the full risk score matrix from a pretrained SurvivEHR model.
#' This function exposes the model's predicted risk for each possible next event
#' at each observed transition, together with the observed next-event IDs used
#' by the Python backend.
#'
#' @param model Fitted pretrain model bundle returned from `survivehr_pretrain()`
#'   or `survivehr_load_model()`.
#'
#' @param events Data frame with patient event history. Required columns:
#'   `patient_id`, `event`, `age`; optional column: `value`.
#'
#' @param static Optional data frame with static covariates. Must contain
#'   `patient_id` and the same static covariate columns used during training.
#'
#' @param max_patients Optional integer. Maximum number of patients to process.
#'   Useful for debugging or memory-limited runs. If `NULL`, all patients are
#'   processed.
#'
#' @return List with elements:
#'   \item{risk_matrix}{Numeric matrix of shape
#'     `total_transitions x n_events`.}
#'   \item{risk_scores}{List of per-patient risk matrices.}
#'   \item{observed_events}{Integer vector of observed next-event IDs aligned
#'     to rows of `risk_matrix`.}
#'   \item{patient_ids}{Character vector mapping each transition row to a
#'     patient ID.}
#'   \item{event_vocab}{Named integer vector mapping event names to
#'     1-indexed risk-matrix column IDs.}
#'   \item{event_names}{Character vector of event names ordered by risk-matrix
#'     columns.}
#'   \item{event_vocab_table}{Data frame with columns `event` and `event_id`.}
#'   \item{n_events}{Integer. Number of predictable event types.}
#'
#' @details
#' This function is mainly used internally by `survivehr_evaluate_iec()`, but is
#' exported because it is useful for debugging and manual IEC calculations.
#'
#' Observed events are returned from the Python backend from the same model-built
#' sequence as the risk scores. This avoids dimension mismatches that can occur
#' if observed transitions are reconstructed independently in R.
#'
#' @export
survivehr_predict_event_risks <- function(
  model,
  events,
  static = NULL,
  max_patients = NULL
) {
  # Validate model
  if (!is.list(model) || is.null(model$model) || is.null(model$event_vocab)) {
    stop("model must be a RSurvivEHR model bundle with model and event_vocab")
  }

  # Validate input event/static tables
  survivehr_validate_events(events)

  if (!is.null(static)) {
    survivehr_validate_static(static)
  }

  # Enforce max_patients limit for memory safety
  if (!is.null(max_patients)) {
    unique_patients <- unique(events$patient_id)

    if (length(unique_patients) > max_patients) {
      selected_patients <- head(unique_patients, max_patients)

      events <- events[events$patient_id %in% selected_patients, , drop = FALSE]

      if (!is.null(static)) {
        static <- static[static$patient_id %in% selected_patients, , drop = FALSE]
      }

      warning(
        "Limited to first ", max_patients, " unique patients. ",
        "Set max_patients = NULL to process all patients.",
        call. = FALSE
      )
    }
  }

  # Call Python backend
  backend <- .survivehr_backend()

  py_result <- backend$extract_pretrain_risk_scores(
    model_bundle = model,
    events_df = events,
    static_df = static
  )

  # Convert Python outputs to R
  risk_scores <- reticulate::py_to_r(py_result[["risk_scores"]])
  observed_events <- reticulate::py_to_r(py_result[["observed_events"]])
  patient_ids_py <- reticulate::py_to_r(py_result[["patient_ids"]])
  vocab_table <- reticulate::py_to_r(py_result[["event_vocab_table"]])
  n_events <- as.integer(reticulate::py_to_r(py_result[["n_events"]]))

  # Check risk scores
  if (is.null(risk_scores) || length(risk_scores) == 0) {
    stop(
      "extract_pretrain_risk_scores() returned an empty risk_scores list. ",
      "This usually means no valid patient transitions were available, or the ",
      "model CDF outputs were not in the expected transition-indexed shape.",
      call. = FALSE
    )
  }

  # Ensure each patient risk object is a numeric matrix
  risk_scores <- lapply(risk_scores, function(x) {
    x <- as.matrix(x)
    storage.mode(x) <- "numeric"
    x
  })

  # Combine per-patient matrices into one transition-level matrix
  risk_matrix <- do.call(rbind, risk_scores)

  if (is.null(risk_matrix) || nrow(risk_matrix) == 0) {
    stop("No risk-score rows were returned after combining patient matrices.", call. = FALSE)
  }

  # Observed events are returned from Python from the same model-built sequence
  # as the risk scores, so they should align exactly.
  observed_events <- unlist(lapply(observed_events, as.integer), use.names = FALSE)

  if (length(observed_events) == 0) {
    stop(
      "extract_pretrain_risk_scores() returned no observed_events. ",
      "The Python backend should return observed next-event IDs from outputs['surv']['k'].",
      call. = FALSE
    )
  }

  if (nrow(risk_matrix) != length(observed_events)) {
    stop(
      "Risk/observed transition mismatch: nrow(risk_matrix) = ",
      nrow(risk_matrix), ", length(observed_events) = ",
      length(observed_events), ". This should not happen if both are returned ",
      "from the same Python-built model sequence.",
      call. = FALSE
    )
  }

  # Properly repeat patient IDs: one ID per transition row per patient
  patient_ids <- unlist(Map(
    function(pid, mat) rep(as.character(pid), nrow(mat)),
    patient_ids_py,
    risk_scores
  ), use.names = FALSE)

  if (length(patient_ids) != nrow(risk_matrix)) {
    stop(
      "Patient ID / risk matrix mismatch: length(patient_ids) = ",
      length(patient_ids), ", nrow(risk_matrix) = ", nrow(risk_matrix),
      call. = FALSE
    )
  }

  # Vocabulary table returned from Python as a pandas DataFrame
  if (is.null(vocab_table) || nrow(vocab_table) == 0) {
    stop("event_vocab_table returned from Python is empty.", call. = FALSE)
  }

  vocab_table$event <- as.character(vocab_table$event)
  vocab_table$event_id <- as.integer(vocab_table$event_id)
  vocab_table <- vocab_table[order(vocab_table$event_id), , drop = FALSE]

  # Sanity check: vocab must match risk matrix columns
  if (nrow(vocab_table) != ncol(risk_matrix)) {
    stop(
      "Vocabulary/risk matrix mismatch: vocab_table has ", nrow(vocab_table),
      " rows but risk_matrix has ", ncol(risk_matrix), " columns.",
      call. = FALSE
    )
  }

  if (!identical(vocab_table$event_id, seq_len(nrow(vocab_table)))) {
    stop(
      "event_vocab_table$event_id should be sequential 1-indexed IDs matching ",
      "risk_matrix columns.",
      call. = FALSE
    )
  }

  # Named vector: event_name -> 1-indexed risk-matrix column ID
  event_names <- vocab_table$event
  event_vocab_for_iec <- setNames(vocab_table$event_id, vocab_table$event)

  cat("Risk score extraction complete:\n")
  cat("  Risk matrices:   ", length(risk_scores), " patients\n")
  cat("  Event vocabulary:", length(event_names), " events\n")
  cat("  ncol(risk_matrix):", ncol(risk_matrix), " columns\n")
  cat("  Event names:     ", paste(event_names, collapse = ", "), "\n")
  cat("  Total transitions:", nrow(risk_matrix), "\n")
  cat("  Risk matrix shape:", nrow(risk_matrix), " x ", ncol(risk_matrix), "\n\n")

  return(list(
    risk_matrix = risk_matrix,
    risk_scores = risk_scores,
    observed_events = observed_events,
    patient_ids = patient_ids,
    event_vocab = event_vocab_for_iec,
    event_names = event_names,
    event_vocab_table = vocab_table,
    n_events = n_events
  ))
}