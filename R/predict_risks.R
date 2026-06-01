#' Extract Event Risk Scores for Debugging
#'
#' Extracts the full risk score matrix from a pretrained SurvivEHR model.
#' This is a **debugging and evaluation function** that exposes the model's
#' predicted risk for each possible next event at each transition.
#'
#' Warning: For large validation sets, this can consume significant memory
#' (n_transitions × n_events matrix). Use `max_patients` to limit extractions.
#'
#' @param model Fitted pretrain model bundle returned from `survivehr_pretrain()`.
#'
#' @param events Data frame with patient event history. Columns:
#'   patient_id, time_years, event_name, value (optional).
#'
#' @param static Optional data frame with static covariates. Columns:
#'   patient_id, covariate columns.
#'
#' @param max_patients Maximum number of patients to extract (for memory safety).
#'   If NULL (default), all patients are processed.
#'
#' @return List with elements:
#'   \item{risk_matrix}{Matrix of shape (total_transitions, n_events) containing
#'     risk scores. Rows correspond to transitions across all patients.}
#'   \item{patient_ids}{List mapping row to patient_id for traceability.}
#'   \item{event_vocab}{Character vector of event names (column names).}
#'   \item{n_events}{Integer, total number of events in vocabulary.}
#'
#' @details
#' This function is designed for:
#' - Manual IEC calculation on extracted risk scores
#' - Model debugging and inspection
#' - Computing ranking-based metrics
#'
#' For automated evaluation, use `survivehr_evaluate_iec()` instead,
#' which handles batch processing and memory management automatically.
#'
#' @keywords internal
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

  # Enforce max_patients limit for memory safety
  if (!is.null(max_patients)) {
    unique_patients <- unique(events$patient_id)
    if (length(unique_patients) > max_patients) {
      selected_patients <- head(unique_patients, max_patients)
      events <- events[events$patient_id %in% selected_patients, ]
      if (!is.null(static)) {
        static <- static[static$patient_id %in% selected_patients, ]
      }
      warning(
        "Limited to first ", max_patients, " unique patients. ",
        "Set max_patients=NULL to process all patients."
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

  # Combine risk scores from all patients into single matrix
  # Use [[ ]] to access Python dict keys, not $ (which tries attributes)
  risk_scores <- reticulate::py_to_r(py_result[["risk_scores"]])
  
  if (is.null(risk_scores) || length(risk_scores) == 0) {
    stop(
      "extract_pretrain_risk_scores() returned empty risk_scores list. \n",
      "Check Python debug output above for why patients were skipped. \n",
      "This usually means the model's CDF outputs are not in the expected shape.\n",
      "CDFs should be shaped: (n_patients, n_time_grid) for each event type."
    )
  }

  # Properly repeat patient IDs: one ID per row of risk matrix per patient
  patient_ids_py <- reticulate::py_to_r(py_result[["patient_ids"]])
  patient_ids <- unlist(Map(
    function(pid, mat) rep(as.character(pid), nrow(mat)),
    patient_ids_py,
    risk_scores
  ))

  # Extract vocab table returned from Python (now a pandas DataFrame)
  vocab_table <- reticulate::py_to_r(py_result[["event_vocab_table"]])
  vocab_table$event_id <- as.integer(vocab_table$event_id)
  vocab_table <- vocab_table[order(vocab_table$event_id), ]
  
  # Sanity check: vocab must match risk matrix columns
  if (nrow(vocab_table) != ncol(do.call(rbind, risk_scores))) {
    stop(
      "Vocabulary/risk matrix mismatch: vocab_table has ", nrow(vocab_table),
      " rows but risk_matrix has ", ncol(do.call(rbind, risk_scores)), " columns."
    )
  }
  
  # Create named vector: event_name -> event_id (for IEC)
  event_names <- as.character(vocab_table$event)
  event_vocab_for_iec <- setNames(vocab_table$event_id, vocab_table$event)

  return(list(
    risk_scores = risk_scores,
    patient_ids = patient_ids,
    event_vocab = event_vocab_for_iec,
    event_names = event_names,
    event_vocab_table = vocab_table,
    n_events = as.integer(py_result[["n_events"]])
  ))
}

