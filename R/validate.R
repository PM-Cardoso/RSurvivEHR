#' Validate events schema for SurvivEHR
#'
#' @param events data.frame with columns `patient_id`, `event`, `age` (or FastEHR aliases
#'   `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`), optional `value`/`VALUE`.
#' @return Invisibly returns TRUE.
#' @export
#' @examples
#' events <- data.frame(
#'   patient_id = c(1, 1, 2, 2),
#'   event      = c("HYPERTENSION", "STATIN", "T2D", "METFORMIN"),
#'   age        = c(50.0, 50.5, 45.0, 45.3)
#' )
#' survivehr_validate_events(events)
survivehr_validate_events <- function(events) {
  stopifnot(is.data.frame(events))

  has_patient <- any(c("patient_id", "PATIENT_ID") %in% names(events))
  has_event <- any(c("event", "EVENT") %in% names(events))
  has_age <- any(c("age", "DAYS_SINCE_BIRTH") %in% names(events))
  if (!(has_patient && has_event && has_age)) {
    stop(
      "Events must include either (`patient_id`,`event`,`age`) or FastEHR aliases (`PATIENT_ID`,`EVENT`,`DAYS_SINCE_BIRTH`).",
      call. = FALSE
    )
  }

  patient_col <- if ("patient_id" %in% names(events)) "patient_id" else "PATIENT_ID"
  age_col <- if ("age" %in% names(events)) "age" else "DAYS_SINCE_BIRTH"

  age_vec <- suppressWarnings(as.numeric(events[[age_col]]))
  if (anyNA(age_vec)) {
    stop("Age column must be numeric (or coercible to numeric).", call. = FALSE)
  }
  if (any(age_vec < 0)) {
    stop("Age values must be non-negative.", call. = FALSE)
  }

  ord <- order(events[[patient_col]], age_vec)
  ordered <- events[ord, , drop = FALSE]
  split_age <- split(as.numeric(ordered[[age_col]]), ordered[[patient_col]])
  non_monotonic <- vapply(split_age, function(x) any(diff(x) < 0), logical(1))
  if (any(non_monotonic)) {
    bad <- paste(names(non_monotonic)[non_monotonic], collapse = ", ")
    stop("Events must be time-ordered within patient. Non-monotonic IDs: ", bad, call. = FALSE)
  }

  n_patients <- length(unique(events[[patient_col]]))
  n_rows     <- nrow(events)
  message(sprintf("[OK] Events: %d rows, %d patients. Columns present, ages numeric and time-ordered.",
                  n_rows, n_patients))
  invisible(TRUE)
}

#' Validate static covariates schema for SurvivEHR
#'
#' @param static_covariates data.frame with patient id and covariates. Accepts
#'   `patient_id` or `PATIENT_ID`.
#' @return Invisibly returns TRUE.
#' @export
#' @examples
#' static <- data.frame(
#'   patient_id = c(1, 2),
#'   sex = c("M", "F"),
#'   imd = c(3L, 1L)
#' )
#' survivehr_validate_static(static)
survivehr_validate_static <- function(static_covariates) {
  stopifnot(is.data.frame(static_covariates))
  if (!any(c("patient_id", "PATIENT_ID") %in% names(static_covariates))) {
    stop("Static covariates must include `patient_id` or `PATIENT_ID`.", call. = FALSE)
  }

  covar_cols <- setdiff(names(static_covariates), c("patient_id", "PATIENT_ID"))
  if (length(covar_cols) == 0) {
    warning("No static covariate columns found; backend will use an intercept column.", call. = FALSE)
  }

  n_patients <- nrow(static_covariates)
  message(sprintf("[OK] Static covariates: %d patients, %d covariate column(s): %s.",
                  n_patients, length(covar_cols),
                  if (length(covar_cols) > 0) paste(covar_cols, collapse = ", ") else "none"))
  invisible(TRUE)
}

#' Validate fine-tune targets schema for SurvivEHR
#'
#' @param targets data.frame with columns `patient_id`, `target_event`, `target_age`
#'   or FastEHR aliases (`PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`). Optional
#'   `target_value` / `VALUE`.
#' @return Invisibly returns TRUE.
#' @export
#' @examples
#' targets <- data.frame(
#'   patient_id   = c(5L, 6L),
#'   target_event = c("CVD", "DEATH"),
#'   target_age   = c(51.2, 64.1)
#' )
#' survivehr_validate_targets(targets)
survivehr_validate_targets <- function(targets) {
  stopifnot(is.data.frame(targets))

  has_patient <- any(c("patient_id", "PATIENT_ID") %in% names(targets))
  has_event <- any(c("target_event", "TARGET_EVENT", "EVENT") %in% names(targets))
  has_age <- any(c("target_age", "TARGET_AGE", "DAYS_SINCE_BIRTH") %in% names(targets))
  if (!(has_patient && has_event && has_age)) {
    stop(
      "Targets must include (`patient_id`,`target_event`,`target_age`) or FastEHR aliases (`PATIENT_ID`,`EVENT`,`DAYS_SINCE_BIRTH`).",
      call. = FALSE
    )
  }

  age_col <- if ("target_age" %in% names(targets)) {
    "target_age"
  } else if ("TARGET_AGE" %in% names(targets)) {
    "TARGET_AGE"
  } else {
    "DAYS_SINCE_BIRTH"
  }

  age_vec <- suppressWarnings(as.numeric(targets[[age_col]]))
  if (anyNA(age_vec)) {
    stop("Target age column must be numeric (or coercible to numeric).", call. = FALSE)
  }

  n_targets    <- nrow(targets)
  outcome_vals <- unique(targets[[if ("target_event" %in% names(targets)) "target_event"
                                  else if ("TARGET_EVENT" %in% names(targets)) "TARGET_EVENT"
                                  else "EVENT"]])
  message(sprintf("[OK] Targets: %d labelled patients, outcome(s): %s.",
                  n_targets, paste(outcome_vals, collapse = ", ")))
  invisible(TRUE)
}
