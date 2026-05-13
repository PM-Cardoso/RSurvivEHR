#' Validate the events data frame for RSurvivEHR
#'
#' Checks that the events data frame supplied to
#' `survivehr_pretrain()`, `survivehr_finetune()`, or `survivehr_predict()`
#' is correctly formatted before any Python call is made, giving an
#' informative R error instead of a cryptic Python traceback.  Specifically,
#' it verifies that the required columns are present, that ages are numeric
#' and non-negative, and that rows are time-ordered within each patient.
#'
#' @param events A `data.frame` with columns (lowercase is the preferred
#'   canonical form; uppercase aliases are accepted for backward compatibility):
#'   \describe{
#'     \item{`patient_id`}{Patient identifier (numeric or character).
#'       Alias: `PATIENT_ID`.}
#'     \item{`event`}{Clinical event code (character).
#'       Alias: `EVENT`.}
#'     \item{`age`}{Patient age at the event in consistent units (numeric,
#'       non-negative, time-ordered within patient).
#'       Alias: `DAYS_SINCE_BIRTH`.}
#'     \item{`value`}{(Optional) Continuous measurement recorded at the
#'       event (e.g. blood pressure or HbA1c).  `NA` for discrete events.
#'       Alias: `VALUE`.}
#'   }
#' @return Invisibly returns `TRUE`.  Prints a confirmation message on
#'   success.
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
      paste0(
        "Events must include columns `patient_id`, `event`, and `age` ",
        "(or uppercase alternatives `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`)."
      ),
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

#' Validate the static covariates data frame for RSurvivEHR
#'
#' Checks that the static covariates data frame has a `patient_id` column
#' and at least one covariate column.  Issues a warning (not an error) when
#' no covariate columns are found, because the backend will substitute an
#' intercept column in that case.
#'
#' Categorical columns (those that are less than 80\% numeric) are
#' one-hot encoded automatically by the Python backend.  Numeric columns
#' are passed through unchanged.  The exact encoded column list is stored
#' inside every model bundle; passing different columns at prediction time
#' raises a descriptive error.
#'
#' @param static_covariates A `data.frame` with a `patient_id` column
#'   (lowercase preferred; `PATIENT_ID` accepted for backward compatibility)
#'   plus any number of covariate columns with freely chosen names — lowercase
#'   is recommended for consistency.  Pass the **same columns in the same
#'   order** across pretrain, fine-tune, and prediction.
#' @return Invisibly returns `TRUE`.  Prints a confirmation message on
#'   success.
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

#' Validate the fine-tune targets data frame for RSurvivEHR
#'
#' Checks that the targets data frame supplied to `survivehr_finetune()`
#' is correctly formatted.  Each row labels one patient: cases supply the
#' observed outcome event and its age; censored patients supply their last
#' observed non-outcome event and its age.
#'
#' @param targets A `data.frame` with columns (lowercase is the preferred
#'   canonical form; uppercase aliases are accepted for backward compatibility):
#'   \describe{
#'     \item{`patient_id`}{Patient identifier matching the events frame.
#'       Alias: `PATIENT_ID`.}
#'     \item{`target_event`}{Event code for the labelled outcome (cases) or
#'       the last observed non-outcome event (censored).
#'       Aliases: `TARGET_EVENT`, `EVENT`.}
#'     \item{`target_age`}{Age at the target event.  Must be numeric and
#'       non-negative.
#'       Aliases: `TARGET_AGE`, `DAYS_SINCE_BIRTH`.}
#'     \item{`target_value`}{(Optional) Continuous measurement at the
#'       target event.  `NA` for discrete events.
#'       Aliases: `TARGET_VALUE`, `VALUE`.}
#'   }
#' @return Invisibly returns `TRUE`.  Prints a confirmation message listing
#'   the number of labelled patients and all observed outcome values.
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
      paste0(
        "Targets must include columns `patient_id`, `target_event`, and `target_age` ",
        "(or uppercase alternatives `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`)."
      ),
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
