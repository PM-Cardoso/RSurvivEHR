#' Pre-train the RSurvivEHR backbone transformer
#'
#' Builds the event vocabulary from the supplied event history, then
#' pre-trains the transformer backbone with a competing-risk or single-risk
#' survival head that predicts **when** the next clinical event will occur
#' (over all vocabulary tokens).  The resulting model bundle can be passed
#' directly to `survivehr_finetune()` or saved with `survivehr_save_model()`
#' for later reuse.
#'
#' Pre-training does **not** require a `targets` frame — the next-event age
#' in each patient's raw sequence serves as the supervision signal.
#'
#' @param events A `data.frame` with columns `patient_id`, `event`, `age`
#'   (and optionally `value`) — lowercase is the preferred canonical form;
#'   uppercase aliases `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH` are accepted
#'   for backward compatibility.  Validated with
#'   `survivehr_validate_events()` before being passed to Python.
#' @param static_covariates An optional `data.frame` with `patient_id` and
#'   covariate columns.  Categorical columns are one-hot encoded
#'   automatically; numeric columns pass through unchanged.  Pass `NULL`
#'   (default) to train without static features.
#' @param config A named list from `survivehr_config()` specifying
#'   architecture and training hyperparameters.
#' @param event_vocab An optional named integer vector fixing the token
#'   mapping.  Useful when pre-training multiple models that must share
#'   the same vocabulary.  `NULL` (default) builds the vocabulary from
#'   the supplied events, ordered by descending frequency.
#' @return A named list (model bundle) with elements:
#'   \describe{
#'     \item{`model`}{The trained PyTorch model object.}
#'     \item{`event_vocab`}{Named integer vector mapping event codes to
#'       token IDs (frequency-descending order, most common = smallest ID).}
#'     \item{`inv_vocab`}{Reverse mapping from token IDs to event codes.}
#'     \item{`config`}{The configuration used for training.}
#'     \item{`time_scale`}{The backbone age normalisation divisor stored in
#'       the bundle.  Used to normalise context ages before they enter the
#'       transformer; inherited automatically at fine-tune time.}
#'     \item{`value_standardization`}{Per-event value scaling metadata learned
#'       during pre-training from non-`NA` `value` rows.  For each event with
#'       observed numeric values, the bundle stores event-specific mean/sd used
#'       for internal z-score standardisation.  The same mapping is reused in
#'       fine-tuning and prediction, and value predictions are de-standardised
#'       back to original units before being returned to R.}
#'     \item{`token_policy`}{Token policy flags (`include_unk`,
#'       `include_cls_sep`).}
#'     \item{`history`}{List of per-epoch training losses.}
#'     \item{`training_duration_secs`}{Wall-clock seconds elapsed during
#'       training (a single `numeric` scalar).  Use this to report and
#'       compare training times, e.g.
#'       `cat("Pretrain took", round(pt$training_duration_secs, 1), "s\\n")`.}
#'     \item{`device`}{String identifying the compute device used
#'       (e.g. `"cpu"` or `"cuda:0"`).}
#'   }
#' @export
#' @examples
#' \dontrun{
#' # ---- Pre-training on the 10-patient population from the Getting Started
#' #      vignette.  Patients 1-3 have CVD in their history so the vocabulary
#' #      includes CVD — required for CVD fine-tuning later.
#' events_pop <- data.frame(
#'   patient_id = c(rep(1,4), rep(2,6), rep(3,6), rep(4,4), rep(5,6),
#'                  rep(6,4), rep(7,4), rep(8,4), rep(9,6), rep(10,4)),
#'   event = c(
#'     "HYPERTENSION","STATIN","BP_CHECK","CVD",
#'     "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HYPERTENSION","CVD",
#'     "HYPERTENSION","BP_CHECK","STATIN","T2D","BP_CHECK","CVD",
#'     "HYPERTENSION","STATIN","T2D","METFORMIN",
#'     "HYPERTENSION","BP_CHECK","T2D","HBA1C","METFORMIN","STATIN",
#'     "HYPERTENSION","AMLODIPINE","BP_CHECK","STATIN",
#'     "STATIN","T2D","HBA1C","METFORMIN",
#'     "HYPERTENSION","BP_CHECK","STATIN","T2D",
#'     "HYPERTENSION","BP_CHECK","T2D","METFORMIN","HBA1C","STATIN",
#'     "STATIN","BP_CHECK","HYPERTENSION","T2D"
#'   ),
#'   age = c(
#'     55.0, 55.5, 56.2, 58.0,
#'     44.0, 45.5, 48.0, 48.3, 50.5, 52.0,
#'     58.0, 59.5, 62.0, 63.5, 64.0, 65.5,
#'     45.0, 45.5, 47.0, 48.3,
#'     48.0, 49.0, 51.0, 52.0, 53.0, 54.5,
#'     60.0, 61.0, 62.3, 63.0,
#'     40.0, 42.0, 43.5, 44.8,
#'     58.0, 59.2, 60.0, 62.0,
#'     46.0, 47.5, 50.0, 52.0, 52.5, 54.0,
#'     44.0, 46.5, 47.0, 48.5
#'   ),
#'   value = c(
#'     NA,  NA,  148, NA,
#'     NA,  145, NA,  NA,  NA,  NA,
#'     NA,  158, NA,  NA,  162, NA,
#'     NA,  NA,  NA,  NA,
#'     NA,  152, NA,  68,  NA,  NA,
#'     NA,  NA,  155, NA,
#'     NA,  NA,  74,  NA,
#'     NA,  145, NA,  NA,
#'     NA,  138, NA,  NA,  71,  NA,
#'     NA,  138, NA,  NA
#'   )
#' )
#' static_pop <- data.frame(
#'   patient_id    = 1:10,
#'   sex           = c("M","F","M","F","M","F","M","F","M","F"),
#'   ethnicity     = c("White","Asian","White","Black","White",
#'                     "Asian","White","White","Black","White"),
#'   imd           = c(3L, 1L, 5L, 2L, 4L, 3L, 1L, 5L, 2L, 4L),
#'   year_of_birth = c(1960L,1970L,1952L,1975L,1963L,1958L,1978L,1960L,1968L,1975L)
#' )
#' # Year-by-year backbone normalisation (ages in years → model sees plain year values)
#' cfg <- survivehr_config(
#'   block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
#'   epochs = 10, batch_size = 4, time_scale = 1.0
#' )
#' pt <- survivehr_pretrain(events_pop, static_pop, cfg)
#' # Vocabulary is frequency-ordered: most-common events get the smallest IDs
#' # HYPERTENSION=10, BP_CHECK=9, STATIN=9, T2D=8, METFORMIN=5, CVD=3, HBA1C=3, AMLODIPINE=1
#' pt$event_vocab
#' }
survivehr_pretrain <- function(events,
                               static_covariates = NULL,
                               config = survivehr_config(),
                               event_vocab = NULL) {
  survivehr_validate_events(events)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }
  backend <- .survivehr_backend()
  backend$train_pretrain_model(
    events_df = events,
    static_df = static_covariates,
    config = .to_py_dict(config),
    event_vocab = event_vocab
  )
}

#' Fine-tune the RSurvivEHR backbone on labelled outcomes
#'
#' Attaches a fresh outcome-level survival head on top of the pre-trained
#' backbone and fine-tunes the combined model on a labelled cohort.  Two
#' head types are supported:
#' \itemize{
#'   \item **`"competing-risk"`** — models **two or more** outcomes that
#'     compete (the first to occur prevents the others from being observed).
#'     Requires `length(outcomes) >= 2`; an error is raised otherwise.
#'     Patients whose `target_event` is not one of the `outcomes` codes are
#'     right-censored.
#'   \item **`"single-risk"`** — models **exactly one** endpoint.
#'     Requires `length(outcomes) == 1`; an error is raised otherwise.
#'     Patients whose `target_event` does not match that code are
#'     right-censored.
#' }
#'
#' In both cases, right-censored patients contribute `log(1 - CDF(target_age))`
#' to the DeSurv loss; patients with an observed outcome contribute the
#' event-density term for their respective risk.
#'
#' To avoid data leakage, the `events` frame passed here must:
#' 1. Have all outcome event codes **removed** (they are supplied only via
#'    `targets`).
#' 2. Contain only events that occurred **before** each patient's
#'    `target_age` (events after the outcome would not be available at
#'    prediction time).
#'
#' @param events A `data.frame` of context events (outcome codes and
#'   post-outcome rows removed) with columns `patient_id`, `event`, `age`,
#'   and optionally `value`.
#' @param targets A `data.frame` with columns `patient_id`, `target_event`,
#'   and `target_age` labelling the observed outcome (cases) or the last
#'   recorded non-outcome event (censored patients).  The censoring
#'   convention is: if a patient's `target_event` code is **not** one of
#'   the strings in `outcomes`, they are treated as right-censored and
#'   contribute `log(1 - CDF(target_age))` to the loss; if it **is** in
#'   `outcomes`, they are treated as having experienced that event and
#'   contribute the event-density term.  Build with
#'   `survivehr_validate_targets()`.
#' @param outcomes Character vector of outcome event codes that the
#'   fine-tuned head predicts.  Must match the codes used in `targets`.
#' @param risk_model `"competing-risk"` (default) or `"single-risk"`.
#'   Controls the architecture of the outcome-level head — independent of
#'   the `surv_layer` used during pre-training.
#' @param static_covariates An optional `data.frame` with the **same
#'   columns** as those used at pre-training time.  Pass `NULL` to omit.
#' @param config A named list from `survivehr_config()`.  `time_scale` is
#'   inherited automatically from the pretrained bundle — no need to set it
#'   here.  Use `outcome_horizon` to set the ODE prediction window length
#'   (in the same units as `age`) independently of `time_scale`.  For
#'   example, `outcome_horizon = 5` gives a 5-year risk window regardless
#'   of the backbone normalisation scale.
#' @param pretrained_model Model bundle returned by `survivehr_pretrain()`
#'   or `survivehr_load_model()`.  When supplied, the vocabulary, weights,
#'   and `time_scale` are inherited from the bundle.
#' @param event_vocab An optional named integer vector overriding the
#'   vocabulary.  Rarely needed; prefer supplying `pretrained_model`.
#' @return A named list (fine-tuned model bundle) with the same structure
#'   as the pre-trained bundle plus fine-tune-specific fields, including
#'   `training_duration_secs` (wall-clock seconds for this fine-tune run) and
#'   inherited `value_standardization` metadata from the pretrained bundle.
#'   Pass to `survivehr_predict()` or `survivehr_save_model()`.
#' @export
#' @examples
#' \dontrun{
#' # Uses events_pop / static_pop / cfg / pt from survivehr_pretrain() example.
#' ft_static <- static_pop[static_pop$patient_id %in% 1:6, ]
#'
#' # ---- Competing-risk: CVD vs T2D (patients 1-6) ---------------------------
#' targets_cr <- data.frame(
#'   patient_id   = c(1L,    2L,    3L,    4L,    5L,    6L),
#'   target_event = c("CVD", "T2D", "T2D", "T2D", "T2D", "STATIN"),
#'   target_age   = c(58.0,  48.0,  63.5,  47.0,  51.0,  63.0)
#' )
#' survivehr_validate_targets(targets_cr)
#'
#' # Remove both outcomes; keep only events before target_age
#' ft_events_cr <- events_pop[events_pop$patient_id %in% 1:6 &
#'                               !events_pop$event %in% c("CVD","T2D"), ]
#' ft_events_cr <- merge(ft_events_cr,
#'                       targets_cr[, c("patient_id","target_age")],
#'                       by = "patient_id")
#' ft_events_cr <- ft_events_cr[ft_events_cr$age < ft_events_cr$target_age,
#'                               c("patient_id","event","age","value")]
#'
#' cfg_cr <- survivehr_config(
#'   block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
#'   epochs = 10, batch_size = 4, surv_layer = "competing-risk",
#'   outcome_horizon = 5.0  # 5-year prediction window; time_scale inherited from pt
#' )
#' ft_cr <- survivehr_finetune(
#'   events = ft_events_cr, targets = targets_cr,
#'   outcomes = c("CVD","T2D"), risk_model = "competing-risk",
#'   static_covariates = ft_static, config = cfg_cr, pretrained_model = pt
#' )
#' cat("CR loss:", unlist(ft_cr$history), "\n")
#'
#' # ---- Single-risk: CVD only (patients 1-6) --------------------------------
#' targets_sr <- data.frame(
#'   patient_id   = c(1L,    2L,    3L,    4L,          5L,       6L),
#'   target_event = c("CVD", "CVD", "CVD", "METFORMIN", "STATIN", "STATIN"),
#'   target_age   = c(58.0,  52.0,  65.5,  48.3,         54.5,     63.0)
#' )
#' survivehr_validate_targets(targets_sr)
#'
#' # Remove CVD only; keep only events before target_age
#' ft_events_sr <- events_pop[events_pop$patient_id %in% 1:6 &
#'                               events_pop$event != "CVD", ]
#' ft_events_sr <- merge(ft_events_sr,
#'                       targets_sr[, c("patient_id","target_age")],
#'                       by = "patient_id")
#' ft_events_sr <- ft_events_sr[ft_events_sr$age < ft_events_sr$target_age,
#'                               c("patient_id","event","age","value")]
#'
#' cfg_sr <- survivehr_config(
#'   block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
#'   epochs = 10, batch_size = 4, surv_layer = "competing-risk"
#' )
#' ft_sr <- survivehr_finetune(
#'   events = ft_events_sr, targets = targets_sr,
#'   outcomes = "CVD", risk_model = "single-risk",
#'   static_covariates = ft_static, config = cfg_sr, pretrained_model = pt
#' )
#' cat("SR loss:", unlist(ft_sr$history), "\n")
#' }
survivehr_finetune <- function(events,
                               targets,
                               outcomes,
                               risk_model = c("competing-risk", "single-risk"),
                               static_covariates = NULL,
                               config = survivehr_config(),
                               pretrained_model = NULL,
                               event_vocab = NULL) {
  risk_model <- match.arg(risk_model)
  outcomes   <- as.character(outcomes)
  if (risk_model == "competing-risk" && length(outcomes) < 2L) {
    stop(
      "'competing-risk' requires at least 2 outcome codes (got ",
      length(outcomes), ": ", paste(outcomes, collapse = ", "), "). ",
      "Use risk_model = \"single-risk\" for a single endpoint.",
      call. = FALSE
    )
  }
  if (risk_model == "single-risk" && length(outcomes) != 1L) {
    stop(
      "'single-risk' requires exactly 1 outcome code (got ",
      length(outcomes), ": ", paste(outcomes, collapse = ", "), "). ",
      "Use risk_model = \"competing-risk\" for multiple endpoints.",
      call. = FALSE
    )
  }
  survivehr_validate_events(events)
  survivehr_validate_targets(targets)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }

  backend <- .survivehr_backend()
  backend$train_finetune_model(
    events_df = events,
    targets_df = targets,
    outcomes = as.list(outcomes),
    risk_model = risk_model,
    static_df = static_covariates,
    config = .to_py_dict(config),
    pretrained_bundle = pretrained_model,
    event_vocab = event_vocab
  )
}

#' Predict cumulative incidence with a fine-tuned RSurvivEHR model
#'
#' Runs forward inference on new event sequences using a fine-tuned model
#' bundle.  The prediction window length and age normalisation divisor
#' (`time_scale`) are read automatically from the bundle — no need to
#' supply them at inference time.
#'
#' The `events` frame should have the outcome event codes removed (same
#' as the context filtering applied at fine-tune time) to ensure
#' leakage-free predictions.
#'
#' @param model_bundle A model bundle returned by `survivehr_finetune()` or
#'   `survivehr_load_model()`.
#' @param events data.frame with columns `patient_id`, `event`, `age`,
#'   optional `value`.  Should **not** contain the outcome event (leakage-free).
#' @param static_covariates optional data.frame with `patient_id` and the
#'   same covariate columns used at training time.
#' @param max_new_tokens number of autoregressive steps (pretrain models only;
#'   ignored for fine-tuned models).
#' @param eval_times An optional numeric vector of time points (in the same
#'   units as `age`) at which to read the cumulative-incidence CDF for
#'   fine-tuned models.  Each value must be in `(0, outcome_horizon]` — the
#'   ODE prediction window stored in the fine-tuned bundle.  For example,
#'   with `outcome_horizon = 5.0` (years) use `eval_times = c(1, 2, 3, 5)`
#'   to obtain 1-, 2-, 3- and 5-year risks.
#'   When `NULL` (default) only `_cdf_last` (risk at the full horizon) and
#'   `_auc` (average risk) are returned, preserving backward compatibility.
#'   Ignored for pretrain models.
#' @return For a **fine-tuned** model, a `data.frame` with columns:
#'   \describe{
#'     \item{`patient_id`}{Patient identifier.}
#'     \item{`{outcome}_cdf_last`}{Cumulative incidence at the **end** of the
#'       prediction window (`t = outcome_horizon`).}
#'     \item{`{outcome}_auc`}{Area under the CDF integrated from 0 to
#'       `time_scale`.  Interpretable as average risk over the window.}
#'     \item{`{outcome}_cdf_t{X}`}{*(Only when `eval_times` is supplied.)*
#'       Cumulative incidence at time `X` (same units as `age`).  One column
#'       per requested time point, e.g. `CVD_cdf_t1`, `CVD_cdf_t2.5`.}
#'   }
#'   For a **pretrain** model, a `data.frame` with one row per generated step
#'   per patient:
#'   \describe{
#'     \item{`patient_id`}{Patient identifier.}
#'     \item{`step`}{Generation step (1 = next event, 2 = event after that, …).}
#'     \item{`generated_token`}{Vocabulary token ID of the generated event.}
#'     \item{`generated_event`}{Decoded event name (e.g. `"HYPERTENSION"`).}
#'     \item{`generated_age`}{Predicted age of the generated event in the same
#'       units as `age` (de-normalised by `time_scale`).}
#'     \item{`generated_value`}{Predicted numeric value for that event (e.g.
#'       a lab result) in the original input units (automatically
#'       de-standardised per event); `NaN` for non-measurement events.}
#'   }
#' @export
#' @examples
#' \dontrun{
#' # ---- Prediction using ft / events_pop / static_pop from examples above -----
#' # Remove the outcome event from prediction context (leakage-free)
#' pred_events <- events_pop[events_pop$event != "CVD", ]
#'
#' preds <- survivehr_predict(ft, pred_events, static_pop)
#' print(preds)
#' # Columns: patient_id, CVD_cdf_last, CVD_auc
#' #
#' # CVD_cdf_last : probability of CVD within outcome_horizon (5 years when
#' #                outcome_horizon = 5.0); stored in the bundle automatically
#' # CVD_auc      : average cumulative CVD risk over that window;
#' #                higher = greater overall risk
#' }
survivehr_predict <- function(model_bundle,
                              events,
                              static_covariates = NULL,
                              max_new_tokens = 1L,
                              eval_times = NULL) {
  survivehr_validate_events(events)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }
  backend <- .survivehr_backend()
  out <- backend$predict_next_events(
    model_bundle = model_bundle,
    events_df = events,
    static_df = static_covariates,
    max_new_tokens = as.integer(max_new_tokens),
    eval_times = if (is.null(eval_times)) NULL else as.list(as.numeric(eval_times))
  )
  reticulate::py_to_r(out)
}

#' Predict the numeric value of a named event (pretrain or fine-tuned models)
#'
#' Queries the backbone's Gaussian value regression head to estimate the
#' numeric measurement (e.g. blood pressure, HbA1c) that would be recorded
#' alongside a specific clinical event, given each patient's history in
#' `events`.
#'
#' The value head is trained during pre-training on events that carried
#' non-`NA` `value` entries.  Values are standardised internally per event
#' (event-specific z-score using pre-training mean/sd) and predictions are
#' automatically transformed back to original units before returning to R.
#' For events that never appeared with a value
#' (e.g. `"CVD"`, which is a discrete diagnosis), the function returns `NaN`
#' for both the mean and standard deviation.  The head is preserved in
#' fine-tuned bundles because fine-tuning only replaces the outcome survival
#' head, not the backbone.
#'
#' @param model_bundle A model bundle returned by `survivehr_pretrain()`,
#'   `survivehr_finetune()`, or `survivehr_load_model()`.
#' @param events A `data.frame` with columns `patient_id`, `event`, `age`,
#'   optional `value`.  The `outcome_event` code should **not** appear in
#'   this frame (same leakage-free filtering as for fine-tuning and
#'   prediction).
#' @param outcome_event Character scalar.  The event code whose value should
#'   be predicted (e.g. `"BP_CHECK"` for blood pressure).  Must exist in the
#'   model vocabulary built at pre-training time.
#' @param static_covariates An optional `data.frame` with `patient_id` and
#'   the same covariate columns used at training time.  Pass `NULL` to omit.
#' @return A `data.frame` with one row per patient and columns:
#'   \describe{
#'     \item{`patient_id`}{Patient identifier.}
#'     \item{`outcome_event`}{The event code that was queried.}
#'     \item{`predicted_value_mean`}{Predicted mean of the Gaussian
#'       distribution for the event's numeric value.  `NaN` if the event
#'       never appeared with a value at pre-training time.}
#'     \item{`predicted_value_sd`}{Predicted standard deviation.  `NaN` for
#'       the same reason as above.}
#'   }
#' @export
#' @examples
#' \dontrun{
#' # Using pt_model and events_pop / static_pop from survivehr_pretrain() example.
#' # Predict the expected blood-pressure reading at the next BP_CHECK.
#' # (Remove BP_CHECK from context first — same leakage-free logic.)
#' ctx <- events_pop[events_pop$event != "BP_CHECK", ]
#' bp_preds <- survivehr_predict_value(pt_model, ctx, "BP_CHECK", static_pop)
#' print(bp_preds)
#' # Columns: patient_id, outcome_event, predicted_value_mean, predicted_value_sd
#' }
survivehr_predict_value <- function(model_bundle,
                                    events,
                                    outcome_event,
                                    static_covariates = NULL) {
  survivehr_validate_events(events)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }
  backend <- .survivehr_backend()
  out <- backend$predict_outcome_value(
    model_bundle  = model_bundle,
    events_df     = events,
    outcome_event = as.character(outcome_event),
    static_df     = static_covariates
  )
  reticulate::py_to_r(out)
}

#' Save an RSurvivEHR model bundle to disk
#'
#' Serialises a model bundle (returned by `survivehr_pretrain()` or
#' `survivehr_finetune()`) to a `.pt` file.  The bundle includes the
#' model weights, vocabulary, static column schema, `time_scale`,
#' `value_standardization`, token policy, and training history.
#' Reload with `survivehr_load_model()`.
#'
#' @param model_bundle A model bundle returned by a training function.
#' @param path File path for the output file.  Should end in `.pt`.
#' @return Invisibly returns `path`.
#' @export
#' @examples
#' \dontrun{
#' tmp <- tempfile(fileext = ".pt")
#' survivehr_save_model(ft, tmp)
#' ft2 <- survivehr_load_model(tmp)
#' unlink(tmp)
#' }
survivehr_save_model <- function(model_bundle, path) {
  backend <- .survivehr_backend()
  backend$save_model_bundle(model_bundle, normalizePath(path, mustWork = FALSE))
  invisible(path)
}

#' Load an RSurvivEHR model bundle from disk
#'
#' Restores a model bundle previously saved with `survivehr_save_model()`.
#' The returned object is identical in structure to the original bundle and
#' can be passed directly to `survivehr_predict()` or used as
#' `pretrained_model` in a further `survivehr_finetune()` call.
#'
#' @param path File path to a `.pt` bundle created by
#'   `survivehr_save_model()`.
#' @return A named list (model bundle) with elements `model`,
#'   `event_vocab`, `inv_vocab`, `config`, `time_scale`,
#'   `value_standardization`, `token_policy`, `history`, and `device`.
#' @export
#' @examples
#' \dontrun{
#' ft2 <- survivehr_load_model("my_model.pt")
#' }
survivehr_load_model <- function(path) {
  backend <- .survivehr_backend()
  backend$load_model_bundle(normalizePath(path, mustWork = TRUE))
}
