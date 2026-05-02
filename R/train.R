#' Train a pretrain SurvivEHR model from R data frames
#'
#' @param events data.frame with columns patient_id, event, age, optional value.
#' @param static_covariates optional data.frame with patient_id + numeric columns.
#' @param config list from `survivehr_config()`.
#' @param event_vocab optional named integer map to keep fixed tokenization.
#' @return A named list (model bundle) with elements `model`, `event_vocab`,
#'   `inv_vocab`, `config`, `time_scale`, `token_policy`, `history`, and
#'   `device` (a string such as `"cpu"` or `"cuda:0"`).
#' @export
#' @examples
#' \dontrun{
#' events <- data.frame(
#'   patient_id = c(1L,1L,1L, 2L,2L,2L),
#'   event  = c("HYPERTENSION","STATIN","T2D", "T2D","METFORMIN","HYPERTENSION"),
#'   age    = c(50, 50.5, 52,  45, 45.3, 47.5)
#' )
#' static <- data.frame(patient_id=c(1L,2L), sex=c("M","F"), imd=c(3L,1L))
#' cfg    <- survivehr_config(block_size=32, n_layer=2, n_head=2, n_embd=64, epochs=1)
#' pt     <- survivehr_pretrain(events, static, cfg)
#' pt$event_vocab  # <PAD>=0, <UNK>=1, HYPERTENSION=2, ...
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

#' Fine-tune SurvivEHR from R data frames
#'
#' @param events data.frame with columns patient_id, event, age, optional value.
#' @param targets data.frame with columns patient_id, target_event, target_age, optional target_value.
#' @param outcomes character vector of outcomes for the fine-tuned head.
#' @param risk_model "competing-risk" or "single-risk".
#' @param static_covariates optional data.frame with patient_id + numeric columns.
#' @param config list from `survivehr_config()`.
#' @param pretrained_model optional model handle from `survivehr_pretrain()`.
#' @param event_vocab optional named integer map to keep fixed tokenization.
#' @return A named list (fine-tuned model bundle) with an additional `device`
#'   field (e.g. `"cpu"` or `"cuda:0"`). Pass to `survivehr_predict()`
#'   or `survivehr_save_model()`.
#' @export
#' @examples
#' \dontrun{
#' # Using events/static/cfg/pt from survivehr_pretrain() example above
#' targets <- data.frame(
#'   patient_id   = 1L,
#'   target_event = "CVD",
#'   target_age   = 54.0
#' )
#' ft <- survivehr_finetune(
#'   events, targets,
#'   outcomes     = "CVD",
#'   risk_model   = "single-risk",
#'   static_covariates = static,
#'   config       = cfg,
#'   pretrained_model  = pt
#' )
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
  survivehr_validate_events(events)
  survivehr_validate_targets(targets)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }

  backend <- .survivehr_backend()
  backend$train_finetune_model(
    events_df = events,
    targets_df = targets,
    outcomes = as.list(as.character(outcomes)),
    risk_model = risk_model,
    static_df = static_covariates,
    config = .to_py_dict(config),
    pretrained_bundle = pretrained_model,
    event_vocab = event_vocab
  )
}

#' Predict next events with SurvivEHR
#'
#' @param model_bundle model object returned by pretrain/fine-tune.
#' @param events data.frame with columns patient_id, event, age, optional value.
#' @param static_covariates optional data.frame with patient_id + numeric columns.
#' @param max_new_tokens number of autoregressive steps.
#' @return `data.frame` with columns `patient_id`, `event`, `age`, `value`.
#' @export
#' @examples
#' \dontrun{
#' preds <- survivehr_predict(ft, events, static)
#' preds
#' }
survivehr_predict <- function(model_bundle,
                              events,
                              static_covariates = NULL,
                              max_new_tokens = 1L) {
  survivehr_validate_events(events)
  if (!is.null(static_covariates)) {
    survivehr_validate_static(static_covariates)
  }
  backend <- .survivehr_backend()
  out <- backend$predict_next_events(
    model_bundle = model_bundle,
    events_df = events,
    static_df = static_covariates,
    max_new_tokens = as.integer(max_new_tokens)
  )
  reticulate::py_to_r(out)
}

#' Save a SurvivEHR model bundle
#'
#' @param model_bundle object returned by training functions.
#' @param path file path ending in `.pt`.
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

#' Load a SurvivEHR model bundle
#'
#' @param path file path created by `survivehr_save_model()`.
#' @return Named list (model bundle) identical in structure to the original
#'   bundle returned by `survivehr_finetune()`.
#' @export
#' @examples
#' \dontrun{
#' ft2 <- survivehr_load_model("my_model.pt")
#' }
survivehr_load_model <- function(path) {
  backend <- .survivehr_backend()
  backend$load_model_bundle(normalizePath(path, mustWork = TRUE))
}
