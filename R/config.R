#' Build default SurvivEHR configuration
#'
#' @param block_size Sequence length after padding/truncation.
#' @param n_layer Number of transformer blocks.
#' @param n_head Number of attention heads.
#' @param n_embd Hidden embedding size.
#' @param dropout Dropout probability.
#' @param learning_rate Learning rate.
#' @param epochs Number of epochs.
#' @param batch_size Batch size.
#' @param surv_layer Pretrain survival head: "competing-risk" or "single-risk".
#' @param surv_weight Survival loss weight.
#' @param value_weight Value regression loss weight.
#' @param device "auto", "cpu", or "cuda".
#' @param include_unk Whether to reserve and use `<UNK>` for unseen events.
#' @param include_cls_sep Whether to add `<CLS>` and `<SEP>` around each sequence.
#' @param time_scale Divisor applied to every raw age before it enters the model.
#'   Use `1.0` (default) when ages are in years.
#'   Use `1825.0` when ages are in days (`DAYS_SINCE_BIRTH`, matching FastEHR default).
#'   Must be consistent between pretrain, fine-tune, and prediction.
#'
#' @return Named list used by training functions.
#' @export
#' @examples
#' # Minimal config for a quick CPU run
#' cfg <- survivehr_config(
#'   block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
#'   epochs = 1, batch_size = 4
#' )
#' cfg$surv_layer   # "competing-risk"
#' cfg$time_scale   # 1.0
survivehr_config <- function(
  block_size = 128,
  n_layer = 4,
  n_head = 4,
  n_embd = 256,
  dropout = 0,
  learning_rate = 3e-4,
  epochs = 1,
  batch_size = 16,
  surv_layer = "competing-risk",
  surv_weight = 1,
  value_weight = 0,
  device = "auto",
  include_unk = TRUE,
  include_cls_sep = FALSE,
  time_scale = 1.0
) {
  list(
    block_size = as.integer(block_size),
    n_layer = as.integer(n_layer),
    n_head = as.integer(n_head),
    n_embd = as.integer(n_embd),
    dropout = as.numeric(dropout),
    learning_rate = as.numeric(learning_rate),
    epochs = as.integer(epochs),
    batch_size = as.integer(batch_size),
    surv_layer = as.character(surv_layer),
    surv_weight = as.numeric(surv_weight),
    value_weight = as.numeric(value_weight),
    device = as.character(device),
    include_unk = as.logical(include_unk),
    include_cls_sep = as.logical(include_cls_sep),
    time_scale = as.numeric(time_scale)
  )
}
