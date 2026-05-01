#' Set up the SurvivEHR Python environment
#'
#' Creates a dedicated Python virtual environment named `"survivehrR"` and
#' installs all required packages.  Only needs to be called **once** after
#' installing the R package.  Subsequent calls skip reinstallation unless
#' `force = TRUE`.
#'
#' Works on macOS, Windows, and Linux.  An internet connection is required on
#' first run so that pip can download the packages.
#'
#' @param envname Name of the virtualenv to create/use.  Default `"survivehrR"`.
#' @param force   If `TRUE`, reinstall all packages even when the environment
#'   already exists.
#' @param python  Path to the Python executable.  `NULL` (default) lets
#'   reticulate choose Python 3.9+ automatically.
#'
#' @return Invisibly returns the path to the Python executable in the
#'   new environment.
#' @export
#' @examples
#' \dontrun{
#' # Run once after installing the package:
#' survivehr_setup()
#'
#' # Force a clean reinstall:
#' survivehr_setup(force = TRUE)
#' }
survivehr_setup <- function(envname = "survivehrR",
                            force   = FALSE,
                            python  = NULL) {

  packages <- c(
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "omegaconf",
    "pytorch-lightning",
    "hydra-core",
    "transformers"
  )

  already_exists <- reticulate::virtualenv_exists(envname)

  if (already_exists && !force) {
    message(sprintf(
      "Virtual environment '%s' already exists.  Use force=TRUE to reinstall.\n",
      envname
    ))
  } else {
    message(sprintf(
      "Creating Python virtual environment '%s' ...\n", envname
    ))
    create_args <- list(envname = envname)
    if (!is.null(python)) create_args[["python"]] <- python
    do.call(reticulate::virtualenv_create, create_args)

    message(
      "Installing Python packages (this may take several minutes on first run) ...\n"
    )
    reticulate::py_install(
      packages = packages,
      envname  = envname,
      pip      = TRUE
    )
    message("Python environment ready.\n")
  }

  .activate_survivehr_env(envname)

  invisible(reticulate::virtualenv_python(envname))
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Activate the survivehrR virtualenv when one exists.
# Called from .onLoad() and from .survivehr_backend() on every first call.
.activate_survivehr_env <- function(envname = "survivehrR") {
  if (reticulate::virtualenv_exists(envname)) {
    reticulate::use_virtualenv(envname, required = FALSE)
  }
}

