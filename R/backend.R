# Activate the dedicated virtualenv as early as possible (before any
# reticulate call locks in a Python binary).  .onLoad runs when the package
# namespace is loaded, before any user code runs.
.onLoad <- function(libname, pkgname) {
  # Flush Python stdout/stderr immediately so training progress appears live.
  Sys.setenv(PYTHONUNBUFFERED = "1")
  .activate_survivehr_env("survivehrR")
}

.survivehr_backend <- local({
  backend <- NULL

  function() {
    if (!is.null(backend)) {
      return(backend)
    }

    # Ensure the right Python is active.
    .activate_survivehr_env("survivehrR")

    # Give an actionable error if the Python environment has not been set up.
    if (!reticulate::py_module_available("torch")) {
      stop(
        "Required Python packages are not installed.\n",
        "Run `survivehrR::survivehr_setup()` once to create the Python ",
        "environment, then restart R and try again.",
        call. = FALSE
      )
    }

    # Locate inst/python — works both when the package is installed and when
    # it is loaded via devtools::load_all() / pkgload from the repo root.
    installed_path <- system.file("python", package = "survivehrR")
    backend_path <- if (nzchar(installed_path) && dir.exists(installed_path)) {
      installed_path
    } else {
      # Fallback for development: look for inst/python relative to the repo root.
      normalizePath(file.path(getwd(), "inst", "python"), mustWork = FALSE)
    }

    if (!dir.exists(backend_path)) {
      stop(
        "Could not locate `inst/python` for survivehrR backend.\n",
        "Searched: ", backend_path,
        call. = FALSE
      )
    }

    backend <<- reticulate::import_from_path(
      module     = "survivehr_backend",
      path       = backend_path,
      delay_load = FALSE
    )
    backend
  }
})

.to_py_dict <- function(x) {
  reticulate::r_to_py(x)
}
