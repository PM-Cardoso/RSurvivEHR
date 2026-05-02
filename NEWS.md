# RSurvivEHR 0.7.4

* **Python 3.8 compatibility fix (type annotations)**: two vendored Python
  files used PEP 585 built-in generic annotations (`list[int]`) that are only
  valid on Python 3.9+.  Replaced with `typing.List[int]` (added `List` to the
  `typing` import) in:
  - `src/modules/head_layers/survival/single_risk.py` — `target_indicies: list[int]`
  - `src/modules/head_layers/value_layers.py` — `Optional[list[int]]`
  This resolves the `TypeError: 'type' object is not subscriptable` crash on
  HPC systems running Python 3.8 (e.g. EasyBuild foss-2021b toolchain).

# RSurvivEHR 0.7.3

* **README quick start is now fully self-contained**: `events`,
  `static_covariates`, and `targets` data frames are defined inline (5 synthetic
  patients matching the Getting Started vignette) so the code block runs
  without error on copy-paste.  The `survivehr_finetune()` call now correctly
  filters out the outcome event (`CVD`) from the context to prevent data
  leakage, and `survivehr_validate_targets()` is included.
* **Getting Started vignette: fixed section numbering**: subsections
  `### 4a Competing-risk fine-tuning` and `### 4b Single-risk fine-tuning`
  were incorrectly nested inside `## 5 Fine-tuning`; renamed to `### 5a` and
  `### 5b` to match the parent section number.
* **Getting Started vignette: added References section** citing Gadd et al.
  (2025) with a DOI link.
* **External links now open in a new tab** on the pkgdown site: the website
  URL, issues link, and documentation section links in `README.md`, as well as
  the upstream config YAML links and DOI reference in
  `vignettes/model-architecture.Rmd`, are now HTML `<a target="_blank">` tags.
* **Citation section added to README**: directs users to Gadd et al. (2025),
  *SurvivEHR: Transformer-based survival analysis on electronic health records*,
  medRxiv, <https://doi.org/10.1101/2025.08.04.25332916>.

# RSurvivEHR 0.7.2

* **Package repository renamed to `RSurvivEHR`**: installation instructions
  updated throughout — `remotes::install_github("PM-Cardoso/RSurvivEHR")` and
  `pak::pkg_install("PM-Cardoso/RSurvivEHR")`.  The `ref = "R-package"`
  branch qualifier is no longer needed.
* **README rewritten**: modern layout with badges, pipeline diagram, quick-start
  example, and key-features table.  Dead table-of-contents sections removed;
  full detail lives in the pkgdown vignettes.
* **All internal URLs corrected**: `_pkgdown.yml`, `DESCRIPTION` (`URL:`,
  `BugReports:`), and the Getting started vignette now consistently reference
  `https://github.com/PM-Cardoso/RSurvivEHR` and
  `https://pm-cardoso.github.io/RSurvivEHR/`.

# RSurvivEHR 0.7.1

* **Python 3.8 / 3.9 compatibility fix**: the vendored SurvivEHR model code used
  Python 3.10+ structural pattern matching (`match`/`case`) in four files:
  `src/models/TTE/base.py`, `src/models/TTE/task_heads/causal.py`,
  `src/models/TTE/task_heads/causal_tabular.py`, and
  `src/modules/head_layers/survival/desurv.py`.  All `match`/`case` blocks have
  been rewritten as `if`/`elif`/`else` chains, with `isinstance()` replacing
  type-pattern cases (`case int()`, `case list()`).  The package now runs on
  managed HPC systems (e.g. EasyBuild foss-2021b toolchain, Python 3.9.6) and
  any Python >= 3.8.

# RSurvivEHR 0.7.0

* **Fixed static covariate validation for new / small-batch patients**: the
  previous strict equality check on one-hot encoded column names failed for
  new patients because `pd.get_dummies` on a single row only produces dummy
  columns for categories present in that row (e.g. `SEX_M` but not `SEX_F`).
  Validation is now split into two layers:
  – **Raw column names** (e.g. `SEX`, `IMD`, `HEIGHT_CM`) are validated
    strictly — a clear `ValueError` is raised if the user passes the wrong
    feature names.
  – **Encoded (one-hot) columns** are aligned silently via `reindex` with
    `fill_value=0.0` — categories absent from a small prediction batch are
    correctly filled with zeros rather than causing an error.
* **Model bundles now store both `static_raw_cols` and `static_col_names`**:
  raw feature names (before one-hot expansion) are stored alongside encoded
  names. `save_model_bundle` / `load_model_bundle` persist both so the
  two-layer validation survives a save/reload cycle.
* **Terminology standardised across all documentation**: static covariate
  column names are now consistently `UPPER_CASE` (`SEX`, `ETHNICITY`,
  `SMOKING_STATUS`, `IMD`, `YEAR_OF_BIRTH`) in the README, getting-started
  vignette, advanced-topics vignette, and `test.R`.  The variable holding
  static data is consistently named `static_covariates` (not `static`).
  The duplicate `## 2` section heading in the getting-started vignette has
  been corrected to `## 3 Configuration`.
* **Column naming rules now explicitly documented**: the README, getting-started
  vignette, and advanced-topics vignette now state clearly that `events` and
  `targets` tables require **fixed column names** (`patient_id`, `event`, `age`,
  `value`; and `patient_id`, `target_event`, `target_age`, `target_value`) or
  their FastEHR UPPER_CASE aliases, while **static covariate column names are
  completely user-defined** — any name is accepted beyond `patient_id`.  A
  "Column naming rules" summary table has been added to the README Input format
  section.
* **All examples now include realistic numeric readings**: `BP_CHECK` (systolic
  blood pressure, mmHg), `HBA1C` (HbA1c, mmol/mol), and `BMI` (kg/m²) events
  with actual values are used throughout the README, getting-started vignette,
  and advanced-topics vignette.  This shows clearly how continuous measurements
  are recorded alongside discrete events and how `value_weight > 0` can enable
  an auxiliary regression head for these readings.  Vocabulary and token-layout
  comments have been updated to reflect the richer event set.
* **Fixed getting-started vignette section numbering**: the cascade of duplicate
  `## 3` headings (introduced when `## 2 Configuration` was renumbered) has
  been corrected to 1 Setup → 2 Input data → 3 Configuration → 4 Pre-training
  → 5 Fine-tuning → 6 Save and load → 7 Prediction.

# RSurvivEHR 0.6.0

* **Static covariate column validation**: the model now records the exact
  encoded column list (including one-hot expanded categories) at training time.
  Passing mismatched columns at prediction time raises a descriptive
  `ValueError` rather than silently producing wrong predictions.
* **Save/load pretrained model**: examples in README, getting-started vignette,
  and `test.R` now demonstrate `survivehr_save_model()` immediately after
  pre-training so the backbone can be reused for multiple fine-tuned models.
* **Rewritten Getting Started vignette**: concise code-first style matching the
  README worked example.  Now includes both competing-risk and single-risk
  fine-tuning examples, save/load of pretrained and fine-tuned models, and
  a correct new-patient inference example.
* **Fixed README new-patient example**: `new_static` now supplies all static
  columns used during training (previously it was missing `ETHNICITY`,
  `SMOKING_STATUS`, `EYE_COLOUR`, and `HEIGHT_CM`, which would have raised a
  column-mismatch error with the new validation).
* **Validate functions now print a console summary on success** — e.g.
  `[OK] Events: 17 rows, 5 patients. Columns present, ages numeric and time-ordered.`
  — so users get positive confirmation rather than silent `invisible(TRUE)`.
* `save_model_bundle` / `load_model_bundle` now persist `static_col_names` in
  the `.pt` checkpoint so column validation survives a save/reload cycle.

# RSurvivEHR 0.5.0

* Updated vignette **"Model architecture & parameter reference"** with the
  official hyperparameter table from Gadd et al. (2025). Now documents
  pre-training vs fine-tuning differences for every parameter:
  `block_size` (256 vs 512), `batch_size` (64 vs 512), `epochs` (10 vs 20),
  backbone LR (3e-4 vs 5e-5), head LR (3e-4 vs 5e-4), scheduler (linear
  warmup + cosine annealing with warm restarts vs ReduceLROnPlateau), and
  early stopping (disabled vs enabled with patience 30).
* Added context-window explanation: fine-tuning uses last-unique context
  with global diagnoses appended.
* Added separate `cfg_pretrain` and `cfg_finetune` recommended configs for
  HPC / large-dataset users.

# RSurvivEHR 0.4.0

* Fixed R CMD check WARNING: added missing `man/survivehr_setup.Rd` by
  running `devtools::document()`. The package now passes `R CMD check`
  with no warnings on macOS-latest and ubuntu-latest.
* Added new vignette **"Model architecture & parameter reference"**
  (`vignettes/model-architecture.Rmd`) documenting every `survivehr_config()`
  parameter, its upstream default (from `cwlgadd/SurvivEHR`), and guidance
  on choosing values for small, medium, and large datasets.
  Includes a full table of transformer architecture choices, ODE survival
  head design, vocabulary construction rules, and advanced fine-tuning
  options (PEFT, layer-wise LR decay, compression layer).
* Updated `_pkgdown.yml` to include the new vignette in the Articles menu and
  added a "Setup" section for `survivehr_setup` in the function reference
  index, fixing the `pkgdown::build_site()` error *"1 topic missing from
  index: 'survivehr_setup'"*.

# RSurvivEHR 0.3.0

* Replaced the simple dot-per-batch progress indicator in `_run_train_loop`
  with a real-time inline progress bar showing epoch number, a filled/empty
  bar, batch fraction, current loss, and estimated time remaining (ETA).
  The final line for each epoch is overwritten with a clean summary so the
  console stays readable across many epochs.
* Added a **single-risk fine-tuning** example to `test.R` (Example B)
  alongside the existing competing-risk example (Example A). The two
  examples use independent configs (`cfg_cr` / `cfg_sr`) and produce
  separate prediction frames (`preds_cr` / `preds_sr`).
* Updated the *Getting started* vignette to document both fine-tuning modes
  in separate sub-sections (§ 4a Competing-risk, § 4b Single-risk), with a
  comparison table and updated prediction examples for both models.

# RSurvivEHR 0.2.0

* Added `survivehr_setup()` to create and populate the `RSurvivEHR` Python
  virtual environment. Works on macOS, Windows, and Linux.  Added to
  `NAMESPACE` exports and `R/setup.R`.
* Added `transformers` to the list of required Python packages installed by
  `survivehr_setup()` (needed by `src/models/TTE/base.py` and
  `src/models/transformer/base.py` which use
  `transformers.modeling_utils.ModuleUtilsMixin`).
* Fixed circular import: cleared eager top-level imports from
  `inst/python/SurvivEHR/__init__.py` (`from . import examples` /
  `from . import src`) that caused `ImportError` on package load.
* Replaced broken import (`SurvivEHR.examples.modelling...`) in
  `inst/python/survivehr_backend.py` with the new
  `SurvivEHR.experiments` module.
* Added `inst/python/SurvivEHR/experiments.py` providing
  `CausalExperiment` and `FineTuneExperiment` wrapper classes with a
  dict-batch interface compatible with `_run_train_loop`.  Includes a
  no-op `wandb` stub so the backend works without `wandb` installed.
* Fine-tune weight transfer now handles vocabulary size mismatches (e.g.
  when an outcome token such as `CVD` was never seen during pre-training):
  missing tokens are appended to the vocabulary and the embedding weight
  matrix is extended using an overlapping-slice copy of the pre-trained
  parameters.
* Fixed `load_state_dict` failure caused by uninitialised lazy PyTorch
  parameters in `FineTuneExperiment`: a dummy forward pass now initialises
  all parameter shapes before pre-trained weights are transferred.
* Set `PYTHONUNBUFFERED=1` in `.onLoad` so training-loop progress lines
  are printed to the R console in real time.
* Improved error message in `.survivehr_backend()` when the Python
  environment has not been set up (directs user to `survivehr_setup()`).



# RSurvivEHR 0.1.0

## First public release

* Full R interface to SurvivEHR: `survivehr_config()`, `survivehr_pretrain()`,
  `survivehr_finetune()`, `survivehr_predict()`, `survivehr_save_model()`,
  `survivehr_load_model()`, `survivehr_validate_events()`,
  `survivehr_validate_static()`, `survivehr_validate_targets()`.
* Python backend vendored inside `inst/python/` — no FastEHR installation required.
* Explicit PAD / UNK / CLS / SEP token policy configurable via `survivehr_config()`.
* Global `time_scale` normalisation (default `1.0` for ages in years;
  use `1825.0` to match FastEHR's `DAYS_SINCE_BIRTH` convention).
* Automatic static covariate encoding: categorical columns → one-hot dummies;
  numeric columns → passed through as-is.
* Competing-risk and single-risk survival heads supported.
* Leakage-free fine-tuning: context sequence is restricted to events *before*
  the target event age.
* Accepts FastEHR column aliases (`PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`).
