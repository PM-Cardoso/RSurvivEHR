# Changelog

## RSurvivEHR 0.8.2

#### Vignette rewrite — model architecture and parameter reference

`vignettes/model-architecture.Rmd` has been fully rewritten:

- **Accurate documentation**: removed incorrect claims that had been
  introduced in earlier drafts — no context-window deduplication in the
  backend, no learning-rate scheduler, and no early stopping are
  implemented; the single `AdamW` optimiser is now clearly described.
- **New structure**: navigable jump links at the top; “How the model
  works” section with architecture diagram; “Quick-start configurations”
  with annotated code blocks for laptop / workstation / HPC; upstream
  hyperparameter comparison table with clickable anchors; full
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  annotated code block; per-parameter sections grouped by theme
  (architecture, regularisation, optimisation, loss weights,
  tokenisation, time & age, hardware); fine-tuning notes table;
  references.
- **All 15
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  parameters** now covered with default and upstream comparison values,
  including `dropout`, `time_scale`, and `value_weight` which were
  previously missing from the comparison table.
- **`time_scale` table** extended to show desired-window guidance
  (1-year vs 5-year windows for both year-based and day-based age
  columns).

#### Documentation fixes (this release)

- `vignettes/advanced-topics.Rmd`: `static_pop` worked example now uses
  lowercase column names (`sex`, `ethnicity`, `imd`, `year_of_birth`).
- `vignettes/getting-started.Rmd`: `time_scale` and `value_weight`
  argument order in the main `cfg` block corrected to match canonical
  parameter order.

#### Citation update — paper published in *npj Digital Medicine* (2026)

The SurvivEHR paper has been published:

> Gadd, C., Gokhale, K., Acharya, A. et al. (2026). *SurvivEHR: a
> competing risks, time-to-event foundation model for multiple long-term
> conditions from primary care electronic health records*. npj Digital
> Medicine. <doi:10.1038/s41746-026-02709-z>

All citations updated from the medRxiv preprint (2025,
<doi:10.1101/2025.08.04.25332916>) to the published journal article
throughout:

- `README.md` — citation block.
- `vignettes/getting-started.Rmd` — References section and “Gadd et
  al. 2025” inline reference.
- `vignettes/model-architecture.Rmd` — section heading “Upstream
  hyperparameters”, introductory sentence, and References block.
- `R/config.R` — `@description` Roxygen comment.

#### Documentation fixes

- `vignettes/data-pipeline.Rmd`: `static_pop` example and Step 6 table
  now use lowercase column names (`sex`, `ethnicity`, `imd`,
  `year_of_birth`) consistent with the lowercase-first convention
  adopted in 0.8.1.
- `vignettes/getting-started.Rmd`: `new_static` inference example now
  uses lowercase column names (`sex`, `ethnicity`, `imd`,
  `year_of_birth`).

------------------------------------------------------------------------

## RSurvivEHR 0.8.1

#### Column naming standardised to lowercase

Lowercase column names are now the **canonical, preferred form** across
all input data frames. Uppercase aliases continue to work for backward
compatibility but are no longer showcased in examples.

**Fixed/event column names** (required names or their accepted aliases):

| Table   | Canonical (preferred) | Accepted alias                   |
|---------|-----------------------|----------------------------------|
| Events  | `patient_id`          | `PATIENT_ID`                     |
| Events  | `event`               | `EVENT`                          |
| Events  | `age`                 | `DAYS_SINCE_BIRTH`               |
| Events  | `value`               | `VALUE`                          |
| Targets | `patient_id`          | `PATIENT_ID`                     |
| Targets | `target_event`        | `TARGET_EVENT`, `EVENT`          |
| Targets | `target_age`          | `TARGET_AGE`, `DAYS_SINCE_BIRTH` |
| Targets | `target_value`        | `TARGET_VALUE`, `VALUE`          |
| Static  | `patient_id`          | `PATIENT_ID`                     |

**Static covariate column names are freely chosen by the user** —
lowercase is now recommended (e.g. `sex`, `ethnicity`, `imd`,
`year_of_birth`) but any name is accepted.

#### Documentation

- `R/validate.R`: `@param` descriptions for
  [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md),
  [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md),
  and
  [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md)
  now list lowercase as canonical and show each uppercase alias
  explicitly.
- `R/train.R`: `@param events` clarified; `static_pop` example updated
  to use lowercase covariate names (`sex`, `ethnicity`, `imd`,
  `year_of_birth`).
- `vignettes/getting-started.Rmd`: `static_pop` example and the “Column
  naming” callout updated to use and recommend lowercase covariate
  names; validate output comment updated to match.
- `vignettes/advanced-topics.Rmd`: static covariate example and “Column
  naming rule” callout updated to use and recommend lowercase names; the
  inline encoding output example updated accordingly.
- `DESCRIPTION`: description updated to mention the lowercase column
  convention and backward-compatible uppercase aliases.

#### Bug fix (from 0.8.0)

- Fixed
  `IndexError: index 64 is out of bounds for dimension 1 with size 64`
  in
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  on pre-train models when one or more patients had a fully-packed
  context window (`block_size` events). Root cause: the `generate()`
  method in `causal.py` sliced `attention_mask[:, :block_size]` but
  forgot to assign the result back, leaving the mask one column wider
  than the trimmed token tensor after generation. Fixed with
  `attention_mask = attention_mask[:, :block_size]`.

------------------------------------------------------------------------

## RSurvivEHR 0.8.0

#### New features

- **Multi-step pre-train predictions**
  ([`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  on pre-train models): calling `max_new_tokens = n` now returns one row
  per generated step per patient (instead of only the first step). New
  columns: `step` (1-indexed generation step), `generated_event`,
  `generated_age` (de-normalised by `time_scale`), `generated_value`
  (numeric measurement or `NaN`).

- **Time-specific CDF columns**
  ([`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  on fine-tuned models): the new `eval_times` argument accepts a numeric
  vector of time points in raw units (same scale as `age`). For each
  time point `t` and each outcome, a column `{outcome}_cdf_t{t}` is
  appended to the output. Values must be in `(0, time_scale]`.

- **New function
  [`survivehr_predict_value()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict_value.md)**:
  predicts the numeric value (mean and SD of the model’s Gaussian output
  head) for a specified outcome event, using either a pre-train or
  fine-tuned model bundle. Returns a data frame with columns
  `patient_id`, `outcome_event`, `predicted_value_mean`,
  `predicted_value_sd`; `NaN` for events without a measurement head.

#### Documentation

- Updated `vignettes/getting-started.Rmd` with examples for all three
  new features (Sections 4 and 7).
- Expanded “Understanding the output columns” table to cover pre-train
  and value-prediction outputs in addition to fine-tuned model outputs.

------------------------------------------------------------------------

## RSurvivEHR 0.7.9

- **New vignette — “Data pipeline internals”**
  (`vignettes/data-pipeline.Rmd`): step-by-step walkthrough of how R
  data frames are transformed into PyTorch tensors. Covers R validation
  helpers, the reticulate handoff, event cleaning, vocabulary
  construction, tokenisation, age normalisation and padding, static
  covariate one-hot encoding, fine-tune target delta computation, and
  the final DataLoader batch structure. Each step links directly to the
  relevant function in the source code. The vignette is listed under
  *Articles → Data pipeline internals* in the pkgdown site.

## RSurvivEHR 0.7.8

- **Consistent 10-patient example dataset throughout**: all R function
  `@examples` in `R/train.R` now use the same 10-patient population
  (`events_pop` / `static_pop`) as the *Getting Started* vignette. Each
  patient has realistic prior-history events (e.g. `HYPERTENSION`,
  `BP_CHECK`) before their outcome event, matching the structure of real
  EHR data.
- **Competing-risk example now genuinely uses two outcomes**: the
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  example and the *Getting Started* vignette Section 5a now model **CVD
  vs T2D** (`outcomes = c("CVD", "T2D")`,
  `risk_model = "competing-risk"`). The previous example used only a
  single outcome, which is incorrect for competing-risk models.
- **Leakage-free cohort construction in all examples**: targets are now
  defined *before* filtering the context events. The merge-and-filter
  pattern (`age < target_age`) is used consistently across the vignette
  and function examples, ensuring no post-outcome events appear in the
  input.
- **`time_scale` removed from fine-tune configs in examples**: `cfg_cr`
  and `cfg_sr` no longer set `time_scale` (it is inherited automatically
  from the pretrained bundle). A comment explains the inheritance.
- **Improved roxygen documentation for all exported functions**:
  - [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
    [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
    [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md),
    [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md),
    [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md):
    full `@description`, `@param`, and `@return` sections replacing
    one-liner stubs.
  - [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md),
    [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md),
    [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md):
    detailed `@param` with `\describe{}` item lists explaining each
    column and its accepted aliases.
  - [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md):
    description updated to reference the *Model architecture* vignette
    for Gadd et al. (2025) hyperparameters.
- **Fixed single-risk prediction column naming and post-reload
  breakage**: A reticulate round-trip conversion bug caused a
  single-element Python list `["CVD"]` to become the bare string `"CVD"`
  after the bundle travelled `Python → R list → Python` (i.e. on every
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  call). This produced two symptoms:
  - With the in-memory bundle (`ft_sr`):
    [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
    iterated the string character-by-character, labelling the CDF column
    `"C_cdf_last"` instead of `"CVD_cdf_last"`.
  - After
    [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
    /
    [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md):
    the string was persisted, then `load_model_bundle` iterated `"C"`,
    `"V"`, `"D"` as outcome names — none present in the vocabulary — so
    `outcome_tokens = []` and the reloaded head had 0 risks, returning a
    data frame with only `patient_id`. Fixed by normalising `outcomes`
    to a proper list at four sites in `survivehr_backend.py`:
    `train_finetune_model` return, `predict_next_events`,
    `save_model_bundle`, and `load_model_bundle`. Single-outcome
    single-risk models (the most common case) are now correct in all
    paths. adjacent string literals without a separating comma (not
    valid R syntax). Replaced with
    [`paste0()`](https://rdrr.io/r/base/paste.html) wrapping both
    strings.
- **`DESCRIPTION` description updated**: removed the outdated “no
  FastEHR installation required” clause; added mentions of biomarker
  value support, leakage-free helpers, and frequency-ordered
  vocabularies.
- **`advanced-topics.Rmd` full worked example aligned**: the standalone
  6-patient dataset in the “Full worked example” section has been
  replaced with the same 10-patient population used in *Getting
  Started*, ensuring all documentation is consistent.
- **README quick-start replaced with prose**: the R code block in the
  README that used a different 5-patient dataset has been replaced with
  a prose “Getting started” section pointing readers to the three
  vignettes.

## RSurvivEHR 0.7.7

- **5-year prediction window**: all examples and documentation now use
  `time_scale = 5.0`, giving a 5-year prediction window when ages are in
  years. The
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  default remains `1.0` but the recommended value for clinical
  applications is `5.0`.
- **Clarified `time_scale` semantics**: `time_scale` controls **both**
  the age normalisation divisor and the prediction window length. The
  survival ODE evaluates over a normalised \[0, 1\] grid that maps back
  to \[0, `time_scale`\] in raw age units. `time_scale` is stored inside
  every model bundle;
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  reads it automatically — it does not need to be supplied at inference
  time. This is now documented in
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  `@param`,
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  `@return`, the backend docstring, and the *Getting Started* vignette
  (Section 3 and Section 7).
- **Biomarker values in examples**: the 10-patient population in
  `R/train.R` examples and `vignettes/getting-started.Rmd` now includes
  realistic `value` entries: BP_CHECK readings (138–162 mmHg) and HBA1C
  readings (68–74 mmol/mol) aligned to the correct event rows.
- **Removed FastEHR branding from `README.md`**: the “FastEHR
  compatibility” row has been removed from the key-features table and
  the FastEHR alias description in the Input format section has been
  reworded to plain English.

## RSurvivEHR 0.7.6

- **Frequency-ordered vocabulary**: event tokens are now assigned IDs in
  descending frequency order (most common event = smallest ID after
  reserved tokens `<PAD>`, `<UNK>`). Smaller IDs improve embedding
  compression for common clinical codes. Existing saved models are
  unaffected because the vocabulary is stored inside every `.pt` bundle.
- **Improved examples throughout**: all R function examples and the
  *Getting Started* vignette now use a 10-patient pre-training cohort
  with three CVD cases providing the outcome token, and a 6-patient
  fine-tuning subset (3 CVD cases + 3 right-censored controls). Context
  events are explicitly filtered to remove the outcome (leakage-free).
- **Documented prediction outputs and time window**:
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md)
  `@return` now fully documents `{outcome}_cdf_last` (cumulative
  incidence at the end of the prediction window = `time_scale` time
  units) and `{outcome}_auc` (area under the CDF = average risk over the
  window). The vignette explains how to extend the prediction window by
  changing `time_scale`.
- **Removed FastEHR branding**: `fastEHR` / `FastEHR` mentions removed
  from all documentation, error messages, and parameter descriptions.
  Uppercase column aliases (`PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`)
  continue to work but are described without the FastEHR name.

## RSurvivEHR 0.7.5

- **CPU/GPU device visibility**: the training start-up banner now prints
  a human-readable device string including the GPU model when CUDA is
  available (e.g. `device=cuda:0 (Tesla V100-SXM2-16GB)` vs
  `device=cpu`). The `[survivehrR]` prefix in that message has been
  corrected to `[RSurvivEHR]`.
- **`device` field in model bundles**: the named list returned by
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md)
  and
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  now includes a `device` element (a plain string such as `"cpu"` or
  `"cuda:0"`). The same field is persisted by
  [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
  and restored by
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md).

## RSurvivEHR 0.7.4

- **Python 3.8 compatibility fix (type annotations)**: two vendored
  Python files used PEP 585 built-in generic annotations (`list[int]`)
  that are only valid on Python 3.9+. Replaced with `typing.List[int]`
  (added `List` to the `typing` import) in:
  - `src/modules/head_layers/survival/single_risk.py` —
    `target_indicies: list[int]`
  - `src/modules/head_layers/value_layers.py` — `Optional[list[int]]`
    This resolves the `TypeError: 'type' object is not subscriptable`
    crash on HPC systems running Python 3.8 (e.g. EasyBuild foss-2021b
    toolchain).

## RSurvivEHR 0.7.3

- **README quick start is now fully self-contained**: `events`,
  `static_covariates`, and `targets` data frames are defined inline (5
  synthetic patients matching the Getting Started vignette) so the code
  block runs without error on copy-paste. The
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md)
  call now correctly filters out the outcome event (`CVD`) from the
  context to prevent data leakage, and
  [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md)
  is included.
- **Getting Started vignette: fixed section numbering**: subsections
  `### 4a Competing-risk fine-tuning` and
  `### 4b Single-risk fine-tuning` were incorrectly nested inside
  `## 5 Fine-tuning`; renamed to `### 5a` and `### 5b` to match the
  parent section number.
- **Getting Started vignette: added References section** citing Gadd et
  al.
  2025. with a DOI link.
- **External links now open in a new tab** on the pkgdown site: the
  website URL, issues link, and documentation section links in
  `README.md`, as well as the upstream config YAML links and DOI
  reference in `vignettes/model-architecture.Rmd`, are now HTML
  `<a target="_blank">` tags.
- **Citation section added to README**: directs users to Gadd et
  al. (2025), *SurvivEHR: Transformer-based survival analysis on
  electronic health records*, medRxiv,
  <https://doi.org/10.1101/2025.08.04.25332916>.

## RSurvivEHR 0.7.2

- **Package repository renamed to `RSurvivEHR`**: installation
  instructions updated throughout —
  `remotes::install_github("PM-Cardoso/RSurvivEHR")` and
  `pak::pkg_install("PM-Cardoso/RSurvivEHR")`. The `ref = "R-package"`
  branch qualifier is no longer needed.
- **README rewritten**: modern layout with badges, pipeline diagram,
  quick-start example, and key-features table. Dead table-of-contents
  sections removed; full detail lives in the pkgdown vignettes.
- **All internal URLs corrected**: `_pkgdown.yml`, `DESCRIPTION`
  (`URL:`, `BugReports:`), and the Getting started vignette now
  consistently reference `https://github.com/PM-Cardoso/RSurvivEHR` and
  `https://pm-cardoso.github.io/RSurvivEHR/`.

## RSurvivEHR 0.7.1

- **Python 3.8 / 3.9 compatibility fix**: the vendored SurvivEHR model
  code used Python 3.10+ structural pattern matching (`match`/`case`) in
  four files: `src/models/TTE/base.py`,
  `src/models/TTE/task_heads/causal.py`,
  `src/models/TTE/task_heads/causal_tabular.py`, and
  `src/modules/head_layers/survival/desurv.py`. All `match`/`case`
  blocks have been rewritten as `if`/`elif`/`else` chains, with
  `isinstance()` replacing type-pattern cases (`case int()`,
  `case list()`). The package now runs on managed HPC systems
  (e.g. EasyBuild foss-2021b toolchain, Python 3.9.6) and any Python \>=
  3.8.

## RSurvivEHR 0.7.0

- **Fixed static covariate validation for new / small-batch patients**:
  the previous strict equality check on one-hot encoded column names
  failed for new patients because `pd.get_dummies` on a single row only
  produces dummy columns for categories present in that row
  (e.g. `SEX_M` but not `SEX_F`). Validation is now split into two
  layers: – **Raw column names** (e.g. `SEX`, `IMD`, `HEIGHT_CM`) are
  validated strictly — a clear `ValueError` is raised if the user passes
  the wrong feature names. – **Encoded (one-hot) columns** are aligned
  silently via `reindex` with `fill_value=0.0` — categories absent from
  a small prediction batch are correctly filled with zeros rather than
  causing an error.
- **Model bundles now store both `static_raw_cols` and
  `static_col_names`**: raw feature names (before one-hot expansion) are
  stored alongside encoded names. `save_model_bundle` /
  `load_model_bundle` persist both so the two-layer validation survives
  a save/reload cycle.
- **Terminology standardised across all documentation**: static
  covariate column names are now consistently `UPPER_CASE` (`SEX`,
  `ETHNICITY`, `SMOKING_STATUS`, `IMD`, `YEAR_OF_BIRTH`) in the README,
  getting-started vignette, advanced-topics vignette, and `test.R`. The
  variable holding static data is consistently named `static_covariates`
  (not `static`). The duplicate `## 2` section heading in the
  getting-started vignette has been corrected to `## 3 Configuration`.
- **Column naming rules now explicitly documented**: the README,
  getting-started vignette, and advanced-topics vignette now state
  clearly that `events` and `targets` tables require **fixed column
  names** (`patient_id`, `event`, `age`, `value`; and `patient_id`,
  `target_event`, `target_age`, `target_value`) or their FastEHR
  UPPER_CASE aliases, while **static covariate column names are
  completely user-defined** — any name is accepted beyond `patient_id`.
  A “Column naming rules” summary table has been added to the README
  Input format section.
- **All examples now include realistic numeric readings**: `BP_CHECK`
  (systolic blood pressure, mmHg), `HBA1C` (HbA1c, mmol/mol), and `BMI`
  (kg/m²) events with actual values are used throughout the README,
  getting-started vignette, and advanced-topics vignette. This shows
  clearly how continuous measurements are recorded alongside discrete
  events and how `value_weight > 0` can enable an auxiliary regression
  head for these readings. Vocabulary and token-layout comments have
  been updated to reflect the richer event set.
- **Fixed getting-started vignette section numbering**: the cascade of
  duplicate `## 3` headings (introduced when `## 2 Configuration` was
  renumbered) has been corrected to 1 Setup → 2 Input data → 3
  Configuration → 4 Pre-training → 5 Fine-tuning → 6 Save and load → 7
  Prediction.

## RSurvivEHR 0.6.0

- **Static covariate column validation**: the model now records the
  exact encoded column list (including one-hot expanded categories) at
  training time. Passing mismatched columns at prediction time raises a
  descriptive `ValueError` rather than silently producing wrong
  predictions.
- **Save/load pretrained model**: examples in README, getting-started
  vignette, and `test.R` now demonstrate
  [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md)
  immediately after pre-training so the backbone can be reused for
  multiple fine-tuned models.
- **Rewritten Getting Started vignette**: concise code-first style
  matching the README worked example. Now includes both competing-risk
  and single-risk fine-tuning examples, save/load of pretrained and
  fine-tuned models, and a correct new-patient inference example.
- **Fixed README new-patient example**: `new_static` now supplies all
  static columns used during training (previously it was missing
  `ETHNICITY`, `SMOKING_STATUS`, `EYE_COLOUR`, and `HEIGHT_CM`, which
  would have raised a column-mismatch error with the new validation).
- **Validate functions now print a console summary on success** — e.g.
  `[OK] Events: 17 rows, 5 patients. Columns present, ages numeric and time-ordered.`
  — so users get positive confirmation rather than silent
  `invisible(TRUE)`.
- `save_model_bundle` / `load_model_bundle` now persist
  `static_col_names` in the `.pt` checkpoint so column validation
  survives a save/reload cycle.

## RSurvivEHR 0.5.0

- Updated vignette **“Model architecture & parameter reference”** with
  the official hyperparameter table from Gadd et al. (2025). Now
  documents pre-training vs fine-tuning differences for every parameter:
  `block_size` (256 vs 512), `batch_size` (64 vs 512), `epochs` (10 vs
  20), backbone LR (3e-4 vs 5e-5), head LR (3e-4 vs 5e-4), scheduler
  (linear warmup + cosine annealing with warm restarts vs
  ReduceLROnPlateau), and early stopping (disabled vs enabled with
  patience 30).
- Added context-window explanation: fine-tuning uses last-unique context
  with global diagnoses appended.
- Added separate `cfg_pretrain` and `cfg_finetune` recommended configs
  for HPC / large-dataset users.

## RSurvivEHR 0.4.0

- Fixed R CMD check WARNING: added missing `man/survivehr_setup.Rd` by
  running `devtools::document()`. The package now passes `R CMD check`
  with no warnings on macOS-latest and ubuntu-latest.
- Added new vignette **“Model architecture & parameter reference”**
  (`vignettes/model-architecture.Rmd`) documenting every
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md)
  parameter, its upstream default (from `cwlgadd/SurvivEHR`), and
  guidance on choosing values for small, medium, and large datasets.
  Includes a full table of transformer architecture choices, ODE
  survival head design, vocabulary construction rules, and advanced
  fine-tuning options (PEFT, layer-wise LR decay, compression layer).
- Updated `_pkgdown.yml` to include the new vignette in the Articles
  menu and added a “Setup” section for `survivehr_setup` in the function
  reference index, fixing the
  [`pkgdown::build_site()`](https://pkgdown.r-lib.org/reference/build_site.html)
  error *“1 topic missing from index: ‘survivehr_setup’”*.

## RSurvivEHR 0.3.0

- Replaced the simple dot-per-batch progress indicator in
  `_run_train_loop` with a real-time inline progress bar showing epoch
  number, a filled/empty bar, batch fraction, current loss, and
  estimated time remaining (ETA). The final line for each epoch is
  overwritten with a clean summary so the console stays readable across
  many epochs.
- Added a **single-risk fine-tuning** example to `test.R` (Example B)
  alongside the existing competing-risk example (Example A). The two
  examples use independent configs (`cfg_cr` / `cfg_sr`) and produce
  separate prediction frames (`preds_cr` / `preds_sr`).
- Updated the *Getting started* vignette to document both fine-tuning
  modes in separate sub-sections (§ 4a Competing-risk, § 4b
  Single-risk), with a comparison table and updated prediction examples
  for both models.

## RSurvivEHR 0.2.0

- Added
  [`survivehr_setup()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_setup.md)
  to create and populate the `RSurvivEHR` Python virtual environment.
  Works on macOS, Windows, and Linux. Added to `NAMESPACE` exports and
  `R/setup.R`.
- Added `transformers` to the list of required Python packages installed
  by
  [`survivehr_setup()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_setup.md)
  (needed by `src/models/TTE/base.py` and
  `src/models/transformer/base.py` which use
  `transformers.modeling_utils.ModuleUtilsMixin`).
- Fixed circular import: cleared eager top-level imports from
  `inst/python/SurvivEHR/__init__.py` (`from . import examples` /
  `from . import src`) that caused `ImportError` on package load.
- Replaced broken import (`SurvivEHR.examples.modelling...`) in
  `inst/python/survivehr_backend.py` with the new
  `SurvivEHR.experiments` module.
- Added `inst/python/SurvivEHR/experiments.py` providing
  `CausalExperiment` and `FineTuneExperiment` wrapper classes with a
  dict-batch interface compatible with `_run_train_loop`. Includes a
  no-op `wandb` stub so the backend works without `wandb` installed.
- Fine-tune weight transfer now handles vocabulary size mismatches (e.g.
  when an outcome token such as `CVD` was never seen during
  pre-training): missing tokens are appended to the vocabulary and the
  embedding weight matrix is extended using an overlapping-slice copy of
  the pre-trained parameters.
- Fixed `load_state_dict` failure caused by uninitialised lazy PyTorch
  parameters in `FineTuneExperiment`: a dummy forward pass now
  initialises all parameter shapes before pre-trained weights are
  transferred.
- Set `PYTHONUNBUFFERED=1` in `.onLoad` so training-loop progress lines
  are printed to the R console in real time.
- Improved error message in `.survivehr_backend()` when the Python
  environment has not been set up (directs user to
  [`survivehr_setup()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_setup.md)).

## RSurvivEHR 0.1.0

### First public release

- Full R interface to SurvivEHR:
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md),
  [`survivehr_pretrain()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_pretrain.md),
  [`survivehr_finetune()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_finetune.md),
  [`survivehr_predict()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_predict.md),
  [`survivehr_save_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_save_model.md),
  [`survivehr_load_model()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_load_model.md),
  [`survivehr_validate_events()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_events.md),
  [`survivehr_validate_static()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_static.md),
  [`survivehr_validate_targets()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_validate_targets.md).
- Python backend vendored inside `inst/python/` — no FastEHR
  installation required.
- Explicit PAD / UNK / CLS / SEP token policy configurable via
  [`survivehr_config()`](https://pm-cardoso.github.io/RSurvivEHR/reference/survivehr_config.md).
- Global `time_scale` normalisation (default `1.0` for ages in years;
  use `1825.0` to match FastEHR’s `DAYS_SINCE_BIRTH` convention).
- Automatic static covariate encoding: categorical columns → one-hot
  dummies; numeric columns → passed through as-is.
- Competing-risk and single-risk survival heads supported.
- Leakage-free fine-tuning: context sequence is restricted to events
  *before* the target event age.
- Accepts FastEHR column aliases (`PATIENT_ID`, `EVENT`,
  `DAYS_SINCE_BIRTH`).
