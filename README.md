# RSurvivEHR

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: GPL-3](https://img.shields.io/badge/License-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R ≥ 4.2](https://img.shields.io/badge/R-%E2%89%A5%204.2-276DC3?logo=r&logoColor=white)](https://cran.r-project.org/)
[![pkgdown](https://img.shields.io/badge/docs-pkgdown-brightgreen)](https://pm-cardoso.github.io/RSurvivEHR/)
<!-- badges: end -->

> Transformer-based survival analysis on Electronic Health Records — from R, with no Python experience required.

📦 **Website**: <a href="https://pm-cardoso.github.io/RSurvivEHR/" target="_blank" rel="noopener noreferrer">https://pm-cardoso.github.io/RSurvivEHR/</a>  
🐛 **Issues**: <a href="https://github.com/PM-Cardoso/RSurvivEHR/issues" target="_blank" rel="noopener noreferrer">https://github.com/PM-Cardoso/RSurvivEHR/issues</a>

`RSurvivEHR` wraps the **SurvivEHR** competing-risk transformer so you can pre-train on
longitudinal clinical event sequences, fine-tune on labelled outcomes, and generate patient-level
survival predictions — all from plain R data frames. The Python backend is vendored inside the
package and managed automatically via `reticulate`.

---

## Installation

```r
# install.packages("remotes")
remotes::install_github("PM-Cardoso/RSurvivEHR")
```

```r
# install.packages("pak")
pak::pkg_install("PM-Cardoso/RSurvivEHR")
```

Set up the Python backend once after installing:

```r
library(RSurvivEHR)
survivehr_setup()   # creates a dedicated virtualenv — safe to call repeatedly
```

---

## Pipeline

```
events + static ──▶ survivehr_pretrain()  ──▶ pt_model
                                                  │
                                      survivehr_save_model()  ←── reuse across outcomes
                                                  │
events + targets ─▶ survivehr_finetune()  ──▶ ft_model  (competing-risk or single-risk)
                                                  │
new events ────────▶ survivehr_predict()  ──▶ per-patient survival predictions
```

---

## Input format

| Table | Required columns | Optional |
|---|---|---|
| **events** | `patient_id`, `event`, `age` | `value` |
| **static_covariates** | `patient_id` + any covariate columns | — |
| **targets** (fine-tune) | `patient_id`, `target_event`, `target_age` | `target_value` |

- **`events`** and **`targets`** use fixed column names (`patient_id`, `event`,
  `age`, `value`; and `patient_id`, `target_event`, `target_age`, `target_value`
  respectively).  Uppercase alternatives (`PATIENT_ID`, `EVENT`,
  `DAYS_SINCE_BIRTH`) are also accepted.
- **`static_covariates`** column names are fully user-defined — use whatever your data has
  (`SEX`, `IMD`, `year_of_birth`, …). Categorical columns are one-hot encoded automatically;
  numeric columns pass through unchanged.
- Continuous readings (blood pressure, HbA1c, BMI, etc.) go in the `value` column alongside
  the event that recorded them. Rows without a reading should be `NA`.

---

## Getting started

The [Getting started vignette](https://pm-cardoso.github.io/RSurvivEHR/articles/getting-started.html)
walks through the full pipeline step by step, from raw data frames to patient-level
survival predictions.  It covers:

- Building and validating the `events`, `static_covariates`, and `targets` data frames
- Configuring and running pre-training
- Competing-risk fine-tuning (multiple outcomes) and single-risk fine-tuning (one outcome)
- Applying both leakage-prevention rules: removing outcome codes and truncating at the outcome age
- Generating predictions and interpreting the output columns
- Saving and reloading models

For configuration details and recommended hyperparameters for different dataset sizes, see the
[Model architecture & parameter reference](https://pm-cardoso.github.io/RSurvivEHR/articles/model-architecture.html).

For token policy, age normalisation, static covariate encoding, and a complete worked example,
see the [Advanced topics](https://pm-cardoso.github.io/RSurvivEHR/articles/advanced-topics.html) article.

---

## Key features

| Feature | Detail |
|---|---|
| **No manual Python setup** | Backend is vendored inside the package; `survivehr_setup()` handles everything. |
| **Competing & single risk** | `surv_layer = "competing-risk"` or `"single-risk"`. |
| **Leakage-free fine-tuning** | Context is restricted to events before the target age. |
| **Flexible static covariates** | Any column names; any mix of categorical and numeric. |
| **Continuous readings** | Record measurements (BP, HbA1c, BMI) in the `value` column. |
| **Save / load** | `survivehr_save_model()` / `survivehr_load_model()` preserve vocabulary, weights, and column schema. |
| **HPC-friendly** | Compatible with Python ≥ 3.8 (no `match`/`case` syntax). |

---

## Documentation

Full documentation, vignettes, and configuration reference at  
**<a href="https://pm-cardoso.github.io/RSurvivEHR/" target="_blank" rel="noopener noreferrer">https://pm-cardoso.github.io/RSurvivEHR/</a>**

- <a href="https://pm-cardoso.github.io/RSurvivEHR/articles/getting-started.html" target="_blank" rel="noopener noreferrer">Getting started</a> — step-by-step walkthrough of the full pipeline
- <a href="https://pm-cardoso.github.io/RSurvivEHR/articles/advanced-topics.html" target="_blank" rel="noopener noreferrer">Advanced topics</a> — token policy, age normalisation, static covariate encoding, FastEHR aliases, full worked example
- <a href="https://pm-cardoso.github.io/RSurvivEHR/articles/model-architecture.html" target="_blank" rel="noopener noreferrer">Model architecture & parameter reference</a> — every `survivehr_config()` parameter with recommended values

---

## Citation

If you use RSurvivEHR in your research, please cite the original SurvivEHR paper:

> Gadd, C. et al. (2025). *SurvivEHR: Transformer-based survival analysis on
> electronic health records*. medRxiv.
> <a href="https://doi.org/10.1101/2025.08.04.25332916" target="_blank" rel="noopener noreferrer">doi:10.1101/2025.08.04.25332916</a>
