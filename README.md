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

- **`events`** and **`targets`** use fixed column names. FastEHR `UPPER_CASE` aliases
  (`PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH`) are also accepted.
- **`static_covariates`** column names are fully user-defined — use whatever your data has
  (`SEX`, `IMD`, `year_of_birth`, …). Categorical columns are one-hot encoded automatically;
  numeric columns pass through unchanged.
- Continuous readings (blood pressure, HbA1c, BMI, etc.) go in the `value` column alongside
  the event that recorded them. Rows without a reading should be `NA`.

---

## Quick start

```r

# ── Sample data (5 patients, matching the Getting Started vignette) ───────────
events <- data.frame(
  patient_id = c(1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,5),
  event      = c("HYPERTENSION","STATIN","T2D","BP_CHECK","METFORMIN",
                 "T2D","METFORMIN","HBA1C","HYPERTENSION","STATIN",
                 "STATIN","T2D","BP_CHECK","HBA1C","METFORMIN",
                 "HYPERTENSION","AMLODIPINE","BP_CHECK","STATIN","T2D",
                 "T2D","STATIN","HBA1C","HYPERTENSION","METFORMIN","CVD"),
  age        = c(50.0,50.5,52.0,52.1,52.3,
                 45.0,45.3,46.0,47.5,48.0,
                 58.0,60.1,61.5,62.0,62.3,
                 55.0,55.4,56.0,57.2,58.0,
                 48.0,48.6,49.0,49.1,50.5,51.2),
  value      = c(NA,NA,NA,152,NA,  NA,NA,61,NA,NA,  NA,NA,145,58,NA,
                 NA,NA,168,NA,NA,  NA,NA,67,NA,NA,NA)
)

static_covariates <- data.frame(
  patient_id    = 1:5,
  SEX           = c("M","F","M","F","M"),
  ETHNICITY     = c("White","Asian","White","White","Black"),
  IMD           = c(3L,1L,5L,2L,4L),
  YEAR_OF_BIRTH = c(1965L,1972L,1955L,1960L,1968L)
)

# Labelled outcomes: patient 4 developed T2D, patient 5 developed CVD
targets <- data.frame(
  patient_id   = c(4L,  5L),
  target_event = c("T2D", "CVD"),
  target_age   = c(58.0, 51.2)
)

# ── 1. Validate inputs ────────────────────────────────────────────────────────
survivehr_validate_events(events)
survivehr_validate_static(static_covariates)
survivehr_validate_targets(targets)

# ── 2. Configure ──────────────────────────────────────────────────────────────
cfg <- survivehr_config(
  block_size = 64, n_layer = 2, n_head = 2, n_embd = 64,
  epochs = 10,     batch_size = 16,
  surv_layer = "competing-risk",
  time_scale = 1.0   # ages in years; use 1825.0 for DAYS_SINCE_BIRTH
)

# ── 3. Pre-train the backbone ─────────────────────────────────────────────────
pt_model <- survivehr_pretrain(events, static_covariates, cfg)
survivehr_save_model(pt_model, "backbone.pt")

# ── 4. Fine-tune on labelled outcomes ─────────────────────────────────────────
# Remove CVD from context to prevent data leakage for patient 5
ft_model <- survivehr_finetune(
  events            = events[events$event != "CVD", ],
  targets           = targets,
  outcomes          = c("T2D", "CVD"),
  risk_model        = "competing-risk",
  static_covariates = static_covariates,
  config            = cfg,
  pretrained_model  = pt_model
)

# ── 5. Predict ────────────────────────────────────────────────────────────────
preds <- survivehr_predict(ft_model, events, static_covariates)
# Columns: patient_id, T2D_cdf_last, T2D_auc, CVD_cdf_last, CVD_auc
```

---

## Key features

| Feature | Detail |
|---|---|
| **No manual Python setup** | Backend is vendored inside the package; `survivehr_setup()` handles everything. |
| **Competing & single risk** | `surv_layer = "competing-risk"` or `"single-risk"`. |
| **Leakage-free fine-tuning** | Context is restricted to events before the target age. |
| **Flexible static covariates** | Any column names; any mix of categorical and numeric. |
| **Continuous readings** | Record measurements (BP, HbA1c, BMI) in the `value` column. |
| **FastEHR compatibility** | Accepts `PATIENT_ID`, `EVENT`, `DAYS_SINCE_BIRTH` aliases. |
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
