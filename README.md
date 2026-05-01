# survivehrR

📦 **Website**: <https://pm-cardoso.github.io/RSurvivEHR/>  
🐛 **Issues / source**: <https://github.com/PM-Cardoso/RSurvivEHR>

`survivehrR` is a standalone R package that lets you run the **SurvivEHR** transformer survival
model directly from R data frames. It requires no FastEHR installation; all Python model code is
vendored inside the package.

---

## Contents

1. [Installation](#installation)
2. [Input format](#input-format)
3. [Special tokens — where PAD and UNK appear](#special-tokens)
4. [Age normalisation and `time_scale`](#age-normalisation)
5. [Value normalisation](#value-normalisation)
6. [Preparing fine-tune context (avoiding leakage)](#preparing-fine-tune-context)
7. [Static covariates encoding](#static-covariates-encoding)
8. [Full worked example](#full-worked-example)
9. [Configuration reference](#configuration-reference)
10. [FastEHR column aliases](#fastehr-column-aliases)

---

## Installation

Install from GitHub using `remotes` or `pak`:

```r
# install.packages("remotes")
remotes::install_github("PM-Cardoso/RSurvivEHR")
```

```r
# install.packages("pak")
pak::pkg_install("PM-Cardoso/RSurvivEHR")
```

After installing, set up the Python backend (one-time):

```r
library(survivehrR)
survivehr_setup()
```

---

## Input format

| Table | Required columns | Optional |
|---|---|---|
| **events** | `patient_id`, `event`, `age` | `value` |
| **static_covariates** | `patient_id` + any covariate columns | — |
| **targets** (fine-tune) | `patient_id`, `target_event`, `target_age` | `target_value` |

### Column naming rules

| Table | Column names | Notes |
|---|---|---|
| **events** | **Fixed** | Must be exactly `patient_id`, `event`, `age`, `value`. FastEHR UPPER_CASE aliases are also accepted (see [FastEHR column aliases](#fastehr-column-aliases)). |
| **targets** | **Fixed** | Must be exactly `patient_id`, `target_event`, `target_age`, `target_value`. FastEHR aliases accepted. |
| **static_covariates** | **Completely free** (except `patient_id`) | Name covariate columns whatever you like — `SEX`, `imd`, `YearOfBirth`, etc. The backend encodes whatever it finds. The only rule is that the **same column names must be used consistently** across pretrain, fine-tune, and prediction. |
