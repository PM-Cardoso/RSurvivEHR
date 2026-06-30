from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# NumPy 2.0 removed np.trapz in favour of np.trapezoid.
# Patch once here so all downstream code (including vendored modules) works
# on both NumPy 1.x and 2.x.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]


THIS_FILE = Path(__file__).resolve()
PKG_PYTHON_ROOT = THIS_FILE.parent
if str(PKG_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_PYTHON_ROOT))

# ---------------------------------------------------------------------------
# Ensure wandb is importable even when not installed.
# desurv.py has a bare top-level `import wandb`; this stub prevents failures.
# ---------------------------------------------------------------------------
import types as _types
if "wandb" not in sys.modules:
    _wandb_stub = _types.ModuleType("wandb")
    _wandb_stub.log   = lambda *a, **kw: None
    _wandb_stub.init  = lambda *a, **kw: None
    _wandb_stub.watch = lambda *a, **kw: None
    sys.modules["wandb"] = _wandb_stub

from SurvivEHR.experiments import CausalExperiment, FineTuneExperiment
from iec_metrics import compute_iec_single, compute_iec_batch, compute_iec_stratified


def _as_list_or_none(x):
    """Normalise reticulate/Python round-trip values to list or None.
    
    When R data.frames are passed through reticulate to Python and then
    back to R, column names may arrive as strings instead of lists.
    This normalises them back to proper lists.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except TypeError:
        return [x]


def _device_from_config(config: Dict[str, Any]) -> torch.device:
    choice = str(config.get("device", "auto")).lower()
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_events(events_df: pd.DataFrame) -> pd.DataFrame:
    events = events_df.copy()

    alias_map = {
        "PATIENT_ID": "patient_id",
        "EVENT": "event",
        "DAYS_SINCE_BIRTH": "age",
        "VALUE": "value",
    }
    for src, dst in alias_map.items():
        if dst not in events.columns and src in events.columns:
            events = events.rename(columns={src: dst})

    req = ["patient_id", "event", "age"]
    missing = [c for c in req if c not in events.columns]
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")

    if "value" not in events.columns:
        events["value"] = np.nan

    events = events.sort_values(["patient_id", "age"]).reset_index(drop=True)
    events["event"] = events["event"].astype(str)
    events["age"] = events["age"].astype(float)
    events["value"] = events["value"].astype(float)
    return events


def _encode_static(
    static_df: Optional[pd.DataFrame],
    patient_ids: List[Any],
    reference_raw_cols: Optional[List[str]] = None,
    reference_encoded_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Encode static covariates into a float32 matrix.

    Returns (encoded_matrix, raw_col_names, encoded_col_names).

    Parameters
    ----------
    reference_raw_cols:
        Raw feature column names recorded at training time (before one-hot
        expansion).  At prediction time these are validated strictly: the
        caller must supply exactly the same column names or a ``ValueError``
        is raised.  This catches typos and missing features early.
    reference_encoded_cols:
        One-hot-expanded column names recorded at training time.  The output
        matrix is always reindexed to this list, filling any one-hot columns
        that are absent from the current batch with 0.0.  This is correct
        behaviour — a category not present in a small prediction batch simply
        has no patients with that value.
    """
    # Normalise reticulate round-trip artefacts: single strings -> lists
    reference_raw_cols = _as_list_or_none(reference_raw_cols)
    reference_encoded_cols = _as_list_or_none(reference_encoded_cols)

    n = len(patient_ids)

    if static_df is None:
        if reference_raw_cols:
            raise ValueError(
                "The model was trained with static covariates "
                f"({reference_raw_cols}) but none were supplied for prediction. "
                "Pass the same static_covariates data frame used at training time."
            )
        cols = reference_encoded_cols if reference_encoded_cols else ["intercept"]
        return np.zeros((n, len(cols)), dtype=np.float32), [], cols

    df = static_df.copy()
    if "patient_id" not in df.columns and "PATIENT_ID" in df.columns:
        df = df.rename(columns={"PATIENT_ID": "patient_id"})

    if "patient_id" not in df.columns:
        raise ValueError("`static_df` must contain patient_id")

    feature_cols = [c for c in df.columns if c != "patient_id"]
    mapped = df.set_index("patient_id").reindex(patient_ids)

    # ── Raw column name validation ────────────────────────────────────────────
    if reference_raw_cols is not None:
        missing_raw = [c for c in reference_raw_cols if c not in feature_cols]
        extra_raw   = [c for c in feature_cols if c not in reference_raw_cols]
        if missing_raw or extra_raw:
            msg = (
                "Static covariate column names do not match what the model was "
                "trained on.\n"
                f"  Expected columns : {reference_raw_cols}\n"
                f"  Supplied columns : {feature_cols}\n"
            )
            if missing_raw:
                msg += f"  Missing          : {missing_raw}\n"
            if extra_raw:
                msg += f"  Unexpected       : {extra_raw}\n"
            msg += (
                "Tip: pass exactly the same column names used at training time. "
                "Category values do not need to match — unseen categories are "
                "handled automatically."
            )
            raise ValueError(msg)

    if len(feature_cols) == 0:
        cols = reference_encoded_cols if reference_encoded_cols else ["intercept"]
        return np.zeros((n, len(cols)), dtype=np.float32), [], cols

    encoded_parts = []
    encoded_names: List[str] = []
    for c in feature_cols:
        col = mapped[c]
        numeric = pd.to_numeric(col, errors="coerce")

        is_largely_numeric = numeric.notna().mean() >= 0.8
        if is_largely_numeric:
            encoded_parts.append(numeric.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1))
            encoded_names.append(c)
        else:
            dummies = pd.get_dummies(col.fillna("UNKNOWN").astype(str), prefix=c)
            encoded_parts.append(dummies.to_numpy(dtype=np.float32))
            encoded_names.extend(dummies.columns.tolist())

    if not encoded_parts:
        cols = reference_encoded_cols if reference_encoded_cols else ["intercept"]
        return np.zeros((n, len(cols)), dtype=np.float32), list(feature_cols), cols

    encoded = np.concatenate(encoded_parts, axis=1)
    names_out = encoded_names if encoded_names else ["intercept"]

    # ── Align to training-time encoded columns ────────────────────────────────
    # Reindex fills one-hot columns absent from this batch with 0.0 — correct
    # because no patient in the batch belongs to that category.
    if reference_encoded_cols is not None:
        result_df = pd.DataFrame(encoded, columns=names_out)
        result_df = result_df.reindex(columns=reference_encoded_cols, fill_value=0.0)
        encoded = result_df.to_numpy(dtype=np.float32)
        names_out = list(reference_encoded_cols)

    return encoded.astype(np.float32), list(feature_cols), names_out


def _clean_targets(targets_df: pd.DataFrame) -> pd.DataFrame:
    targets = targets_df.copy()

    alias_map = {
        "PATIENT_ID": "patient_id",
        "TARGET_EVENT": "target_event",
        "TARGET_AGE": "target_age",
        "TARGET_VALUE": "target_value",
        "EVENT": "target_event",
        "DAYS_SINCE_BIRTH": "target_age",
        "VALUE": "target_value",
    }
    for src, dst in alias_map.items():
        if dst not in targets.columns and src in targets.columns:
            targets = targets.rename(columns={src: dst})

    req = ["patient_id", "target_event", "target_age"]
    missing = [c for c in req if c not in targets.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    if "target_value" not in targets.columns:
        targets["target_value"] = np.nan

    targets["target_event"] = targets["target_event"].astype(str)
    targets["target_age"] = pd.to_numeric(targets["target_age"], errors="coerce")
    targets["target_value"] = pd.to_numeric(targets["target_value"], errors="coerce")
    targets = targets.dropna(subset=["target_age"]).reset_index(drop=True)

    return targets


def _fit_value_standardization(events: pd.DataFrame) -> Dict[str, Any]:
    """Fit per-event z-score statistics on non-missing event values.

    Parameters
    ----------
    events:
        Cleaned events frame containing at least ``event`` and ``value``.

    Returns
    -------
    dict
        Standardization payload persisted in model bundles.
    """
    out: Dict[str, Any] = {
        "method": "zscore_by_event",
        "stats": {},
        "fitted": False,
    }

    if "event" not in events.columns or "value" not in events.columns:
        return out

    usable = events.loc[events["value"].notna(), ["event", "value"]].copy()
    if usable.empty:
        return out

    stats: Dict[str, Dict[str, Any]] = {}
    for event_name, group in usable.groupby("event", sort=False):
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue

        mean = float(np.mean(values))
        sd_raw = float(np.std(values, ddof=0))
        is_constant = (not np.isfinite(sd_raw)) or sd_raw <= 1e-12
        sd = 1.0 if is_constant else sd_raw

        stats[str(event_name)] = {
            "mean": mean,
            "sd": float(sd),
            "n": int(values.size),
            "is_constant": bool(is_constant),
        }

    out["stats"] = stats
    out["fitted"] = len(stats) > 0
    return out


def _normalise_value_standardization(value_standardization: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise and validate persisted value standardization metadata."""
    base = {
        "method": "zscore_by_event",
        "stats": {},
        "fitted": False,
    }
    if value_standardization is None:
        return base

    payload = dict(value_standardization)
    payload_method = str(payload.get("method", "zscore_by_event"))
    stats_in = payload.get("stats", {})
    stats_in = dict(stats_in) if isinstance(stats_in, dict) else {}

    stats_out: Dict[str, Dict[str, Any]] = {}
    for event_name, st in stats_in.items():
        if not isinstance(st, dict):
            continue
        mean = st.get("mean", 0.0)
        sd = st.get("sd", 1.0)
        n = st.get("n", 0)
        try:
            mean_f = float(mean)
            sd_f = float(sd)
            n_i = int(n)
        except Exception:
            continue

        if (not np.isfinite(sd_f)) or sd_f <= 1e-12:
            sd_f = 1.0

        stats_out[str(event_name)] = {
            "mean": mean_f,
            "sd": sd_f,
            "n": n_i,
            "is_constant": bool(st.get("is_constant", False)),
        }

    return {
        "method": payload_method,
        "stats": stats_out,
        "fitted": bool(len(stats_out) > 0),
    }


def _standardize_events_df(events: pd.DataFrame,
                           value_standardization: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply per-event z-score standardization to event ``value`` column."""
    out = events.copy()
    if "value" not in out.columns or "event" not in out.columns:
        return out

    vs = _normalise_value_standardization(value_standardization)
    stats = vs.get("stats", {})
    if not stats:
        return out

    for event_name, st in stats.items():
        mask = (out["event"] == event_name) & out["value"].notna()
        if not bool(mask.any()):
            continue
        mean = float(st["mean"])
        sd = float(st["sd"])
        out.loc[mask, "value"] = (out.loc[mask, "value"].astype(float) - mean) / sd

    return out


def _standardize_targets_df(targets: pd.DataFrame,
                            value_standardization: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply per-event z-score standardization to target ``target_value`` column."""
    out = targets.copy()
    if "target_value" not in out.columns or "target_event" not in out.columns:
        return out

    vs = _normalise_value_standardization(value_standardization)
    stats = vs.get("stats", {})
    if not stats:
        return out

    for event_name, st in stats.items():
        mask = (out["target_event"] == event_name) & out["target_value"].notna()
        if not bool(mask.any()):
            continue
        mean = float(st["mean"])
        sd = float(st["sd"])
        out.loc[mask, "target_value"] = (out.loc[mask, "target_value"].astype(float) - mean) / sd

    return out


def _inverse_standardized_value(value: float,
                                event_name: Optional[str],
                                value_standardization: Optional[Dict[str, Any]]) -> float:
    """Map a standardized value prediction back to the original event scale."""
    if not np.isfinite(value):
        return float(value)
    if event_name is None:
        return float(value)

    vs = _normalise_value_standardization(value_standardization)
    stats = vs.get("stats", {})
    st = stats.get(str(event_name), None)
    if st is None:
        return float(value)

    mean = float(st["mean"])
    sd = float(st["sd"])
    return float((value * sd) + mean)


def _inverse_standardized_sd(sd_value: float,
                             event_name: Optional[str],
                             value_standardization: Optional[Dict[str, Any]]) -> float:
    """Map a standardized SD prediction back to the original event scale."""
    if not np.isfinite(sd_value):
        return float(sd_value)
    if event_name is None:
        return float(sd_value)

    vs = _normalise_value_standardization(value_standardization)
    stats = vs.get("stats", {})
    st = stats.get(str(event_name), None)
    if st is None:
        return float(sd_value)

    scale = float(st["sd"])
    return float(sd_value * scale)


def _padded_sequence(values: List[float], block_size: int, pad_value: float) -> np.ndarray:
    arr = np.full(block_size, pad_value, dtype=np.float32)
    take = values[-block_size:]
    arr[: len(take)] = np.array(take, dtype=np.float32)
    return arr


def _padded_tokens(values: List[int], block_size: int) -> np.ndarray:
    arr = np.zeros(block_size, dtype=np.int64)
    take = values[-block_size:]
    arr[: len(take)] = np.array(take, dtype=np.int64)
    return arr


def _attention(length: int, block_size: int) -> np.ndarray:
    out = np.zeros(block_size, dtype=np.bool_)
    out[: min(length, block_size)] = True
    return out


@dataclass
class BuiltData:
    patient_ids: List[Any]
    tokens: np.ndarray
    ages: np.ndarray
    values: np.ndarray
    attention_mask: np.ndarray
    static_covariates: np.ndarray
    static_raw_cols: List[str]      # raw column names before one-hot expansion — used to validate prediction-time column names
    static_col_names: List[str]     # encoded column names after one-hot expansion — used to align prediction-time tensors
    event_vocab: Dict[str, int]
    inv_vocab: Dict[int, str]
    time_scale: float            # backbone age normalisation divisor (raw_age / time_scale → model input)
                                 # always inherited from the pretrained bundle; NOT the prediction window
    token_policy: Dict[str, Any]


def _token_policy_from_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = dict(config or {})
    include_unk = bool(config.get("include_unk", True))
    include_cls_sep = bool(config.get("include_cls_sep", False))
    return {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "cls_token": "<CLS>",
        "sep_token": "<SEP>",
        "include_unk": include_unk,
        "include_cls_sep": include_cls_sep,
    }


def _build_vocab_with_policy(events: pd.DataFrame,
                             policy: Dict[str, Any],
                             event_vocab: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    pad_token = policy["pad_token"]
    unk_token = policy["unk_token"]
    cls_token = policy["cls_token"]
    sep_token = policy["sep_token"]

    if event_vocab is not None:
        vocab = {str(k): int(v) for k, v in dict(event_vocab).items()}
        if pad_token not in vocab:
            raise ValueError(f"Provided event_vocab must include {pad_token}")
        if vocab[pad_token] != 0:
            raise ValueError(f"{pad_token} must have index 0")

        if policy["include_unk"] and unk_token not in vocab:
            vocab[unk_token] = max(vocab.values()) + 1
        if policy["include_cls_sep"]:
            if cls_token not in vocab:
                vocab[cls_token] = max(vocab.values()) + 1
            if sep_token not in vocab:
                vocab[sep_token] = max(vocab.values()) + 1
        return vocab

    vocab: Dict[str, int] = {pad_token: 0}
    next_idx = 1

    if policy["include_unk"]:
        vocab[unk_token] = next_idx
        next_idx += 1

    if policy["include_cls_sep"]:
        vocab[cls_token] = next_idx
        next_idx += 1
        vocab[sep_token] = next_idx
        next_idx += 1

    # Sort by descending frequency so the most-common events receive the
    # smallest integer token IDs; ties are broken alphabetically for stability.
    event_counts = events["event"].dropna().value_counts()
    reserved = set(vocab.keys())
    for event in event_counts.index.tolist():
        if event in reserved:
            continue
        vocab[event] = next_idx
        next_idx += 1

    return vocab


def _build_context_data(events_df: pd.DataFrame,
                        static_df: Optional[pd.DataFrame],
                        block_size: int,
                        event_vocab: Optional[Dict[str, int]] = None,
                        token_policy: Optional[Dict[str, Any]] = None,
                        time_scale: float = 1.0,
                        reference_static_cols: Optional[List[str]] = None,
                        reference_static_encoded_cols: Optional[List[str]] = None) -> BuiltData:
    """
    Build padded tensor arrays from raw event data.

    Age normalisation
    -----------------
    Each event age is divided by *time_scale* to give a dimensionless value
    that the backbone transformer sees as input:

        age_norm = raw_age / time_scale

    ``time_scale`` controls **only** the backbone input scale and is always
    inherited from the pretrained bundle.  The prediction window (ODE grid
    calibration) is controlled separately by ``outcome_horizon`` in
    ``FineTuneDataset`` — keeping the two concepts independent.

    Examples:
    - ``time_scale=1.0`` with ages in years  → backbone sees age as a plain year value
    - ``time_scale=365.25`` with ages in days → backbone sees normalised fraction

    The value is stored inside every model bundle and retrieved automatically
    by ``predict_next_events`` — it does not need to be supplied at inference.

    Token layout per patient (before padding)
    -----------------------------------------
    [CLS?,  event_1, event_2, …, event_n,  SEP?]

    After padding to ``block_size`` the rightmost positions are filled with
    token index 0 (``<PAD>``) and their ``attention_mask`` bits are ``False``.
    Unknown events are mapped to ``<UNK>`` (index 1) when ``include_unk=True``.
    """
    events = _clean_events(events_df)
    policy = dict(token_policy or _token_policy_from_config())
    vocab = _build_vocab_with_policy(events, policy, event_vocab)
    inv_vocab = {v: k for k, v in vocab.items()}

    unk_id = vocab.get(policy["unk_token"], None)
    cls_id = vocab.get(policy["cls_token"], None)
    sep_id = vocab.get(policy["sep_token"], None)

    if time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")

    patient_ids = []
    token_rows = []
    age_rows = []
    value_rows = []
    mask_rows = []

    for patient_id, group in events.groupby("patient_id", sort=False):
        token_ids = []
        for e in group["event"].tolist():
            if e in vocab:
                token_ids.append(vocab[e])
            elif unk_id is not None:
                token_ids.append(unk_id)
            else:
                raise ValueError(
                    f"Found unknown event '{e}' but include_unk is disabled "
                    f"and no matching token exists in vocab."
                )
        ages = group["age"].tolist()
        values = group["value"].tolist()

        if policy.get("include_cls_sep", False):
            if cls_id is None or sep_id is None:
                raise ValueError("Token policy requested CLS/SEP but vocab does not contain these tokens.")
            if len(ages) == 0:
                continue
            token_ids = [cls_id] + token_ids + [sep_id]
            ages = [ages[0]] + ages + [ages[-1]]
            values = [np.nan] + values + [np.nan]

        if len(token_ids) < 2:
            continue

        # Global absolute normalisation: divide by time_scale.
        # Padding positions receive 0.0 (age 0 relative to time_scale).
        # The attention_mask prevents the model from attending to those slots.
        ages_norm = [float(a) / time_scale for a in ages]

        patient_ids.append(patient_id)
        token_rows.append(_padded_tokens(token_ids, block_size))
        age_rows.append(_padded_sequence(ages_norm, block_size, 0.0))
        value_rows.append(_padded_sequence(values, block_size, np.nan))
        mask_rows.append(_attention(len(token_ids), block_size))

    if len(patient_ids) == 0:
        raise ValueError("No patients with at least 2 events after preprocessing.")

    static_covariates, static_raw_cols, static_col_names = _encode_static(
        static_df, patient_ids,
        reference_raw_cols=reference_static_cols,
        reference_encoded_cols=reference_static_encoded_cols,
    )

    return BuiltData(
        patient_ids=patient_ids,
        tokens=np.stack(token_rows),
        ages=np.stack(age_rows),
        values=np.stack(value_rows),
        attention_mask=np.stack(mask_rows),
        static_covariates=static_covariates,
        static_raw_cols=static_raw_cols,
        static_col_names=static_col_names,
        event_vocab=vocab,
        inv_vocab=inv_vocab,
        time_scale=float(time_scale),
        token_policy=policy,
    )


class PretrainDataset(Dataset):
    def __init__(self, built: BuiltData):
        self.built = built

    def __len__(self) -> int:
        return self.built.tokens.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "tokens": torch.tensor(self.built.tokens[idx], dtype=torch.long),
            "ages": torch.tensor(self.built.ages[idx], dtype=torch.float32),
            "values": torch.tensor(self.built.values[idx], dtype=torch.float32),
            "attention_mask": torch.tensor(self.built.attention_mask[idx], dtype=torch.bool),
            "static_covariates": torch.tensor(self.built.static_covariates[idx], dtype=torch.float32),
        }


class FineTuneDataset(Dataset):
    def __init__(self, built: BuiltData, targets_df: pd.DataFrame,
                 outcome_horizon: Optional[float] = None):
        """
        Parameters
        ----------
        built            : BuiltData produced by _build_context_data().
        targets_df       : DataFrame with target event labels.
        outcome_horizon  : Length of the ODE prediction window in the same raw
                           age units as the ``age`` column (e.g. 5.0 for a
                           5-year risk window when ages are in years).
                           Defaults to ``built.time_scale`` for backward
                           compatibility, but can differ freely from it.
                           Examples:
                             time_scale=1.0, outcome_horizon=5.0
                               → backbone sees year-by-year ages;
                                 ODE CDF covers 0–5 years.
                             time_scale=1.0, outcome_horizon=1.0
                               → both backbone and ODE use 1-year scale.
        """
        targets = _clean_targets(targets_df)

        by_patient = targets.groupby("patient_id", sort=False).last().reset_index()

        row_map = {pid: i for i, pid in enumerate(built.patient_ids)}
        selected_rows = []
        target_tokens = []
        target_age_deltas = []
        target_values = []

        # backbone_time_scale: used ONLY to undo the age normalisation applied
        #   in _build_context_data so we can recover a raw age from age_norm.
        backbone_time_scale = float(built.time_scale)
        # outcome_horizon: controls the ODE prediction window.  delta_norm=1.0
        #   corresponds to outcome_horizon raw age units from the last event.
        if outcome_horizon is None:
            outcome_horizon = backbone_time_scale
        outcome_horizon = float(outcome_horizon)
        if outcome_horizon <= 0:
            raise ValueError(f"outcome_horizon must be positive, got {outcome_horizon}")

        for _, row in by_patient.iterrows():
            pid = row["patient_id"]
            if pid not in row_map:
                continue

            event_name = row["target_event"]
            if event_name not in built.event_vocab:
                continue

            i = row_map[pid]
            last_attended = int(np.sum(built.attention_mask[i]) - 1)
            context_age_norm = float(built.ages[i][last_attended])

            # Recover raw age by undoing the backbone normalisation.
            context_age_raw = context_age_norm * backbone_time_scale

            # Delta in raw age units; normalise by outcome_horizon so that
            # delta_norm=1.0 maps to the end of the prediction window.
            delta = max(float(row["target_age"]) - context_age_raw, 0.0)
            delta_norm = delta / outcome_horizon

            selected_rows.append(i)
            target_tokens.append(int(built.event_vocab[event_name]))
            target_age_deltas.append(float(delta_norm))
            target_values.append(float(row["target_value"]))

        if len(selected_rows) == 0:
            raise ValueError("No matching patient targets found with known outcome tokens.")

        self.tokens = built.tokens[selected_rows]
        self.ages = built.ages[selected_rows]
        self.values = built.values[selected_rows]
        self.attention_mask = built.attention_mask[selected_rows]
        self.static_covariates = built.static_covariates[selected_rows]

        self.target_tokens = np.array(target_tokens, dtype=np.int64)
        self.target_age_deltas = np.array(target_age_deltas, dtype=np.float32)
        self.target_values = np.array(target_values, dtype=np.float32)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.long),
            "ages": torch.tensor(self.ages[idx], dtype=torch.float32),
            "values": torch.tensor(self.values[idx], dtype=torch.float32),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.float32),
            "static_covariates": torch.tensor(self.static_covariates[idx], dtype=torch.float32),
            "target_token": torch.tensor(self.target_tokens[idx], dtype=torch.long),
            "target_age_delta": torch.tensor(self.target_age_deltas[idx], dtype=torch.float32),
            "target_value": torch.tensor(self.target_values[idx], dtype=torch.float32),
        }


def _build_cfg(config: Dict[str, Any], vocab_size: int, fine_tune: bool = False) -> Any:
    block_size = int(config.get("block_size", 128))
    n_layer = int(config.get("n_layer", 4))
    n_head = int(config.get("n_head", 4))
    n_embd = int(config.get("n_embd", 256))
    dropout = float(config.get("dropout", 0.0))
    surv_layer = str(config.get("surv_layer", "competing-risk"))
    surv_weight = float(config.get("surv_weight", 1.0))
    value_weight = float(config.get("value_weight", 0.0))

    cfg = {
        "is_decoder": True,
        "data": {"batch_size": int(config.get("batch_size", 16))},
        "experiment": {
            "log": False,
            "verbose": False,
            "run_id": "survivehrR",
            "fine_tune_id": "survivehrR_ft",
            "ckpt_dir": "./outputs/checkpoints/",
            "seed": 1337,
        },
        "optim": {
            "learning_rate": float(config.get("learning_rate", 3e-4)),
            "num_epochs": int(config.get("epochs", 1)),
            "scheduler_warmup": False,
            "scheduler": "cawarmrestarts",
            "scheduler_periods": 100,
            "learning_rate_decay": 0.9,
            "early_stop": False,
            "early_stop_patience": 10,
        },
        "transformer": {
            "block_type": "Neo",
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "layer_norm_bias": False,
            "attention_type": "global",
            "bias": True,
            "dropout": dropout,
            "attention_dropout": dropout,
            "resid_dropout": dropout,
            "private_heads": 0,
        },
        "head": {
            "SurvLayer": surv_layer,
            "surv_weight": surv_weight,
            "value_weight": value_weight,
            "tokens_for_univariate_regression": [],
        },
        "fine_tuning": {
            "fine_tune_outcomes": None,
            "custom_outcome_method": {"_target_": None},
            "custom_stratification_method": {"_target_": None},
            "use_callbacks": {
                "hidden_embedding": {"num_batches": 0, "mask_static": False, "mask_value": False},
                "performance_metrics": False,
                "rmst": False,
            },
            "compression_layer": False,
            "llrd": None,
            "PEFT": {"method": None, "adapter_dim": 8},
            "backbone": {"linear_probe_epochs": 0, "unfreeze_top_k": None},
            "head": {
                "surv_weight": surv_weight,
                "value_weight": value_weight,
                "learning_rate": float(config.get("learning_rate", 3e-4)),
            },
        },
    }

    return OmegaConf.create(cfg)


def _set_measurement_tokens(cfg: Any, values: np.ndarray, tokens: np.ndarray) -> None:
    token_ids = []
    observed = ~np.isnan(values)
    if np.any(observed):
        token_ids = np.unique(tokens[observed]).astype(int).tolist()
        token_ids = [t for t in token_ids if t != 0]
    cfg.head.tokens_for_univariate_regression = token_ids


def _run_train_loop(model: torch.nn.Module,
                    loader: DataLoader,
                    learning_rate: float,
                    epochs: int,
                    device: torch.device,
                    label: str = "Training"):
    """Returns (history: List[float], training_duration_secs: float)."""
    import time
    import math

    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    n_epochs  = int(epochs)
    n_batches = len(loader)
    history: List[float] = []
    BAR_WIDTH  = 20

    # Build a human-readable device string, e.g. "cuda:0 (Tesla V100-SXM2-16GB)" or "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        device_str = f"{device} ({gpu_name})"
    else:
        device_str = str(device)

    print(f"[RSurvivEHR] {label}: {n_epochs} epoch(s), "
          f"{n_batches} batch(es)/epoch, device={device_str}", flush=True)

    wall_start = time.time()
    for epoch in range(n_epochs):
        running   = 0.0
        steps     = 0
        t_start   = time.time()

        for batch_idx, batch in enumerate(loader):
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(device)

            optim.zero_grad()
            _, losses, _ = model(batch)
            loss = losses["loss"]
            loss.backward()
            optim.step()

            running += float(loss.detach().cpu().item())
            steps   += 1

            # ---- inline progress bar ----
            elapsed   = time.time() - t_start
            frac      = steps / n_batches
            filled    = int(BAR_WIDTH * frac)
            bar       = "=" * filled + "-" * (BAR_WIDTH - filled)
            avg_loss  = running / steps
            eta       = (elapsed / frac - elapsed) if frac > 0 else 0.0
            eta_str   = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
            print(
                f"\r  Epoch {epoch + 1}/{n_epochs}  "
                f"[{bar}] {steps}/{n_batches}  "
                f"loss={avg_loss:.4f}  ETA {eta_str}   ",
                end="",
                flush=True,
            )

        epoch_loss = running / max(steps, 1)
        history.append(epoch_loss)
        elapsed = time.time() - t_start
        # Overwrite the progress line with the final epoch summary
        print(
            f"\r  Epoch {epoch + 1}/{n_epochs}  "
            f"[{'=' * BAR_WIDTH}] {n_batches}/{n_batches}  "
            f"loss={epoch_loss:.4f}  {elapsed:.1f}s          ",
            flush=True,
        )

    total_secs = time.time() - wall_start
    return history, total_secs


def train_pretrain_model(events_df: pd.DataFrame,
                         static_df: Optional[pd.DataFrame] = None,
                         config: Optional[Dict[str, Any]] = None,
                         event_vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    config = dict(config or {})
    device = _device_from_config(config)
    token_policy = _token_policy_from_config(config)
    time_scale = float(config.get("time_scale", 1.0))

    events_clean = _clean_events(events_df)
    value_standardization = _fit_value_standardization(events_clean)
    events_std = _standardize_events_df(events_clean, value_standardization)

    built = _build_context_data(events_df=events_std,
                                static_df=static_df,
                                block_size=int(config.get("block_size", 128)),
                                event_vocab=event_vocab,
                                token_policy=token_policy,
                                time_scale=time_scale)

    cfg = _build_cfg(config=config, vocab_size=len(built.event_vocab), fine_tune=False)
    _set_measurement_tokens(cfg, built.values, built.tokens)

    ds = PretrainDataset(built)
    loader = DataLoader(ds, batch_size=int(config.get("batch_size", 16)), shuffle=True)

    model = CausalExperiment(cfg=cfg, vocab_size=len(built.event_vocab))
    history, training_duration_secs = _run_train_loop(
        model=model,
        loader=loader,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        epochs=int(config.get("epochs", 1)),
        device=device,
        label="Pre-training",
    )

    return {
        "type": "pretrain",
        "model": model,
        "cfg": cfg,
        "event_vocab": built.event_vocab,
        "inv_vocab": built.inv_vocab,
        "block_size": int(config.get("block_size", 128)),
        "time_scale": built.time_scale,
        "value_standardization": value_standardization,
        "token_policy": token_policy,
        "static_raw_cols": built.static_raw_cols,
        "static_col_names": built.static_col_names,
        "history": history,
        "training_duration_secs": training_duration_secs,
        "device": str(device),
    }


def train_finetune_model(events_df: pd.DataFrame,
                         targets_df: pd.DataFrame,
                         outcomes: Iterable[str],
                         risk_model: str = "competing-risk",
                         static_df: Optional[pd.DataFrame] = None,
                         config: Optional[Dict[str, Any]] = None,
                         pretrained_bundle: Optional[Dict[str, Any]] = None,
                         event_vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    config = dict(config or {})
    device = _device_from_config(config)
    token_policy = _token_policy_from_config(config)
    time_scale = float(config.get("time_scale", 1.0))
    value_standardization = None

    # outcome_horizon: the ODE prediction window length (raw age units).
    # Independent of time_scale — can differ freely.
    # If not supplied, defaults to time_scale for backward compatibility.
    outcome_horizon_raw = config.get("outcome_horizon", None)
    outcome_horizon = float(outcome_horizon_raw) if outcome_horizon_raw is not None else None

    if pretrained_bundle is not None:
        event_vocab = dict(pretrained_bundle["event_vocab"])
        token_policy = dict(pretrained_bundle.get("token_policy", token_policy))
        # Always inherit time_scale from the pretrained bundle so context
        # normalisation is identical between pretrain and fine-tune.
        time_scale = float(pretrained_bundle.get("time_scale", time_scale))
        value_standardization = _normalise_value_standardization(pretrained_bundle.get("value_standardization", None))

    events_clean = _clean_events(events_df)
    if value_standardization is None:
        value_standardization = _fit_value_standardization(events_clean)
    events_std = _standardize_events_df(events_clean, value_standardization)

    targets_clean = _clean_targets(targets_df)
    targets_std = _standardize_targets_df(targets_clean, value_standardization)

    # Resolve outcome_horizon now that time_scale is finalised.
    if outcome_horizon is None:
        outcome_horizon = time_scale

    built = _build_context_data(events_df=events_std,
                                static_df=static_df,
                                block_size=int(config.get("block_size", 128)),
                                event_vocab=event_vocab,
                                token_policy=token_policy,
                                time_scale=time_scale)

    cfg = _build_cfg(config=config, vocab_size=len(built.event_vocab), fine_tune=True)
    _set_measurement_tokens(cfg, built.values, built.tokens)

    outcomes = [str(o) for o in outcomes]

    # Validate outcome count against risk_model before any expensive work.
    _rm = risk_model.lower()
    if _rm in ("competing-risk", "cr") and len(outcomes) < 2:
        raise ValueError(
            f"risk_model='competing-risk' requires at least 2 outcome codes "
            f"(got {len(outcomes)}: {outcomes}). "
            "Use risk_model='single-risk' for a single endpoint."
        )
    if _rm in ("single-risk", "sr") and len(outcomes) != 1:
        raise ValueError(
            f"risk_model='single-risk' requires exactly 1 outcome code "
            f"(got {len(outcomes)}: {outcomes}). "
            "Use risk_model='competing-risk' for multiple endpoints."
        )

    reserved = {
        built.token_policy["pad_token"],
        built.token_policy["unk_token"],
        built.token_policy["cls_token"],
        built.token_policy["sep_token"],
    }
    reserved = {x for x in reserved if x in built.event_vocab}
    bad_reserved = [o for o in outcomes if o in reserved]
    if bad_reserved:
        raise ValueError(f"Outcomes cannot be special tokens: {bad_reserved}")

    missing_outcomes = [o for o in outcomes if o not in built.event_vocab]
    if missing_outcomes:
        # Outcome tokens that never appeared in the pre-training event
        # sequences (e.g. a terminal event like "CVD" used only as a
        # supervised label) are added to the vocabulary so the fine-tune
        # head can reference them.
        next_id = max(built.event_vocab.values()) + 1
        for tok in missing_outcomes:
            built.event_vocab[tok] = next_id
            built.inv_vocab[next_id] = tok
            next_id += 1
        # Rebuild cfg with the extended vocab size
        cfg = _build_cfg(config=config, vocab_size=len(built.event_vocab), fine_tune=True)
        _set_measurement_tokens(cfg, built.values, built.tokens)


    outcome_tokens = [built.event_vocab[o] for o in outcomes]

    ds = FineTuneDataset(built, targets_df=targets_std, outcome_horizon=outcome_horizon)
    loader = DataLoader(ds, batch_size=int(config.get("batch_size", 16)), shuffle=True)

    model = FineTuneExperiment(cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model, vocab_size=len(built.event_vocab))

    # Run a single dummy forward pass to force initialisation of any lazy
    # parameters before we attempt to copy pre-trained weights.
    model.eval()
    with torch.no_grad():
        _init_batch = next(iter(loader))
        try:
            model(_init_batch)
        except Exception:
            pass  # Even a failing forward initialises the parameter shapes

    if pretrained_bundle is not None:
        pretrain = pretrained_bundle["model"]
        # Copy pre-trained weights using .data (avoids lazy-module issues
        # that arise with load_state_dict when the target has uninitialised
        # parameters).  Shape mismatches (e.g. embedding enlarged by new
        # outcome tokens) are handled by copying the overlapping slice.
        src_params = {n: p for n, p in pretrain.model.named_parameters()}
        src_buffers = {n: b for n, b in pretrain.model.named_buffers()}
        for name, param in model.model.named_parameters():
            if name in src_params:
                src = src_params[name]
                if src.shape == param.shape:
                    param.data.copy_(src.data)
                else:
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, param.shape))
                    param.data[slices].copy_(src.data[slices])
        for name, buf in model.model.named_buffers():
            if name in src_buffers:
                src = src_buffers[name]
                if src.shape == buf.shape:
                    buf.copy_(src)
                else:
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, buf.shape))
                    buf[slices].copy_(src[slices])



    history, training_duration_secs = _run_train_loop(
        model=model,
        loader=loader,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        epochs=int(config.get("epochs", 1)),
        device=device,
        label="Fine-tuning",
    )

    return {
        "type": "finetune",
        "model": model,
        "cfg": cfg,
        "event_vocab": built.event_vocab,
        "inv_vocab": built.inv_vocab,
        "block_size": int(config.get("block_size", 128)),
        "time_scale": built.time_scale,
        "value_standardization": value_standardization,
        "outcome_horizon": outcome_horizon,
        "outcomes": list(outcomes),  # always a list — guards against reticulate round-trip
        "risk_model": risk_model,
        "token_policy": token_policy,
        "static_raw_cols": built.static_raw_cols,
        "static_col_names": built.static_col_names,
        "history": history,
        "training_duration_secs": training_duration_secs,
        "device": str(device),
    }


def predict_next_events(model_bundle: Dict[str, Any],
                        events_df: pd.DataFrame,
                        static_df: Optional[pd.DataFrame] = None,
                        max_new_tokens: int = 1,
                        eval_times: Optional[List[float]] = None) -> pd.DataFrame:
    model = model_bundle["model"]
    event_vocab = dict(model_bundle["event_vocab"])
    inv_vocab = {int(k): v for k, v in dict(model_bundle["inv_vocab"]).items()}
    value_standardization = _normalise_value_standardization(model_bundle.get("value_standardization", None))

    block_size = int(model_bundle["block_size"])
    events_clean = _clean_events(events_df)
    events_std = _standardize_events_df(events_clean, value_standardization)
    built = _build_context_data(events_df=events_std,
                                static_df=static_df,
                                block_size=block_size,
                                event_vocab=event_vocab,
                                token_policy=model_bundle.get("token_policy", None),
                                time_scale=float(model_bundle.get("time_scale", 1.0)),
                                reference_static_cols=model_bundle.get("static_raw_cols", None),
                                reference_static_encoded_cols=model_bundle.get("static_col_names", None))

    device = next(model.parameters()).device

    tokens = torch.tensor(built.tokens, dtype=torch.long, device=device)
    ages = torch.tensor(built.ages, dtype=torch.float32, device=device)
    values = torch.tensor(built.values, dtype=torch.float32, device=device)
    attention_mask = torch.tensor(built.attention_mask, dtype=torch.float32, device=device)
    static_cov = torch.tensor(built.static_covariates, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        if model_bundle.get("type") == "pretrain":
            base_model = model.model if hasattr(model, "model") else model
            # Record the original attended length for every patient so we can
            # identify which positions were newly generated after the call.
            orig_lens = [int(built.attention_mask[i].sum())
                         for i in range(len(built.patient_ids))]
            out_tokens, out_ages, out_values, out_mask = base_model.generate(
                tokens=tokens,
                ages=ages,
                values=values,
                static_covariates=static_cov,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                exceed_block_size=False,
            )

            time_scale = float(model_bundle.get("time_scale", 1.0))
            rows = []
            for i, pid in enumerate(built.patient_ids):
                attended = np.where(out_mask[i].detach().cpu().numpy() > 0)[0]
                if len(attended) == 0:
                    continue
                # Only the positions beyond the original sequence are new.
                new_positions = attended[orig_lens[i]:]
                for step_num, pos in enumerate(new_positions, start=1):
                    tok = int(out_tokens[i, pos].detach().cpu().item())
                    age_norm = float(out_ages[i, pos].detach().cpu().item())
                    value_val = float(out_values[i, pos].detach().cpu().item())
                    event_name = inv_vocab.get(tok, "<UNK>")
                    rows.append(
                        {
                            "patient_id": pid,
                            "step": step_num,
                            "generated_token": tok,
                            "generated_event": event_name,
                            "generated_age": age_norm * time_scale,
                            "generated_value": _inverse_standardized_value(
                                value=value_val,
                                event_name=event_name,
                                value_standardization=value_standardization,
                            ),
                        }
                    )
        else:
            batch = {
                "tokens": tokens,
                "ages": ages,
                "values": values,
                "attention_mask": attention_mask,
                "static_covariates": static_cov,
                "target_token": torch.zeros(tokens.shape[0], dtype=torch.long, device=device),
                "target_age_delta": torch.zeros(tokens.shape[0], dtype=torch.float32, device=device),
                "target_value": torch.full((tokens.shape[0],), torch.nan, dtype=torch.float32, device=device),
            }

            outputs, _, _ = model(batch, is_generation=True, return_loss=False, return_generation=True)
            cdfs = outputs["surv"]["surv_CDF"] or []

            # ── Validate and resolve eval_times to grid indices ───────────────
            # outcome_horizon is the ODE prediction window stored in the fine-tune
            # bundle.  Falls back to time_scale for bundles saved before this
            # feature was introduced.
            _ts = float(model_bundle.get("time_scale", 1.0))
            outcome_horizon = float(model_bundle.get("outcome_horizon", _ts))
            eval_time_indices: List[tuple] = []
            if eval_times is not None:
                n_grid = cdfs[0].shape[1] if cdfs else 1000
                for t in eval_times:
                    t = float(t)
                    if t <= 0:
                        raise ValueError(
                            f"eval_times must be positive, got {t}."
                        )
                    if t > outcome_horizon:
                        raise ValueError(
                            f"eval_time {t} exceeds the model's prediction window "
                            f"(outcome_horizon={outcome_horizon}). "
                            f"Supply times in the range (0, {outcome_horizon}]."
                        )
                    idx = min(int(round((t / outcome_horizon) * (n_grid - 1))), n_grid - 1)
                    eval_time_indices.append((t, idx))

            rows = []
            outcome_labels = model_bundle.get("outcomes") or [f"risk_{i+1}" for i in range(len(cdfs))]
            # Guard against reticulate round-trip: a single-element Python list
            # ["CVD"] becomes the string "CVD" after Python->R->Python conversion.
            if isinstance(outcome_labels, str):
                outcome_labels = [outcome_labels]
            for i, pid in enumerate(built.patient_ids):
                row = {"patient_id": pid}
                for j, cdf in enumerate(cdfs):
                    name = outcome_labels[j] if j < len(outcome_labels) else f"risk_{j+1}"
                    row[f"{name}_cdf_last"] = float(cdf[i, -1])
                    row[f"{name}_auc"] = float(np.trapz(cdf[i, :], dx=1.0 / max(cdf.shape[1] - 1, 1)))
                    for t, idx in eval_time_indices:
                        col = f"{name}_cdf_t{t:.4g}"
                        row[col] = float(cdf[i, idx])
                rows.append(row)

    return pd.DataFrame(rows)

def extract_pretrain_risk_scores(
    model_bundle: Dict[str, Any],
    events_df: pd.DataFrame,
    static_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Extract risk scores and observed next-event IDs for IEC evaluation.

    This function is for pretrain models only. It performs a forward pass through
    the full CausalExperiment wrapper and extracts, for each valid transition:

      1. risk scores for every possible next event
      2. the observed true next-event ID used by the model

    Returning the observed event IDs from Python avoids R/Python mismatch caused by
    reconstructing observed transitions separately from the raw event table.

    Returns
    -------
    dict
        risk_scores:
            List of 2D numpy arrays, one per patient.
            Each matrix has shape (n_transitions_i, n_events).

        observed_events:
            List of 1D numpy arrays, one per patient.
            Each vector has length n_transitions_i and contains the observed
            true next-event IDs for those risk-score rows.

        patient_ids:
            List of patient IDs corresponding to risk_scores/observed_events.

        event_vocab_table:
            pandas DataFrame with columns event and event_id. event_id is the
            1-indexed ID corresponding to the risk matrix columns.

        n_events:
            Number of predictable event types.
    """

    if model_bundle.get("type") != "pretrain":
        raise ValueError(
            "extract_pretrain_risk_scores() only supports pretrain models, "
            f"got type={model_bundle.get('type')}"
        )

    model = model_bundle["model"]
    event_vocab = dict(model_bundle["event_vocab"])

    value_standardization = _normalise_value_standardization(
        model_bundle.get("value_standardization", None)
    )

    block_size = int(model_bundle["block_size"])

    events_clean = _clean_events(events_df)
    events_std = _standardize_events_df(events_clean, value_standardization)

    built = _build_context_data(
        events_df=events_std,
        static_df=static_df,
        block_size=block_size,
        event_vocab=event_vocab,
        token_policy=model_bundle.get("token_policy", None),
        time_scale=float(model_bundle.get("time_scale", 1.0)),
        reference_static_cols=model_bundle.get("static_raw_cols", None),
        reference_static_encoded_cols=model_bundle.get("static_col_names", None)
    )

    device = next(model.parameters()).device

    tokens = torch.tensor(built.tokens, dtype=torch.long, device=device)
    ages = torch.tensor(built.ages, dtype=torch.float32, device=device)
    values = torch.tensor(built.values, dtype=torch.float32, device=device)
    attention_mask = torch.tensor(built.attention_mask, dtype=torch.bool, device=device)
    static_cov = torch.tensor(built.static_covariates, dtype=torch.float32, device=device)

    # Number of valid next-event transitions in the exact model input.
    transition_counts = [
        max(int(np.sum(mask)) - 1, 0)
        for mask in built.attention_mask
    ]
    expected_total_transitions = int(sum(transition_counts))

    model.eval()

    with torch.no_grad():
        batch = {
            "tokens": tokens,
            "ages": ages,
            "values": values,
            "attention_mask": attention_mask,
            "static_covariates": static_cov,
        }

        # Use the full CausalExperiment wrapper, not model.model.
        outputs, _, _ = model(
            batch,
            is_generation=False,
            return_loss=False,
            return_generation=True
        )

    if "surv" not in outputs:
        raise ValueError(
            "Model output does not contain 'surv'. "
            "extract_pretrain_risk_scores() currently expects survival outputs."
        )

    cdfs_all = outputs["surv"].get("surv_CDF", None)

    if cdfs_all is None:
        raise ValueError("No CDFs returned from model: outputs['surv']['surv_CDF'] is None.")

    try:
        n_events = len(cdfs_all)
    except TypeError:
        raise ValueError(
            f"Unexpected surv_CDF type: {type(cdfs_all)}. "
            "Expected a list/sequence of CDF arrays, one per event type."
        )

    if n_events == 0:
        raise ValueError("No CDFs returned from model: surv_CDF has length 0.")

    # Convert CDFs to numpy arrays.
    cdfs_all_np = []
    for cdf in cdfs_all:
        if isinstance(cdf, torch.Tensor):
            cdf = cdf.detach().cpu().numpy()
        else:
            cdf = np.asarray(cdf)
        cdfs_all_np.append(cdf)

    cdfs_all = cdfs_all_np

    # Observed event IDs from the same model output used to create risk scores.
    # This is the key fix: do not reconstruct observed events separately in R.
    if "k" not in outputs["surv"]:
        raise ValueError("Model survival output does not contain observed event IDs: outputs['surv']['k'].")

    k_all = outputs["surv"]["k"]

    def _flatten_observed_event_ids(x):
        """Convert nested torch/list/array event IDs to a flat NumPy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(-1)

        if isinstance(x, np.ndarray):
            if x.dtype == object:
                parts = []
                for item in x.reshape(-1):
                    arr = _flatten_observed_event_ids(item)
                    if arr.size > 0:
                        parts.append(arr)

                if len(parts) == 0:
                    return np.array([], dtype=np.int64)

                return np.concatenate(parts)

            return x.reshape(-1)

        if isinstance(x, (list, tuple)):
            parts = []

            for item in x:
                arr = _flatten_observed_event_ids(item)
                if arr.size > 0:
                    parts.append(arr)

            if len(parts) == 0:
                return np.array([], dtype=np.int64)

            return np.concatenate(parts)

        return np.array([x])

    k_all = _flatten_observed_event_ids(k_all).astype(np.int64)

    if len(k_all) < expected_total_transitions:
        raise ValueError(
            "Observed event vector is shorter than expected transitions: "
            f"len(k_all)={len(k_all)}, expected_total_transitions={expected_total_transitions}"
        )

    # Check CDF shape. In your runs, each CDF is shape:
    #   (total_transitions, time_grid)
    # so we index by global transition, not patient index.
    first_cdf = cdfs_all[0]

    if first_cdf.ndim != 2:
        raise ValueError(
            f"Expected each CDF to be 2D with shape (n_transitions, time_grid), "
            f"got shape {first_cdf.shape}."
        )

    if first_cdf.shape[0] < expected_total_transitions:
        raise ValueError(
            "CDF output has fewer transition rows than expected: "
            f"cdf rows={first_cdf.shape[0]}, expected={expected_total_transitions}"
        )

    risk_scores_by_patient = []
    observed_events_by_patient = []
    patient_ids_list = []

    global_t = 0

    for i, pid in enumerate(built.patient_ids):
        n_transitions_i = transition_counts[i]

        if n_transitions_i <= 0:
            continue

        risk_scores_matrix = np.zeros((n_transitions_i, n_events), dtype=np.float32)
        observed_i = np.zeros(n_transitions_i, dtype=np.int64)

        for t in range(n_transitions_i):
            for event_idx, cdf in enumerate(cdfs_all):
                if cdf.ndim != 2:
                    raise ValueError(
                        f"Expected CDF for event {event_idx} to be 2D, got shape {cdf.shape}."
                    )

                # Correct indexing:
                # cdf rows are flattened transitions across the whole batch.
                risk_scores_matrix[t, event_idx] = float(np.sum(cdf[global_t, :]))

            observed_i[t] = int(k_all[global_t])
            global_t += 1

        risk_scores_by_patient.append(risk_scores_matrix)
        observed_events_by_patient.append(observed_i)
        patient_ids_list.append(pid)

    if global_t != expected_total_transitions:
        raise ValueError(
            f"Transition count mismatch after extraction: extracted {global_t}, "
            f"expected {expected_total_transitions}."
        )

    # Build explicit vocabulary table for R.
    # Risk columns correspond to event IDs 1..n_events.
    inv_vocab = {int(v): str(k) for k, v in event_vocab.items()}

    iec_vocab_records = []
    for event_id in range(1, n_events + 1):
        event_name = inv_vocab.get(event_id, None)

        if event_name is not None:
            iec_vocab_records.append({
                "event": event_name,
                "event_id": int(event_id)
            })

    event_vocab_table = pd.DataFrame(iec_vocab_records)

    if len(event_vocab_table) != n_events:
        raise ValueError(
            f"Vocabulary table has {len(event_vocab_table)} rows but risk scores have "
            f"{n_events} columns. Check event_vocab/inv_vocab alignment."
        )

    return {
        "risk_scores": risk_scores_by_patient,
        "observed_events": observed_events_by_patient,
        "patient_ids": patient_ids_list,
        "event_vocab_table": event_vocab_table,
        "n_events": int(n_events),
        "n_transitions": int(global_t),
    }


def save_model_bundle(model_bundle: Dict[str, Any], path: str) -> None:
    """Save a model bundle (pretrain or finetune) to disk.
    
    Serialises the model weights, vocabulary, configuration, and all metadata
    to a PyTorch .pt file for later loading with load_model_bundle().
    
    Args:
        model_bundle: Dict returned by train_pretrain_model() or train_finetune_model()
        path: File path (string) where the bundle will be saved
    """
    # Ensure path is a string (reticulate sometimes passes R paths)
    path = str(path)
    
    # Extract model state dict
    model = model_bundle["model"]
    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
    
    # Prepare payload to save
    payload = {
        "type": model_bundle["type"],  # "pretrain" or "finetune"
        "state_dict": state_dict,
        "cfg": OmegaConf.to_container(model_bundle["cfg"]),
        "event_vocab": model_bundle["event_vocab"],
        "inv_vocab": model_bundle["inv_vocab"],
        "block_size": model_bundle.get("block_size", 128),
        "time_scale": model_bundle.get("time_scale", 1.0),
        "value_standardization": model_bundle.get("value_standardization", {}),
        "token_policy": model_bundle.get("token_policy", _token_policy_from_config()),
        "static_raw_cols": _as_list_or_none(model_bundle.get("static_raw_cols", None)),
        "static_col_names": _as_list_or_none(model_bundle.get("static_col_names", None)),
        "training_duration_secs": model_bundle.get("training_duration_secs", None),
        "device": str(model_bundle.get("device", "cpu")),
    }
    
    # For fine-tuned bundles, also save outcomes and risk_model
    if model_bundle["type"] == "finetune":
        outcomes = model_bundle.get("outcomes", None)
        if outcomes is not None:
            # Normalize: ensure it's a list, not a string
            if isinstance(outcomes, str):
                outcomes = [outcomes]
            payload["outcomes"] = list(outcomes)
        payload["risk_model"] = model_bundle.get("risk_model", "competing-risk")
        payload["outcome_horizon"] = model_bundle.get("outcome_horizon", model_bundle.get("time_scale", 1.0))
    
    # Save to file
    torch.save(payload, path)


def load_model_bundle(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    cfg = OmegaConf.create(payload["cfg"])

    vocab_size = len(payload["event_vocab"])
    kind = payload["type"]

    if kind == "pretrain":
        model = CausalExperiment(cfg=cfg, vocab_size=vocab_size)
    elif kind == "finetune":
        outcomes = payload.get("outcomes") or []
        # Guard: if saved while outcomes was a bare string (reticulate round-trip
        # artefact), restore it as a proper list before looking up tokens.
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        outcome_tokens = [payload["event_vocab"][o] for o in outcomes if o in payload["event_vocab"]]
        risk_model = payload.get("risk_model", "competing-risk")
        model = FineTuneExperiment(cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model bundle type: {kind}")

    model.load_state_dict(payload["state_dict"], strict=False)

    return {
        "type": kind,
        "model": model,
        "cfg": cfg,
        "event_vocab": payload["event_vocab"],
        "inv_vocab": payload["inv_vocab"],
        "block_size": int(payload["block_size"]),
        "time_scale": float(payload.get("time_scale", 1.0)),
        "value_standardization": _normalise_value_standardization(payload.get("value_standardization", None)),
        "outcome_horizon": float(payload.get("outcome_horizon", payload.get("time_scale", 1.0))),
        "outcomes": outcomes if kind == "finetune" else payload.get("outcomes", None),
        "risk_model": payload.get("risk_model", "competing-risk"),
        "token_policy": payload.get("token_policy", _token_policy_from_config()),
        "static_raw_cols": _as_list_or_none(payload.get("static_raw_cols", None)),
        "static_col_names": _as_list_or_none(payload.get("static_col_names", None)),
        "training_duration_secs": payload.get("training_duration_secs", None),
        "device": payload.get("device", "cpu"),
    }
