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
    time_scale: float            # global age divisor (e.g. 1825.0 for DAYS_SINCE_BIRTH, 1.0 for years)
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

    unique_events = sorted(events["event"].dropna().unique().tolist())
    reserved = set(vocab.keys())
    for event in unique_events:
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
    Each event age is divided by *time_scale* to give a dimensionless value:

        age_norm = raw_age / time_scale

    Use ``time_scale=1825.0`` when ``age`` is in days (``DAYS_SINCE_BIRTH``,
    matching FastEHR default).  Use ``time_scale=1.0`` when ``age`` is already
    in years.

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
    def __init__(self, built: BuiltData, targets_df: pd.DataFrame):
        targets = _clean_targets(targets_df)

        by_patient = targets.groupby("patient_id", sort=False).last().reset_index()

        row_map = {pid: i for i, pid in enumerate(built.patient_ids)}
        selected_rows = []
        target_tokens = []
        target_age_deltas = []
        target_values = []

        time_scale = float(built.time_scale)

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

            # Recover raw age: ages_norm = raw_age / time_scale, so
            #   raw_age = context_age_norm * time_scale
            context_age_raw = context_age_norm * time_scale

            # Delta in the same raw units, then normalise by time_scale
            delta = max(float(row["target_age"]) - context_age_raw, 0.0)
            delta_norm = delta / time_scale

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
                    label: str = "Training") -> List[float]:
    import time
    import math

    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    n_epochs  = int(epochs)
    n_batches = len(loader)
    history: List[float] = []
    BAR_WIDTH  = 20

    print(f"[survivehrR] {label}: {n_epochs} epoch(s), "
          f"{n_batches} batch(es)/epoch, device={device}", flush=True)

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

    return history


def train_pretrain_model(events_df: pd.DataFrame,
                         static_df: Optional[pd.DataFrame] = None,
                         config: Optional[Dict[str, Any]] = None,
                         event_vocab: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    config = dict(config or {})
    device = _device_from_config(config)
    token_policy = _token_policy_from_config(config)
    time_scale = float(config.get("time_scale", 1.0))

    built = _build_context_data(events_df=events_df,
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
    history = _run_train_loop(
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
        "token_policy": token_policy,
        "static_raw_cols": built.static_raw_cols,
        "static_col_names": built.static_col_names,
        "history": history,
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

    if pretrained_bundle is not None:
        event_vocab = dict(pretrained_bundle["event_vocab"])
        token_policy = dict(pretrained_bundle.get("token_policy", token_policy))
        # Always inherit time_scale from the pretrained bundle so context
        # normalisation is identical between pretrain and fine-tune.
        time_scale = float(pretrained_bundle.get("time_scale", time_scale))

    built = _build_context_data(events_df=events_df,
                                static_df=static_df,
                                block_size=int(config.get("block_size", 128)),
                                event_vocab=event_vocab,
                                token_policy=token_policy,
                                time_scale=time_scale)

    cfg = _build_cfg(config=config, vocab_size=len(built.event_vocab), fine_tune=True)
    _set_measurement_tokens(cfg, built.values, built.tokens)

    outcomes = [str(o) for o in outcomes]
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

    ds = FineTuneDataset(built, targets_df=targets_df)
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



    history = _run_train_loop(
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
        "outcomes": outcomes,
        "risk_model": risk_model,
        "token_policy": token_policy,
        "static_raw_cols": built.static_raw_cols,
        "static_col_names": built.static_col_names,
        "history": history,
    }


def predict_next_events(model_bundle: Dict[str, Any],
                        events_df: pd.DataFrame,
                        static_df: Optional[pd.DataFrame] = None,
                        max_new_tokens: int = 1) -> pd.DataFrame:
    model = model_bundle["model"]
    event_vocab = dict(model_bundle["event_vocab"])
    inv_vocab = {int(k): v for k, v in dict(model_bundle["inv_vocab"]).items()}

    block_size = int(model_bundle["block_size"])
    built = _build_context_data(events_df=events_df,
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
            out_tokens, out_ages, out_values, out_mask = base_model.generate(
                tokens=tokens,
                ages=ages,
                values=values,
                static_covariates=static_cov,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                exceed_block_size=False,
            )

            rows = []
            for i, pid in enumerate(built.patient_ids):
                attended = np.where(out_mask[i].detach().cpu().numpy() > 0)[0]
                if len(attended) == 0:
                    continue
                last_idx = int(attended[-1])
                tok = int(out_tokens[i, last_idx].detach().cpu().item())
                age_val = float(out_ages[i, last_idx].detach().cpu().item())
                value_val = float(out_values[i, last_idx].detach().cpu().item())
                rows.append(
                    {
                        "patient_id": pid,
                        "generated_token": tok,
                        "generated_event": inv_vocab.get(tok, "<UNK>"),
                        "generated_age": age_val,
                        "generated_value": value_val,
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

            rows = []
            outcome_labels = model_bundle.get("outcomes") or [f"risk_{i+1}" for i in range(len(cdfs))]
            for i, pid in enumerate(built.patient_ids):
                row = {"patient_id": pid}
                for j, cdf in enumerate(cdfs):
                    name = outcome_labels[j] if j < len(outcome_labels) else f"risk_{j+1}"
                    row[f"{name}_cdf_last"] = float(cdf[i, -1])
                    row[f"{name}_auc"] = float(np.trapz(cdf[i, :], dx=1.0 / max(cdf.shape[1] - 1, 1)))
                rows.append(row)

    return pd.DataFrame(rows)


def save_model_bundle(model_bundle: Dict[str, Any], path: str) -> None:
    payload = {
        "type": model_bundle["type"],
        "state_dict": model_bundle["model"].state_dict(),
        "cfg": OmegaConf.to_container(model_bundle["cfg"], resolve=True),
        "event_vocab": model_bundle["event_vocab"],
        "inv_vocab": model_bundle["inv_vocab"],
        "block_size": model_bundle["block_size"],
        "time_scale": model_bundle.get("time_scale", 1.0),
        "outcomes": model_bundle.get("outcomes", None),
        "risk_model": model_bundle.get("risk_model", "competing-risk"),
        "token_policy": model_bundle.get("token_policy", _token_policy_from_config()),
        "static_raw_cols": model_bundle.get("static_raw_cols", None),
        "static_col_names": model_bundle.get("static_col_names", None),
    }
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
        "outcomes": payload.get("outcomes", None),
        "risk_model": payload.get("risk_model", "competing-risk"),
        "token_policy": payload.get("token_policy", _token_policy_from_config()),
        "static_raw_cols": payload.get("static_raw_cols", None),
        "static_col_names": payload.get("static_col_names", None),
    }
