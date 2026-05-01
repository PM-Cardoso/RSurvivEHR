"""
Experiment wrappers for the survivehrR R-package backend.

CausalExperiment   – wraps SurvStreamGPTForCausalModelling with a dict-batch
                     interface expected by survivehr_backend._run_train_loop.
FineTuneExperiment – adds a fine-tune head on top of the pre-trained backbone
                     for supervised outcome prediction (competing-risk or
                     single-risk).

Neither class depends on pytorch-lightning or wandb at runtime.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure wandb is importable even when not installed.
# desurv.py does a bare top-level `import wandb` which fails when the package
# is absent.  We insert a no-op stub before any SurvivEHR.src imports run.
# ---------------------------------------------------------------------------
if "wandb" not in sys.modules:
    _wandb_stub = types.ModuleType("wandb")
    _wandb_stub.log   = lambda *a, **kw: None
    _wandb_stub.init  = lambda *a, **kw: None
    _wandb_stub.watch = lambda *a, **kw: None
    sys.modules["wandb"] = _wandb_stub

# ---------------------------------------------------------------------------
# Model imports (all live inside inst/python/SurvivEHR/src/).
# ---------------------------------------------------------------------------
from SurvivEHR.src.models.survival.task_heads.causal import (
    SurvStreamGPTForCausalModelling,
)
from SurvivEHR.src.modules.head_layers.survival.competing_risk import (
    ODESurvCompetingRiskLayer,
)
from SurvivEHR.src.modules.head_layers.survival.single_risk_for_causal import (
    CausalODESurvSingleRiskLayer,
)


# ---------------------------------------------------------------------------
# CausalExperiment
# ---------------------------------------------------------------------------

class CausalExperiment(nn.Module):
    """
    Thin dict-batch wrapper around SurvStreamGPTForCausalModelling.

    The backend's _run_train_loop passes a batch *dict* to model(batch).
    SurvStreamGPTForCausalModelling.forward() expects individual keyword
    arguments. This class bridges the two interfaces.

    Attributes
    ----------
    model : SurvStreamGPTForCausalModelling
        The underlying survival transformer.  Exposed so that the fine-tune
        step can do ``ft.model.load_state_dict(pt.model.state_dict())``.
    """

    def __init__(self, cfg: Any, vocab_size: int) -> None:
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size
        self.model      = SurvStreamGPTForCausalModelling(cfg=cfg, vocab_size=vocab_size)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        is_generation:    bool = False,
        return_loss:      bool = True,
        return_generation: bool = False,
    ):
        """
        Parameters
        ----------
        batch : dict with keys
            ``tokens``           (bsz, seq_len) int64
            ``ages``             (bsz, seq_len) float32
            ``values``           (bsz, seq_len) float32
            ``static_covariates`` (bsz, n_cov) float32
            ``attention_mask``   (bsz, seq_len) bool / float32

        Returns
        -------
        outputs : dict  – ``{"surv": surv_dict, "values_dist": ...}``
        losses  : dict  – ``{"loss": tensor, ...}``
        hidden  : tensor – final transformer hidden states
        """
        return self.model(
            tokens          = batch["tokens"],
            ages            = batch["ages"],
            values          = batch["values"],
            covariates      = batch["static_covariates"],
            attention_mask  = batch["attention_mask"],
            is_generation   = is_generation,
            return_generation = return_generation,
            return_loss     = return_loss,
        )


# ---------------------------------------------------------------------------
# FineTuneExperiment
# ---------------------------------------------------------------------------

class FineTuneExperiment(nn.Module):
    """
    Fine-tuning wrapper.

    Adds a *new* competing-risk or single-risk survival head on top of the
    pre-trained backbone. Only the last attended hidden state is used for the
    supervised prediction, so this works with variable-length context windows.

    Attributes
    ----------
    model      : SurvStreamGPTForCausalModelling
        Full backbone (pre-trained).  Exposed for ``load_state_dict``.
    surv_layer : ODESurvCompetingRiskLayer | CausalODESurvSingleRiskLayer
        Fine-tune survival head; one risk per supplied outcome.
    outcome_tokens : list[int]
        Vocabulary indices of the fine-tune outcome events.
    """

    def __init__(
        self,
        cfg: Any,
        outcome_tokens: List[int],
        risk_model: str = "competing-risk",
        vocab_size: int = 2,
    ) -> None:
        super().__init__()

        self.cfg            = cfg
        self.outcome_tokens = list(outcome_tokens)
        self.risk_model     = risk_model
        self.vocab_size     = vocab_size

        n_embd     = cfg.transformer.n_embd
        n_outcomes = len(outcome_tokens)
        device     = "cuda" if torch.cuda.is_available() else "cpu"

        # Full pre-trained backbone (backbone weights loaded externally via
        # load_state_dict after construction).
        self.model = SurvStreamGPTForCausalModelling(cfg=cfg, vocab_size=vocab_size)

        # Fine-tune-specific survival head (freshly initialised).
        rm = risk_model.lower()
        if rm in ("competing-risk", "cr"):
            self.surv_layer = ODESurvCompetingRiskLayer(
                in_dim     = n_embd,
                hidden_dim = 32,
                num_risks  = n_outcomes,
                device     = device,
            )
        else:
            self.surv_layer = CausalODESurvSingleRiskLayer(
                in_dim     = n_embd,
                hidden_dim = 32,
                num_risks  = n_outcomes,
                device     = device,
            )

        # Token→head-index mapping (vocab_idx → 1-based head risk index).
        self._tok_to_head: Dict[int, int] = {
            int(tok): idx + 1 for idx, tok in enumerate(self.outcome_tokens)
        }

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _last_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run the transformer backbone and return the last-attended hidden state
        for every patient: shape (bsz, n_embd).
        """
        hidden = self.model.transformer(
            tokens         = batch["tokens"],
            ages           = batch["ages"],
            values         = batch["values"],
            covariates     = batch["static_covariates"],
            attention_mask = batch["attention_mask"],
        )  # (bsz, seq_len, n_embd)

        mask    = batch["attention_mask"]               # (bsz, seq_len)
        lengths = mask.sum(dim=1).long() - 1            # (bsz,)  0-based last index
        lengths = lengths.clamp(min=0)

        bsz, _, emb = hidden.shape
        idx = lengths.view(bsz, 1, 1).expand(bsz, 1, emb)
        context = hidden.gather(1, idx).squeeze(1)      # (bsz, n_embd)
        return context

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        is_generation:     bool = False,
        return_loss:       bool = True,
        return_generation: bool = False,
    ):
        """
        Parameters
        ----------
        batch : dict
            Same structure as CausalExperiment plus, during training:
            ``target_token``      (bsz,) int64  – vocab index of outcome event
            ``target_age_delta``  (bsz,) float32 – normalised time to event
            ``target_value``      (bsz,) float32 – optional measurement value

        Returns
        -------
        outputs, losses, context  – same shape contract as CausalExperiment.
        """
        context    = self._last_context(batch)          # (bsz, n_embd)
        context_3d = context.unsqueeze(1)               # (bsz, 1, n_embd) – seq_len=1 for the head

        # ---- generation / inference path ------------------------------
        if is_generation or not return_loss:
            bsz          = context.shape[0]
            # Dummy targets so the head can return CDFs without a loss.
            dummy_tokens = torch.ones(bsz, 1, dtype=torch.long,    device=context.device)
            dummy_ages   = torch.zeros(bsz, 1, dtype=torch.float32, device=context.device)

            surv_dict, _ = self.surv_layer.predict(
                hidden_states  = context_3d,
                target_tokens  = dummy_tokens,
                target_ages    = dummy_ages,
                attention_mask = None,          # signals is_generation branch inside predict()
                is_generation  = True,
                return_cdf     = True,
                return_loss    = False,
            )
            outputs = {"surv": surv_dict, "values_dist": None}
            losses  = {"loss": None, "loss_desurv": None, "loss_values": None}
            return outputs, losses, context

        # ---- training path -------------------------------------------
        # target_token: vocab indices  →  remap to 1-based head risk indices
        raw_tok  = batch["target_token"].view(-1, 1)    # (bsz, 1) vocab indices
        head_tok = torch.ones_like(raw_tok)             # default to risk-1 for unmapped tokens
        for vocab_idx, head_idx in self._tok_to_head.items():
            head_tok = torch.where(raw_tok == vocab_idx,
                                   torch.full_like(head_tok, head_idx),
                                   head_tok)

        target_ages = batch["target_age_delta"].view(-1, 1)  # (bsz, 1)

        surv_dict, losses_desurv = self.surv_layer.predict(
            hidden_states  = context_3d,
            target_tokens  = head_tok,
            target_ages    = target_ages,
            attention_mask = None,          # triggers the generation-branch inside predict()
            is_generation  = True,          # re-use that branch for single-position fine-tuning
            return_cdf     = return_generation,
            return_loss    = True,
        )

        loss_desurv = torch.sum(torch.stack(losses_desurv))
        outputs = {"surv": surv_dict, "values_dist": None}
        losses  = {
            "loss":         loss_desurv,
            "loss_desurv":  loss_desurv,
            "loss_values":  torch.tensor(0.0, device=context.device),
        }
        return outputs, losses, context
