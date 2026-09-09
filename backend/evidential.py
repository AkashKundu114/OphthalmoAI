"""
OphthalmoAI - Evidential Deep Learning Module
=============================================
Implements:
1. DirichletMetaClassifier: Single-pass deterministic evidential neural network.
2. AsymmetricClinicalLoss: Type-I Digamma loss over Dirichlet distributions,
   incorporating an asymmetric clinical urgency penalty matrix C_ij that severely
   penalizes hazardous misclassifications (e.g., Keratitis or Uveitis missed as routine conditions).
3. EvidentialOODDetector: Vacuity-based out-of-distribution / invalid image rejection.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12 Classes standard ordering
CLASS_NAMES = [
    "Blepharitis", "Cataract", "Chalazion", "Conjunctivitis",
    "Jaundice", "Keratitis", "Normal", "Ptosis",
    "Pterygium", "Stye", "Subconjunctival Hemorrhage", "Uveitis"
]

# Clinical Urgency mapping
URGENCY_TIERS: Dict[str, str] = {
    "Keratitis": "Emergency",
    "Jaundice": "Emergency",
    "Uveitis": "Urgent",
    "Cataract": "Elective",
    "Pterygium": "Elective",
    "Ptosis": "Non-urgent",
    "Blepharitis": "Non-urgent",
    "Chalazion": "Non-urgent",
    "Stye": "Non-urgent",
    "Conjunctivitis": "Non-urgent",
    "Subconjunctival Hemorrhage": "Non-urgent",
    "Normal": "None"
}

URGENCY_RANKS: Dict[str, int] = {
    "Emergency": 4,
    "Urgent": 3,
    "Elective": 2,
    "Non-urgent": 1,
    "None": 0
}


def build_clinical_cost_matrix(num_classes: int = 12) -> torch.Tensor:
    """
    Constructs an asymmetric clinical cost matrix C in R^{K x K}.
    C[i, j] represents the penalty of predicting class j when true class is i.
    - True emergency (Keratitis, Jaundice) predicted as routine/normal: 5.0x penalty
    - True urgent (Uveitis) predicted as elective/normal: 3.0x penalty
    - Routine predicted as emergency (over-triage): 0.2x penalty (safe over-referral)
    - Intra-tier benign error (Chalazion vs Stye): 0.1x penalty
    """
    C = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for i, true_name in enumerate(CLASS_NAMES):
        true_urgency = URGENCY_TIERS.get(true_name, "Non-urgent")
        true_rank = URGENCY_RANKS[true_urgency]

        for j, pred_name in enumerate(CLASS_NAMES):
            if i == j:
                C[i, j] = 0.0
                continue

            pred_urgency = URGENCY_TIERS.get(pred_name, "Non-urgent")
            pred_rank = URGENCY_RANKS[pred_urgency]

            if true_rank == 4 and pred_rank < 3:
                # Hazardous: Missing emergency as elective/routine
                C[i, j] = 5.0
            elif true_rank == 3 and pred_rank < 2:
                # Hazardous: Missing urgent uveitis
                C[i, j] = 3.0
            elif true_rank > pred_rank:
                # Under-triage
                C[i, j] = 1.5 * (true_rank - pred_rank)
            elif true_rank < pred_rank:
                # Safe over-triage (conservative clinical referral)
                C[i, j] = 0.2
            else:
                # Same urgency tier misclassification
                C[i, j] = 0.5

    return C


class DirichletMetaClassifier(nn.Module):
    """
    Single-pass Evidential Meta-Classifier.
    Parameterizes a Dirichlet distribution Dir(alpha) over the 12 disease classes
    from concatenated base model logits or feature vectors.
    """
    def __init__(self, in_features: int = 36, num_classes: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, in_features), e.g. concatenated logits from 3 models
        Returns:
            probs: Expected probability vector (B, num_classes)
            vacuity: Epistemic uncertainty scalar per sample (B,) in [0, 1]
            alpha: Dirichlet parameters (B, num_classes)
        """
        logits = self.net(x)
        # Non-negative evidence via softplus: e_k >= 0
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        total_strength = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / total_strength
        # Epistemic vacuity: u = K / S
        vacuity = (self.num_classes / total_strength.squeeze(-1)).clamp(0.0, 1.0)
        return probs, vacuity, alpha


class AsymmetricClinicalLoss(nn.Module):
    """
    Asymmetric Clinical-Cost Evidential Loss (AC-HDL).
    Combines:
    1. Expected Cross-Entropy under Dirichlet prior (Digamma loss)
    2. Asymmetric Clinical Urgency Cost penalty
    3. KL-divergence regularizer on misleading non-target evidence
    """
    def __init__(
        self,
        num_classes: int = 12,
        lambda_cost: float = 1.0,
        lambda_kl: float = 0.005,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cost = lambda_cost
        self.lambda_kl = lambda_kl
        self.register_buffer("cost_matrix", build_clinical_cost_matrix(num_classes))

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        """
        Args:
            alpha: Dirichlet parameters (B, K)
            target: Ground truth class indices (B,)
            epoch: Current training epoch (for KL annealing)
        """
        batch_size = alpha.size(0)
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float().to(alpha.device)

        S = torch.sum(alpha, dim=-1, keepdim=True) # (B, 1)
        expected_probs = alpha / S # (B, K)

        # 1. Type-I Digamma loss: E[ -log(p_y) ] = psi(S) - psi(alpha_y)
        loss_digamma = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)

        # 2. Asymmetric Clinical Cost penalty: sum_j C[y, j] * p_j
        cost_mat = self.cost_matrix.to(alpha.device)
        sample_costs = cost_mat[target] # (B, K)
        loss_clinical_cost = torch.sum(sample_costs * expected_probs, dim=-1)

        # 3. KL Divergence Regularization on false evidence
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        beta = torch.ones_like(alpha_tilde) # Uniform Dirichlet prior Dir(1)
        
        kl_div = self._dirichlet_kl_divergence(alpha_tilde, beta)
        annealing_coef = min(1.0, epoch / 10.0)

        total_loss = loss_digamma + (self.lambda_cost * loss_clinical_cost) + (self.lambda_kl * annealing_coef * kl_div)
        return total_loss.mean()

    def _dirichlet_kl_divergence(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)
        lnB_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(S_alpha)
        lnB_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)
        kl = torch.sum((alpha - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True) + lnB_beta - lnB_alpha
        return kl.squeeze(-1)


class EvidentialOODDetector:
    """
    Evaluates whether an image input is Out-Of-Distribution (non-eye, severely degraded,
    or unrecognizable) using single-pass epistemic vacuity and Dirichlet total strength.
    """
    def __init__(self, vacuity_threshold: float = 0.65, min_evidence_threshold: float = 0.5):
        self.vacuity_threshold = vacuity_threshold
        self.min_evidence_threshold = min_evidence_threshold

    def evaluate(self, vacuity: float, alpha: torch.Tensor) -> Tuple[bool, str]:
        total_evidence = float(torch.sum(alpha - 1.0).item())
        if vacuity > self.vacuity_threshold or total_evidence < self.min_evidence_threshold:
            return True, (
                f"Image rejected as Out-Of-Distribution or uninterpretable: "
                f"epistemic vacuity {vacuity:.3f} exceeds safety threshold ({self.vacuity_threshold}), "
                f"with total evidence mass {total_evidence:.2f}."
            )
        return False, "Image passes evidential confidence criteria."
