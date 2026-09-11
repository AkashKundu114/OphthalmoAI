from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Unified 6-Class Retinal Fundus Taxonomy
CLASS_NAMES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]

# Clinical Urgency mapping for Posterior Pole conditions
URGENCY_TIERS: Dict[str, str] = {
    "Age-related Macular Degeneration": "Emergency", # Acute wet AMD conversion requires same-day anti-VEGF
    "Diabetic Retinopathy": "Urgent",
    "Glaucoma": "Urgent",
    "Hypertensive Retinopathy / Pathological Myopia": "Urgent",
    "Cataract": "Elective",
    "Normal": "None"
}

URGENCY_RANKS: Dict[str, int] = {
    "Emergency": 4,
    "Urgent": 3,
    "Elective": 2,
    "Non-urgent": 1,
    "None": 0
}

def build_clinical_cost_matrix(num_classes: int = 6) -> torch.Tensor:
    """
    Constructs an asymmetric clinical cost matrix C in R^{K x K}.
    C[i, j] represents the penalty of predicting class j when true class is i.
    - True sight-threatening (AMD, DR, Glaucoma) mispredicted as Normal: 5.0x penalty
    - True urgent condition mispredicted as elective: 3.0x penalty
    - Safe over-triage (Normal suspected as DR): 0.2x penalty
    - Correct diagnosis: 0.0 penalty
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

            if true_rank >= 3 and pred_rank == 0:
                # Catastrophic under-triage: Sight-threatening emergency labelled Normal
                C[i, j] = 5.0
            elif true_rank >= 3 and pred_rank < true_rank:
                # Significant under-triage
                C[i, j] = 3.0
            elif true_rank > pred_rank:
                # Moderate under-triage
                C[i, j] = 2.0
            elif true_rank < pred_rank:
                # Safe over-triage: False alarm is benign in healthcare screening
                C[i, j] = 0.25
            else:
                # Intra-tier misclassification
                C[i, j] = 1.0

    return C

class DirichletMetaClassifier(nn.Module):
    """
    Dirichlet Evidential Meta-Classifier for Posterior Pole Retinal Fundus Screening.
    Takes concatenated backbone representation features (e.g. 6 logits * 3 = 18),
    and outputs Dirichlet concentration parameters alpha in R^K.
    """
    def __init__(self, in_features: int = 18, num_classes: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        # Evidence e >= 0 using softplus
        evidence = F.softplus(logits)
        # Concentration alpha_k = evidence_k + 1
        alpha = evidence + 1.0
        # Dirichlet strength S = sum(alpha_k)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        # Expected categorical probabilities
        probs = alpha / S
        # Epistemic vacuity (total subjective uncertainty) u = K / S
        vacuity = self.net[-1].out_features / S.squeeze(-1)
        return probs, vacuity, alpha

class AsymmetricClinicalLoss(nn.Module):
    """
    Combines Evidential Type-II Maximum Likelihood with an Asymmetric Clinical Urgency Penalty
    and a KL-divergence regularization term to shrink evidence on misleading classes.
    """
    def __init__(self, num_classes: int = 6, lambda_cost: float = 1.0, lambda_kl: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cost = lambda_cost
        self.lambda_kl = lambda_kl
        self.register_buffer("cost_matrix", build_clinical_cost_matrix(num_classes))

    def forward(self, alpha: torch.Tensor, targets: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / S

        # 1. Sum of Squared Errors loss on Dirichlet distribution
        err = torch.sum((y_onehot - probs) ** 2, dim=-1)
        var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0)), dim=-1)
        base_loss = torch.mean(err + var)

        # 2. Asymmetric Clinical Cost Penalty
        cost_weights = self.cost_matrix[targets] # (B, K)
        cost_penalty = torch.mean(torch.sum(cost_weights * probs, dim=-1))

        # 3. Annealed KL divergence to uniform prior on misleading classes
        kl_anneal = min(1.0, epoch / 10.0)
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        kl = torch.mean(torch.sum((alpha_tilde - 1.0) * 0.05, dim=-1))

        return base_loss + self.lambda_cost * cost_penalty + self.lambda_kl * kl_anneal * kl

class EvidentialOODDetector:
    def __init__(self, vacuity_threshold: float = 0.50):
        self.vacuity_threshold = vacuity_threshold

    def evaluate(self, vacuity: float, max_prob: float) -> dict:
        is_ood = vacuity > self.vacuity_threshold or max_prob < 0.45
        return {
            "is_ood": is_ood,
            "epistemic_vacuity": round(float(vacuity), 4),
            "rejection_reason": "High epistemic uncertainty / atypical retinal morphology" if is_ood else None
        }
