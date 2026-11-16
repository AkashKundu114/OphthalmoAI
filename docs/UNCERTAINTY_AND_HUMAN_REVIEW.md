# Uncertainty Estimation and Human Review

OphthalmoAI is a research and educational retinal-screening prototype. The uncertainty and review helpers described here are engineering safeguards for directing uncertain outputs to human assessment. They are not validated clinical decision thresholds, diagnostic guarantees, or a substitute for a licensed eye-care professional.

## Monte Carlo dropout

`backend.uncertainty.mc_dropout_predict` performs repeated stochastic forward passes and returns the mean class-probability vector plus a scalar variance-based uncertainty summary.

During these passes, dropout modules are enabled while the rest of the model remains in evaluation mode. In particular, Batch Normalization layers must remain in evaluation mode so their running means and variances are not updated by repeated inference. The helper restores each module's original training/evaluation flag after inference, including when the forward pass raises an exception.

The current helper reads the first item in the model output batch (`logits[0]`). Call it with a single-image batch unless the helper is extended to return per-item estimates.

## Human-review triggers

`needs_human_review` returns a boolean and a list of reasons. Its default thresholds are defined in `backend/uncertainty.py`:

| Signal | Default trigger |
|---|---:|
| Predicted confidence | Below 0.75 |
| MC-dropout uncertainty | Above 0.15 |
| Evidential vacuity | Above 0.40 |
| Critical-diagnosis confidence | Below 0.90 |
| Conformal candidate count | More than 2 candidates |

A conformal candidate set containing a critical diagnosis also triggers review when the selected diagnosis is not itself in the critical-diagnosis set.

## Interpreting the result

- A review flag means the software has identified a condition that warrants human assessment; it does not determine urgency or diagnose disease.
- A missing review flag does not establish that an output is correct, safe, or clinically appropriate.
- Thresholds are configurable arguments to `needs_human_review` and must be evaluated on the intended population and deployment domain before any operational use.
- The uncertainty scalar is an implementation-specific summary. It should not be compared across models or model versions without a documented validation protocol.
- Conformal prediction behavior depends on the validity of its calibration procedure and assumptions. A set size alone is not a clinical risk guarantee.

See [Intended Use](INTENDED_USE.md), [Limitations](LIMITATIONS.md), [Safety Mechanisms](SAFETY_MECHANISMS.md), and [Evaluation](EVALUATION.md) for the broader research scope and validation requirements.
