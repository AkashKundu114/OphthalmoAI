# Uncertainty Estimation and Human Review

OphthalmoAI is a research and educational retinal-screening prototype. The uncertainty and review helpers described here are engineering safeguards for directing uncertain outputs to human assessment. They are not validated clinical decision thresholds, diagnostic guarantees, or a substitute for a licensed eye-care professional.

## Monte Carlo dropout

`backend.uncertainty.mc_dropout_predict` performs repeated stochastic forward passes and returns the mean class-probability vector plus a scalar variance-based uncertainty summary. The current helper uses the first item in the output batch, so pass a single-image batch when per-image estimates are required.

The helper enables registered `torch.nn.Dropout` modules while placing all modules in evaluation mode. Batch Normalization therefore remains in evaluation mode and its running statistics are not updated. The helper restores each module's original training flag and serializes concurrent calls made through this helper for the same model instance.

Functional dropout calls (such as `torch.nn.functional.dropout`) and recurrent-layer internal dropout are not enabled by this implementation. Models relying on those mechanisms must use registered dropout modules or provide a separately validated inference implementation. The helper does not claim to support arbitrary custom training-mode side effects.

## Human-review triggers

`needs_human_review` returns a boolean and a list of reasons. Its default thresholds are defined in `backend/uncertainty.py`:

| Signal | Default trigger |
|---|---:|
| Predicted confidence | Below 0.75 |
| MC-dropout uncertainty | Above 0.15 |
| Evidential vacuity | Above 0.40 |
| Critical-diagnosis confidence | Below 0.90 |
| Conformal candidate count | More than 2 candidates |

The conformal candidate-count trigger is currently fixed at more than two; it is not a configurable argument. A conformal candidate set containing a critical diagnosis also triggers review when the selected diagnosis is not itself in the critical-diagnosis set.

These helper defaults are not evidence-based clinical cut-offs. They require prospective evaluation and calibration on the intended population and deployment domain before any operational use.

## Endpoint coverage

The review helper is not automatically applied to every inference path. In particular, asynchronous screening responses do not currently expose the same uncertainty and human-review fields as the synchronous helper-based path. Clients must not infer that an asynchronous response has passed these review checks.

## Interpreting the result

- A review flag means the software has identified a condition that warrants human assessment; it does not determine urgency or diagnose disease.
- A missing review flag does not establish that an output is correct, safe, or clinically appropriate.
- The uncertainty scalar is an implementation-specific summary. It should not be compared across models or model versions without a documented validation protocol.
- Conformal prediction behavior depends on the validity of its calibration procedure and assumptions. A set size alone is not a clinical risk guarantee.

See [Intended Use](INTENDED_USE.md), [Limitations](LIMITATIONS.md), [Safety Mechanisms](SAFETY_MECHANISMS.md), and [Evaluation](EVALUATION.md) for the broader research scope and validation requirements.
