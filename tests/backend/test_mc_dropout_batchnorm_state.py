import pytest
import torch
import torch.nn as nn

from backend.uncertainty import mc_dropout_predict


class _BatchNormDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(4, 3)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.classifier(self.dropout(self.batch_norm(inputs)))


def test_mc_dropout_keeps_batch_norm_in_eval_mode_and_preserves_running_stats():
    model = _BatchNormDropoutModel().eval()
    before_mean = model.batch_norm.running_mean.clone()
    before_var = model.batch_norm.running_var.clone()

    mc_dropout_predict(model, torch.randn(8, 4), n_passes=5)

    assert model.forward_calls == 5
    assert model.batch_norm.training is False
    assert torch.equal(model.batch_norm.running_mean, before_mean)
    assert torch.equal(model.batch_norm.running_var, before_var)
    assert model.training is False


def test_mc_dropout_restores_original_training_states():
    model = _BatchNormDropoutModel().train()
    model.batch_norm.eval()
    expected = {module: module.training for module in model.modules()}

    mc_dropout_predict(model, torch.randn(8, 4), n_passes=3)

    assert model.forward_calls == 3
    assert all(module.training is state for module, state in expected.items())


def test_mc_dropout_restores_modes_when_forward_raises():
    class BrokenModel(_BatchNormDropoutModel):
        def forward(self, inputs):
            self.forward_calls += 1
            raise RuntimeError("forward failed")

    model = BrokenModel().train()
    model.batch_norm.eval()
    expected = {module: module.training for module in model.modules()}
    with pytest.raises(RuntimeError, match="forward failed"):
        mc_dropout_predict(model, torch.randn(8, 4), n_passes=2)
    assert all(module.training is state for module, state in expected.items())


def test_mc_dropout_rejects_invalid_pass_count():
    with pytest.raises(ValueError, match="positive integer"):
        mc_dropout_predict(_BatchNormDropoutModel(), torch.randn(8, 4), n_passes=0)
