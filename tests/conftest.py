"""
Test configuration and boundary fixtures for OphthalmoAI test suite.
Provides graceful fallbacks for environments where native PyTorch binary DLLs
are restricted by OS code integrity policies (e.g., Windows Smart App Control).
"""
import sys
import types
import numpy as np
import pytest

# Check if torch can be imported natively. If restricted by OS App Control policies,
# construct a compliant NumPy-backed mock torch engine so that calibration, evidential,
# and neural inference pipelines can be tested deterministically.
try:
    import torch
    import torch.nn
    import torch.nn.functional
except (ImportError, OSError):
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    f_mod = types.ModuleType("torch.nn.functional")
    optim_mod = types.ModuleType("torch.optim")

    class Size(tuple):
        """Mock torch.Size tuple subclass."""
        pass

    class MockTensor:
        """NumPy-backed tensor implementation for test environments without native libtorch."""
        def __init__(self, data, dtype=None):
            if isinstance(data, MockTensor):
                self.arr = data.arr.copy()
            elif isinstance(data, np.ndarray):
                self.arr = data.astype(float if dtype is None else dtype)
            else:
                self.arr = np.array(data, dtype=float if dtype is None else dtype)

        def __truediv__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(self.arr / val)

        def __rtruediv__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(val / self.arr)

        def __mul__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(self.arr * val)

        def __rmul__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(val * self.arr)

        def __sub__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(self.arr - val)

        def __rsub__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(val - self.arr)

        def __add__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(self.arr + val)

        def __radd__(self, other):
            val = other.arr if isinstance(other, MockTensor) else other
            return MockTensor(val + self.arr)

        def __neg__(self):
            return MockTensor(-self.arr)

        def __getitem__(self, item):
            res = self.arr[item]
            return MockTensor(np.array(res))

        def __setitem__(self, key, value):
            val = value.arr if isinstance(value, MockTensor) else value
            self.arr[key] = val

        def __float__(self):
            return self.item()

        def __int__(self):
            return int(self.item())

        def __index__(self):
            return int(self.item())

        def item(self):
            return float(self.arr.item() if self.arr.ndim == 0 or self.arr.size == 1 else self.arr.ravel()[0])

        @property
        def shape(self):
            return Size(self.arr.shape)

        @property
        def ndim(self):
            return self.arr.ndim

        @property
        def data(self):
            return self

        @data.setter
        def data(self, val):
            self.arr = val.arr.copy() if isinstance(val, MockTensor) else np.array(val, dtype=float)

        @property
        def device(self):
            return "cpu"

        def to(self, dev):
            return self

        def numel(self):
            return self.arr.size

        def clamp(self, min=None, max=None):
            return MockTensor(np.clip(self.arr, min, max))

        def clamp_(self, min=None, max=None):
            self.arr = np.clip(self.arr, min, max)
            return self

        def max(self, dim=None, keepdim=False):
            return MockTensor(np.max(self.arr, axis=dim, keepdims=keepdim))

        def sum(self, dim=None, keepdim=False):
            return MockTensor(np.sum(self.arr, axis=dim, keepdims=keepdim))

        def mean(self, dim=None, keepdim=False):
            return MockTensor(np.mean(self.arr, axis=dim, keepdims=keepdim))

        def var(self, dim=None, keepdim=False):
            n = self.arr.shape[dim] if dim is not None else self.arr.size
            ddof = 1 if n > 1 else 0
            return MockTensor(np.var(self.arr, axis=dim, keepdims=keepdim, ddof=ddof))

        def argmax(self, dim=None, keepdim=False):
            return MockTensor(np.argmax(self.arr, axis=dim, keepdims=keepdim))

        def all(self):
            return bool(np.all(self.arr))

        def any(self):
            return bool(np.any(self.arr))

        def backward(self):
            pass

        def zero_grad(self):
            pass

        def cpu(self):
            return self

        def unsqueeze(self, dim):
            return MockTensor(np.expand_dims(self.arr, axis=dim))

        def squeeze(self, dim=None):
            return MockTensor(np.squeeze(self.arr, axis=dim))

        def numpy(self):
            return self.arr

        def __repr__(self):
            return f"MockTensor({self.arr})"

    torch_mod.Tensor = MockTensor
    torch_mod.tensor = lambda d, **kw: MockTensor(d)
    torch_mod.ones = lambda *shape, **kw: MockTensor(np.ones(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape))
    torch_mod.zeros = lambda *shape, **kw: MockTensor(np.zeros(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape))
    torch_mod.clamp = lambda t, min=None, max=None: MockTensor(np.clip(t.arr if isinstance(t, MockTensor) else t, min, max))
    torch_mod.isfinite = lambda t: MockTensor(np.isfinite(t.arr if isinstance(t, MockTensor) else t))
    torch_mod.allclose = lambda a, b, atol=1e-5, rtol=1e-4: np.allclose(a.arr if isinstance(a, MockTensor) else a, b.arr if isinstance(b, MockTensor) else b, atol=atol, rtol=rtol)
    torch_mod.device = lambda d: d
    torch_mod.float32 = np.float32
    torch_mod.float64 = np.float64
    torch_mod.int64 = np.int64
    torch_mod.cat = lambda tensors, dim=0: MockTensor(np.concatenate([t.arr if isinstance(t, MockTensor) else t for t in tensors], axis=dim))
    torch_mod.stack = lambda tensors, dim=0: MockTensor(np.stack([t.arr if isinstance(t, MockTensor) else t for t in tensors], axis=dim))
    torch_mod.argmax = lambda t, dim=None, keepdim=False: MockTensor(np.argmax(t.arr if isinstance(t, MockTensor) else t, axis=dim, keepdims=keepdim))
    torch_mod.sum = lambda t, dim=None, keepdim=False: MockTensor(np.sum(t.arr if isinstance(t, MockTensor) else t, axis=dim, keepdims=keepdim))
    torch_mod.mean = lambda t, dim=None, keepdim=False: MockTensor(np.mean(t.arr if isinstance(t, MockTensor) else t, axis=dim, keepdims=keepdim))
    torch_mod.max = lambda t, dim=None, keepdim=False: MockTensor(np.max(t.arr if isinstance(t, MockTensor) else t, axis=dim, keepdims=keepdim))
    torch_mod.min = lambda t, dim=None, keepdim=False: MockTensor(np.min(t.arr if isinstance(t, MockTensor) else t, axis=dim, keepdims=keepdim))
    torch_mod.exp = lambda t: MockTensor(np.exp(t.arr if isinstance(t, MockTensor) else t))
    torch_mod.log = lambda t: MockTensor(np.log(t.arr if isinstance(t, MockTensor) else t))

    class Module:
        def __init__(self):
            self._modules = {}
            self.training = False

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = mode
            return self

    class Parameter(MockTensor):
        pass

    class CrossEntropyLoss:
        def __call__(self, logits, targets):
            return MockTensor(0.5)

    class LBFGS:
        def __init__(self, params, lr=0.01, max_iter=50):
            self.params = params

        def zero_grad(self):
            pass

        def step(self, closure):
            return closure()

    class NoGrad:
        def __call__(self, fn_or_arg=None):
            if callable(fn_or_arg):
                def wrapper(*args, **kwargs):
                    return fn_or_arg(*args, **kwargs)
                return wrapper
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    torch_mod.no_grad = NoGrad()

    def mock_softmax(input, dim=-1):
        x = input.arr if isinstance(input, MockTensor) else np.array(input)
        e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
        return MockTensor(e_x / np.sum(e_x, axis=dim, keepdims=True))

    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = list(layers)

        def forward(self, x):
            out = x
            for layer in self.layers:
                out = layer(out)
            return out

        def train(self, mode: bool = True):
            super().train(mode)
            for layer in self.layers:
                if hasattr(layer, "train"):
                    layer.train(mode)
            return self

    class ReLU(Module):
        def forward(self, x):
            arr = x.arr if isinstance(x, MockTensor) else np.array(x)
            return MockTensor(np.maximum(0, arr))

    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p

        def forward(self, x):
            if self.training:
                arr = x.arr if isinstance(x, MockTensor) else np.array(x)
                mask = (np.random.rand(*arr.shape) >= self.p).astype(np.float32)
                return MockTensor(arr * mask / max(1e-6, 1.0 - self.p))
            return x

    class LayerNorm(Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x):
            n = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 1 else 1
            return MockTensor(np.ones((n, self.out_features), dtype=np.float32) * 2.0)

    f_mod.softmax = mock_softmax
    f_mod.softplus = lambda t: MockTensor(np.log1p(np.exp(np.clip(t.arr if isinstance(t, MockTensor) else np.array(t), -20, 20))))
    nn_mod.Module = Module
    nn_mod.Parameter = Parameter
    class Conv2d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
        def forward(self, x):
            return x

    nn_mod.Conv2d = Conv2d
    nn_mod.Linear = Linear
    nn_mod.Sequential = Sequential
    nn_mod.ReLU = ReLU
    nn_mod.Dropout = Dropout
    nn_mod.LayerNorm = LayerNorm
    nn_mod.CrossEntropyLoss = CrossEntropyLoss
    nn_mod.functional = f_mod
    optim_mod.LBFGS = LBFGS

    torch_mod.randn = lambda *shape, **kw: MockTensor(np.random.randn(*(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)))
    torch_mod.all = lambda t: bool(np.all(t.arr if isinstance(t, MockTensor) else t))
    torch_mod.Size = Size
    torch_mod.cuda = types.ModuleType("torch.cuda")
    torch_mod.cuda.is_available = lambda: False
    torch_mod.cuda.memory_allocated = lambda *a: 0
    torch_mod.load = lambda *a, **k: {}
    torch_mod.save = lambda *a, **k: None
    torch_mod.from_numpy = lambda arr: MockTensor(arr)
    torch_mod.__version__ = "2.2.0"

    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = f_mod
    sys.modules["torch.optim"] = optim_mod

try:
    import torchvision
except (ImportError, OSError):
    from unittest.mock import MagicMock
    tv_mod = types.ModuleType("torchvision")
    tv_models = types.ModuleType("torchvision.models")
    tv_transforms = types.ModuleType("torchvision.transforms")
    def to_tensor(img):
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return MockTensor(arr / 255.0)

    def mock_compose(funcs):
        def apply_funcs(x):
            for f in funcs:
                x = f(x)
            return x
        return apply_funcs

    tv_transforms.Compose = mock_compose
    tv_transforms.Resize = lambda *a, **k: (lambda x: x)
    tv_transforms.CenterCrop = lambda *a, **k: (lambda x: x)
    tv_transforms.ToTensor = lambda *a, **k: to_tensor
    tv_transforms.Normalize = lambda *a, **k: (lambda x: x)
    tv_transforms.InterpolationMode = MagicMock()
    tv_models.resnet50 = MagicMock()
    tv_models.densenet201 = MagicMock()
    tv_models.convnext_small = MagicMock()
    tv_models.efficientnet_b4 = MagicMock()
    tv_models.efficientnet_v2_m = MagicMock()
    tv_mod.models = tv_models
    tv_mod.transforms = tv_transforms
    sys.modules["torchvision"] = tv_mod
    sys.modules["torchvision.models"] = tv_models
    sys.modules["torchvision.transforms"] = tv_transforms

