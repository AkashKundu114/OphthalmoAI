"""
Root conftest for OphthalmoAI test suite.
Provides graceful fallbacks for environments where native PyTorch binary DLLs
are restricted by OS code integrity policies (e.g., Windows Smart App Control).
"""
import sys
import types
import numpy as np
import pytest

# Ensure tests/conftest logic runs globally
try:
    import tests.conftest  # noqa: F401
except ImportError:
    pass
