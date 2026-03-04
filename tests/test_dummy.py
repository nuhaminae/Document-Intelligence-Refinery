# tests/test_dummy.py

import warnings


def test_placeholder():
    warnings.warn("This is a dummy test.", UserWarning)
    assert True
