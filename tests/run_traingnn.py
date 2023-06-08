import logging
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.utils as utils

tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)

import os

import pytest
from chemprop.train import cross_validate

from dfpl import options
from dfpl.train import traindmpnn


@pytest.fixture
def mock_opts():
    opts = options.GnnOptions()
    opts.gpu = 0
    opts.configFile = "config.json"
    return opts


def test_traindmpnn(mock_opts, monkeypatch, capsys):
    # Define a mock implementation of cross_validate
    def mock_cross_validate(args, train_func):
        return 0.85, 0.02

    # Monkeypatch the cross_validate function with the mock implementation
    monkeypatch.setattr(cross_validate, "cross_validate", mock_cross_validate)

    # Call the function
    traindmpnn(mock_opts)

    # Capture the printed output
    captured = capsys.readouterr()

    # Assertions
    assert captured.out.strip() == "Training DMPNN..."
    assert captured.err.strip() == f"Results: 0.85000 +/- 0.02000"
