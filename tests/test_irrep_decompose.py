"""Tests for magirrep.irrep_decompose."""

import numpy as np

from magirrep.irrep_decompose import find_active_irrep


class TestFindActiveIrrep:
    def test_single_active(self):
        n_mu = np.array([0.0, 0.0, 1.0, 0.0])
        result = find_active_irrep(n_mu)
        assert len(result) == 1
        assert result[0][0] == 2
        assert result[0][1] == 1.0

    def test_multiple_active(self):
        n_mu = np.array([1.0, 0.0, 2.0])
        result = find_active_irrep(n_mu)
        assert len(result) == 2

    def test_none_active(self):
        n_mu = np.array([0.0, 0.0, 0.0])
        result = find_active_irrep(n_mu)
        assert len(result) == 0

    def test_below_threshold_ignored(self):
        n_mu = np.array([1e-6, 1.0])
        result = find_active_irrep(n_mu)
        assert len(result) == 1
        assert result[0][0] == 1
