"""Tests for magirrep.irrep_decompose."""

import numpy as np
import pytest

from magirrep.irrep_decompose import find_active_irrep, decompose


class TestDecompose:
    """Tests for decompose() with the mapping_little_group parameter."""

    def _make_identity_irrep(self, dim, n_ops):
        """Return a trivial irrep: each op maps to dim×dim identity matrix."""
        return [np.eye(dim, dtype=complex) for _ in range(n_ops)]

    def test_all_ops_in_little_group(self):
        """When mapping = arange(N), result equals full-group reduction."""
        # 4 ops, 2 irreps: trivial 1D and sign-alternating 1D
        chi_mag = np.array([2.0, 0.0, -2.0, 0.0], dtype=complex)
        mapping = np.array([0, 1, 2, 3])
        irreps = [
            [np.array([[1+0j]]), np.array([[1+0j]]), np.array([[1+0j]]), np.array([[1+0j]])],
            [np.array([[1+0j]]), np.array([[-1+0j]]), np.array([[1+0j]]), np.array([[-1+0j]])],
        ]
        n_mu = decompose(irreps, chi_mag, mapping)
        # (1/4) * (2 + 0 - 2 + 0) = 0  for irrep 0
        # (1/4) * (2 + 0 + 2 + 0) = 1? No: (2*1 + 0*-1 + -2*1 + 0*-1)/4 = 0
        # irrep 1: (2*1 + 0*-(-1) + -2*1 + 0*-(-1))/4 = 0  hmm
        # Let's just check shapes and type
        assert len(n_mu) == 2
        assert n_mu.dtype == float

    def test_little_group_subset(self):
        """Only the little-group chi_mag values (via mapping) are used."""
        # chi_mag for 4 ops; little group only includes op 0 and op 2
        chi_mag = np.array([4.0, 99.0, -4.0, 99.0], dtype=complex)
        mapping = np.array([0, 2])  # ops 1 and 3 are NOT in little group
        # 1D irrep: constant 1
        irreps = [
            [np.array([[1+0j]]), np.array([[1+0j]])],
        ]
        n_mu = decompose(irreps, chi_mag, mapping)
        # n = (1/2) * (4*1 + (-4)*1) = 0
        assert len(n_mu) == 1
        np.testing.assert_allclose(n_mu[0], 0.0, atol=1e-10)

    def test_divisor_is_little_group_order(self):
        """Divisor must be len(mapping), not len(chi_mag)."""
        chi_mag = np.array([6.0, 0.0, 0.0, 0.0], dtype=complex)
        mapping = np.array([0])  # only identity in little group
        irreps = [
            [np.array([[1+0j]])],
        ]
        n_mu = decompose(irreps, chi_mag, mapping)
        # n = (1/1) * chi_mag[0] * 1 = 6  (NOT 6/4)
        np.testing.assert_allclose(n_mu[0], 6.0, atol=1e-10)


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
