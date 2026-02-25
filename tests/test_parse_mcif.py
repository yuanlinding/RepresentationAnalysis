"""Tests for magirrep.parse_mcif."""

import numpy as np
import pytest

from magirrep.parse_mcif import parse_kvector


class TestParseKvector:
    def test_gamma(self):
        result = parse_kvector("0 0 0")
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_fractions(self):
        result = parse_kvector("1/2 1/2 1/2")
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_brackets_stripped(self):
        result = parse_kvector("[1/2 0 0]")
        np.testing.assert_array_almost_equal(result, [0.5, 0.0, 0.0])

    def test_commas(self):
        result = parse_kvector("1/3, 1/3, 0")
        np.testing.assert_array_almost_equal(result, [1 / 3, 1 / 3, 0.0])

    def test_negative_fraction(self):
        result = parse_kvector("-1/2 0 1/2")
        np.testing.assert_array_almost_equal(result, [-0.5, 0.0, 0.5])
