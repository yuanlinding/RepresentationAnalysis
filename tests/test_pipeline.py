"""Integration tests for the full pipeline.

These tests require gemmi to be installed. They are skipped if gemmi is not
available.
"""

import pytest

gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

from magirrep.pipeline import run_analysis


class TestPipeline:
    def test_cumnas(self, cumnas_mcif):
        """CuMnAs should run without error (active irrep: mGM5-, SG#129, k=Gamma)."""
        run_analysis(cumnas_mcif)

    def test_nio(self, nio_mcif):
        """NiO should run without error (active irrep: mL3+, SG#225, k=L)."""
        run_analysis(nio_mcif)
