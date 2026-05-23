"""Integration tests for the full pipeline.

These tests require gemmi to be installed. They are skipped if gemmi is not
available.
"""

import pytest

gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

from magirrep.pipeline import run_analysis


class TestPipeline:
    def test_mnf2(self, mnf2_mcif):
        """MnF2 should run without error (active irrep: mGM3+, SG#136, k=Gamma, AFM)."""
        run_analysis(mnf2_mcif)

    def test_nio(self, nio_mcif):
        """NiO should run without error (active irrep: mL3+, SG#225, k=L)."""
        run_analysis(nio_mcif)
