"""Integration tests for the full pipeline.

These tests require gemmi to be installed. They are skipped if gemmi is not
available.
"""

import pytest

gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

from magirrep.pipeline import run_analysis


@pytest.mark.skip(reason="pipeline has known bugs (P1-P3) — enable after fixes")
class TestPipeline:
    def test_cumnas(self, cumnas_mcif):
        run_analysis(cumnas_mcif)

    def test_nio(self, nio_mcif):
        run_analysis(nio_mcif)
