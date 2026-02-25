"""Shared fixtures for magirrep tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def cumnas_mcif():
    return str(DATA_DIR / "0.222_CuMnAs.mcif")


@pytest.fixture
def nio_mcif():
    return str(DATA_DIR / "1.6_NiO.mcif")
