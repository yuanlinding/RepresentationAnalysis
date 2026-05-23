"""Shared fixtures for magirrep tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def mnf2_mcif():
    return str(DATA_DIR / "0.15_MnF2.mcif")


@pytest.fixture
def nio_mcif():
    return str(DATA_DIR / "1.6_NiO.mcif")
