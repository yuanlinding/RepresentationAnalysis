"""Tests for magirrep.little_group."""

import numpy as np
import pytest

from magirrep.little_group import get_hall_number, get_parent_sg_operations


class TestGetHallNumber:
    def test_p1(self):
        assert get_hall_number(1) == 1

    def test_fm3m(self):
        # SG #225 (Fm-3m) standard Hall number
        hall = get_hall_number(225)
        assert isinstance(hall, int)
        assert 1 <= hall <= 530

    def test_p4nmm(self):
        # SG #129 (P4/nmm)
        hall = get_hall_number(129)
        assert isinstance(hall, int)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_hall_number(0)


class TestGetParentSgOperations:
    def test_p1_has_1_op(self):
        rots, trans = get_parent_sg_operations(1)
        assert len(rots) == 1

    def test_fm3m_has_192_ops(self):
        rots, trans = get_parent_sg_operations(225)
        assert len(rots) == 192

    def test_rotation_shape(self):
        rots, trans = get_parent_sg_operations(129)
        assert rots.shape[1:] == (3, 3)
        assert trans.shape[1:] == (3,)
