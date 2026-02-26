"""Tests for magirrep.little_group."""

import numpy as np
import pytest

import spglib

from magirrep.little_group import get_hall_number, get_parent_sg_operations


class TestGetHallNumber:
    def test_p1(self):
        assert get_hall_number(1) == 1

    def test_fm3m(self):
        # SG #225 (Fm-3m) has a unique setting — any valid Hall number is fine
        hall = get_hall_number(225)
        assert isinstance(hall, int)
        assert 1 <= hall <= 530

    def test_p4nmm_prefers_origin_choice_2(self):
        # SG #129 (P4/nmm) has two origin choices; should return OC2 (Hall 409)
        hall = get_hall_number(129)
        sg_type = spglib.get_spacegroup_type(hall)
        choice = sg_type['choice'] if hasattr(sg_type, '__getitem__') else sg_type.choice
        assert choice == '2', f"Expected origin choice 2, got {choice} (Hall {hall})"

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
