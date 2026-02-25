"""Tests for magirrep.mag_rep."""

import numpy as np

from magirrep.mag_rep import map_atoms_to_parent_cell


class TestMapAtomsToParentCell:
    def test_identity_transform_passthrough(self):
        """With identity transforms, positions and moments should be unchanged."""
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        magmoms = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        I = np.eye(3)
        zero = np.zeros(3)

        out_pos, out_mom = map_atoms_to_parent_cell(positions, magmoms, I, zero, I, zero)

        np.testing.assert_array_almost_equal(out_pos, positions)
        np.testing.assert_array_almost_equal(out_mom, magmoms)

    def test_positions_wrapped_to_unit_cell(self):
        """Positions outside [0,1) should be wrapped."""
        positions = np.array([[1.5, -0.3, 0.0]])
        magmoms = np.array([[0.0, 0.0, 1.0]])
        I = np.eye(3)
        zero = np.zeros(3)

        out_pos, _ = map_atoms_to_parent_cell(positions, magmoms, I, zero, I, zero)

        assert np.all(out_pos >= 0.0)
        assert np.all(out_pos < 1.0)
