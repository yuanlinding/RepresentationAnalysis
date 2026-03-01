"""Tests for magirrep.mag_rep."""

import numpy as np

from magirrep.mag_rep import map_atoms_to_parent_cell, build_mag_rep_matrices, compute_characters


class TestMapAtomsToParentCell:
    def test_identity_transform_passthrough(self):
        """With identity transforms, positions and moments should be unchanged."""
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        magmoms = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        I = np.eye(3)
        zero = np.zeros(3)

        out_pos, out_mom, _ = map_atoms_to_parent_cell(positions, magmoms, I, zero, I, zero)

        np.testing.assert_array_almost_equal(out_pos, positions)
        np.testing.assert_array_almost_equal(out_mom, magmoms)

    def test_positions_wrapped_to_unit_cell(self):
        """Positions outside [0,1) should be wrapped."""
        positions = np.array([[1.5, -0.3, 0.0]])
        magmoms = np.array([[0.0, 0.0, 1.0]])
        I = np.eye(3)
        zero = np.zeros(3)

        out_pos, _, _ = map_atoms_to_parent_cell(positions, magmoms, I, zero, I, zero)

        assert np.all(out_pos >= 0.0)
        assert np.all(out_pos < 1.0)

    def test_returns_wrap_offsets(self):
        """map_atoms_to_parent_cell returns a third array of integer wrap offsets.

        Atom at (0,0,0) with P=2I: r_raw=(0,0,0), wrapped=(0,0,0), L=[0,0,0].
        Atom at (0.5,0,0) with P=2I: r_raw=(1.0,0,0), wrapped=(0,0,0), L=[1,0,0].
        """
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        magmoms = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        P = 2 * np.eye(3)
        zero = np.zeros(3)

        _, _, offsets = map_atoms_to_parent_cell(positions, magmoms, P, zero, np.eye(3), zero)

        np.testing.assert_array_equal(offsets[0], [0, 0, 0])
        np.testing.assert_array_equal(offsets[1], [1, 0, 0])

    def test_moment_scaling_no_det_factor(self):
        """With P=2I (child_transform), moment [1,0,0] becomes [2,0,0] (not [16,0,0]).

        sign(det(2I)) = sign(8) = +1, so the axial-vector factor is +1, and the
        transform is just P @ m = 2*m.
        """
        positions = np.array([[0.0, 0.0, 0.0]])
        magmoms = np.array([[1.0, 0.0, 0.0]])
        P = 2 * np.eye(3)
        zero = np.zeros(3)

        _, out_mom, _ = map_atoms_to_parent_cell(positions, magmoms, P, zero, np.eye(3), zero)

        np.testing.assert_array_almost_equal(out_mom[0], [2.0, 0.0, 0.0])


class TestBuildMagRepMatrices:
    """Tests for build_mag_rep_matrices()."""

    def test_identity_single_atom(self):
        """Identity op on one atom at origin: D = I_3, Tr = 3."""
        rotations = np.array([np.eye(3, dtype=int)])
        translations = np.zeros((1, 3))
        kpoint = np.array([0.0, 0.0, 0.0])
        positions = np.array([[0.0, 0.0, 0.0]])
        mapping = np.array([0])

        D_list = build_mag_rep_matrices(rotations, translations, kpoint, positions, mapping)
        assert len(D_list) == 1
        np.testing.assert_array_almost_equal(D_list[0], np.eye(3, dtype=complex))

    def test_inversion_single_atom_at_origin(self):
        """Inversion {-E|0} at origin: D = det(-I)*(-I) = (-1)*(-I) = I, Tr = 3.

        Matches chi_axial = det(-I)*Tr(-I) = (-1)*(-3) = +3.
        """
        rotations = np.array([-np.eye(3, dtype=int)])
        translations = np.zeros((1, 3))
        kpoint = np.array([0.0, 0.0, 0.0])
        positions = np.array([[0.0, 0.0, 0.0]])
        mapping = np.array([0])

        D_list = build_mag_rep_matrices(rotations, translations, kpoint, positions, mapping)
        np.testing.assert_array_almost_equal(D_list[0], np.eye(3, dtype=complex))
        assert np.isclose(np.trace(D_list[0]), 3.0)

    def test_trace_consistency(self):
        """Tr(D[i]) == chi_mag[mapping[i]] for all little-group ops."""
        # Two atoms; two ops: identity and a 2-fold rotation about z
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        kpoint = np.array([0.0, 0.0, 0.0])
        rotations = np.array([
            np.eye(3, dtype=int),
            np.diag([-1, -1, 1]),   # 2-fold rotation about z
        ])
        translations = np.zeros((2, 3))
        mapping = np.array([0, 1])

        chi_mag = compute_characters(rotations, translations, kpoint, positions)
        D_list = build_mag_rep_matrices(rotations, translations, kpoint, positions, mapping)

        for i, idx in enumerate(mapping):
            tr_D = np.trace(D_list[i])
            assert np.isclose(tr_D, chi_mag[idx], atol=1e-10), (
                f"Op {i}: Tr(D)={tr_D:.6f}, chi_mag={chi_mag[idx]:.6f}"
            )

    def test_trace_consistency_with_phase(self):
        """Trace consistency holds for non-Gamma k with inter-atom mapping."""
        # Two atoms related by a body-centering translation
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        kpoint = np.array([0.5, 0.0, 0.0])
        # Translation by [0.5, 0.5, 0.5] maps atom 0 -> atom 1 (and 1 -> 0+L)
        rotations = np.array([np.eye(3, dtype=int)])
        translations = np.array([[0.5, 0.5, 0.5]])
        mapping = np.array([0])

        chi_mag = compute_characters(rotations, translations, kpoint, positions)
        D_list = build_mag_rep_matrices(rotations, translations, kpoint, positions, mapping)

        tr_D = np.trace(D_list[0])
        assert np.isclose(tr_D, chi_mag[0], atol=1e-10)
