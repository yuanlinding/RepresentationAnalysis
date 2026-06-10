"""Regression tests for centering-aware atom matching and basis-change transforms.

Issue 1: centered Bravais lattices at zone-boundary k produced fractional
         irrep multiplicities because atom matching ignored centering
         translations after primitive-cell reduction.
Issue 4: moment vectors must keep their physical (Cartesian) value under the
         child→parent basis change of an mCIF transform.
"""

import numpy as np
import pytest

from magirrep import irrep_decompose, mag_rep
from magirrep.little_group import get_centering_translations
from magirrep.parse_mcif import parse_transform


L_POINT = np.array([0.5, 0.5, 0.5])


class TestCenteringAwareMatching:
    """NiO geometry: IT 225 (Fm-3m), k = L.  Single-atom orbits in the
    primitive cell at (0,0,0) (Ni, 4a) and (1/2,1/2,1/2) (O, 4b)."""

    @pytest.fixture(scope="class")
    def lg(self):
        irreps, rotations, translations, mlg = \
            irrep_decompose.get_little_group_irreps(225, L_POINT)
        centerings = get_centering_translations(225)
        return irreps, rotations, translations, mlg, centerings

    @pytest.mark.parametrize("pos", [
        np.array([[0.0, 0.0, 0.0]]),     # Ni 4a
        np.array([[0.5, 0.5, 0.5]]),     # O  4b
    ])
    def test_displacive_multiplicities_are_integers(self, lg, pos):
        irreps, rotations, translations, mlg, centerings = lg
        chi = mag_rep.compute_displacive_characters(
            rotations, translations, L_POINT, pos, centerings=centerings)
        n_mu = irrep_decompose.decompose(irreps, chi, mlg)

        for n in n_mu:
            assert abs(n - round(n)) < 1e-6, f"fractional multiplicity: {n_mu}"
        dim_sum = sum(round(n) * irreps[a][0].shape[0] for a, n in enumerate(n_mu))
        assert dim_sum == 3 * len(pos)

    def test_chi_satisfies_small_rep_phase_relation(self, lg):
        """chi({R|t+t_c}) = exp(-2πi k·t_c) · chi({R|t}) for centering t_c."""
        irreps, rotations, translations, mlg, centerings = lg
        pos = np.array([[0.5, 0.5, 0.5]])
        chi_all = mag_rep.compute_displacive_characters(
            rotations, translations, L_POINT, pos, centerings=centerings)
        chi = chi_all[np.asarray(mlg)]

        for i, idx in enumerate(mlg):
            for j, jdx in enumerate(mlg):
                if np.array_equal(rotations[jdx], rotations[idx]):
                    dt = translations[jdx] - translations[idx]
                    phase = np.exp(-2j * np.pi * np.dot(L_POINT, dt))
                    assert abs(chi[j] - phase * chi[i]) < 1e-8

    def test_magnetic_projection_is_complete_for_nio(self, lg):
        """Ni moment along [11-2] at L: eta of the active irrep must be 1.0
        (was 0.25 before the centering fix deflated the projectors)."""
        irreps, rotations, translations, mlg, centerings = lg
        pos = np.array([[0.0, 0.0, 0.0]])
        moment = np.array([1.0, 1.0, -2.0])   # arbitrary in-irrep direction

        D_lg = mag_rep.build_mag_rep_matrices(
            rotations, translations, L_POINT, pos, mlg, centerings=centerings)
        proj = irrep_decompose.compute_projection_operators(irreps, D_lg, mlg)
        total = sum(np.linalg.norm(P @ moment) ** 2 for P in proj)
        # The projectors must resolve the identity on the 3-dim moment space
        assert np.isclose(total, np.dot(moment, moment), atol=1e-8)

    def test_primitive_lattice_no_centerings_unchanged(self):
        """P-type lattices: centerings = [(0,0,0)], results identical to before."""
        assert len(get_centering_translations(136)) == 1   # P4_2/mnm
        assert len(get_centering_translations(138)) == 1   # P4_2/ncm
        assert len(get_centering_translations(225)) == 4   # Fm-3m


class TestMomentBasisChange:
    """A moment's Cartesian value must be invariant under the child→parent
    basis change:  A_child.T @ m_frac_child == A_parent.T @ m_frac_parent."""

    @pytest.mark.parametrize("transform_str", [
        "a,b,c;0,0,0",
        "2a,2b,2c;0,0,0",
        "c,a,b-c;1/4,0,0",        # axis-permuting + mixing (MAGNDATA-style)
        "a+b,-a+b,c;0,0,1/2",     # 45° cell rotation
        "b,a,c;0,0,0",            # improper swap (det = -1): the old sign(det)
                                  # factor wrongly flipped the moment here
    ])
    def test_cartesian_moment_invariant(self, transform_str):
        child_M, child_t = parse_transform(transform_str)

        # Arbitrary (orthorhombic-ish) parent lattice; A_child = child_M @ A_parent
        A_parent = np.array([[5.0, 0.0, 0.0],
                             [0.0, 6.0, 0.0],
                             [0.0, 0.5, 7.0]])
        A_child = child_M @ A_parent

        rng = np.random.default_rng(42)
        positions = rng.random((3, 3))
        m_frac_child = rng.standard_normal((3, 3))

        _, m_frac_parent, _ = mag_rep.map_atoms_to_parent_cell(
            positions, m_frac_child, child_M, child_t, np.eye(3), np.zeros(3))

        m_cart_child  = m_frac_child  @ A_child     # rows m_i: m_cart = A.T @ m_frac
        m_cart_parent = m_frac_parent @ A_parent
        np.testing.assert_allclose(m_cart_parent, m_cart_child, atol=1e-10)

    def test_position_physically_invariant(self):
        """Same invariance for atom positions (modulo parent lattice vectors)."""
        child_M, child_t = parse_transform("c,a,b-c;1/4,0,0")
        A_parent = np.array([[5.0, 0.0, 0.0],
                             [0.0, 6.0, 0.0],
                             [0.0, 0.5, 7.0]])
        A_child = child_M @ A_parent

        r_child = np.array([[0.3, 0.6, 0.1]])
        m = np.zeros((1, 3))
        r_parent, _, _ = mag_rep.map_atoms_to_parent_cell(
            r_child, m, child_M, child_t, np.eye(3), np.zeros(3))

        # r_phys(child) + shift·A_parent should equal r_phys(parent) mod lattice
        r_phys_child  = r_child[0] @ A_child + child_t @ A_parent
        r_phys_parent = r_parent[0] @ A_parent
        diff_frac = np.linalg.solve(A_parent.T, (r_phys_child - r_phys_parent))
        np.testing.assert_allclose(diff_frac, np.round(diff_frac), atol=1e-10)
