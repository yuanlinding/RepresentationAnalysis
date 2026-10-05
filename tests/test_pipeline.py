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


class TestDeduplicatePositions:
    """Issue 9: integer binning of positions failed at the 0/1 cell boundary
    (0.99999 vs 0.00001) and at bin edges (values within tol straddling a
    multiple of tol)."""

    def test_cell_boundary_wrap(self):
        import numpy as np
        from magirrep.pipeline import _deduplicate_positions
        pos = np.array([[0.99999, 0.0, 0.0], [0.00001, 0.0, 0.0]])
        out = _deduplicate_positions(pos)
        assert len(out) == 1

    def test_bin_edge_within_tol(self):
        import numpy as np
        from magirrep.pipeline import _deduplicate_positions
        # 2e-7 apart (well within tol=1e-4) but on opposite sides of a bin edge
        pos = np.array([[0.450049, 0.5, 0.5], [0.450051, 0.5, 0.5]])
        out = _deduplicate_positions(pos)
        assert len(out) == 1

    def test_distinct_positions_kept(self):
        import numpy as np
        from magirrep.pipeline import _deduplicate_positions
        pos = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        out = _deduplicate_positions(pos)
        assert len(out) == 2

    def test_smallest_offset_representative_wins(self):
        import numpy as np
        from magirrep.pipeline import _deduplicate_positions
        pos = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        mag = np.array([[1.0, 0, 0], [2.0, 0, 0]])
        offsets = np.array([[1, 0, 0], [0, 0, 0]])  # second is canonical
        out_pos, out_mag = _deduplicate_positions(pos, magmoms=mag, offsets=offsets)
        assert len(out_pos) == 1
        np.testing.assert_allclose(out_mag[0], [2.0, 0, 0])


class TestComplexBasisVectorDisplay:
    """Issue 11: complex basis vectors were displayed as real parts only,
    silently mangling genuinely complex modes (e.g. interior-k irreps)."""

    def test_constant_phase_removed(self):
        import numpy as np
        from magirrep.pipeline import _scale_to_integers
        # an overall phase is unphysical — display should be the real vector
        v = np.exp(1j * np.pi / 4) * np.array([1.0, -1.0, 0.0])
        out = _scale_to_integers(v)
        assert not np.iscomplexobj(out)
        np.testing.assert_allclose(out, [1, -1, 0])

    def test_genuinely_complex_preserved(self):
        import numpy as np
        from magirrep.pipeline import _scale_to_integers
        v = np.array([1.0, 1j, 0.0])
        out = _scale_to_integers(v)
        assert np.iscomplexobj(out)
        np.testing.assert_allclose(out, [1, 1j, 0])

    def test_fmt_complex_value(self):
        from magirrep.pipeline import _fmt_bv_val
        assert _fmt_bv_val(1.0 + 0.0j) == "1"
        assert _fmt_bv_val(0.0 + 1.0j) == "0+1i"
        assert _fmt_bv_val(-0.5 - 0.5j) == "-0.500-0.500i"

    def test_print_basis_vectors_shows_complex(self, capsys):
        import numpy as np
        from magirrep.pipeline import _print_basis_vectors
        basis = [[np.array([1.0, 1j, 0.0])]]
        _print_basis_vectors(
            active_irreps=[(0, 1)], all_basis=basis,
            atom_labels=['Ni'], parent_positions=np.array([[0.0, 0.0, 0.0]]),
            identified=[(0, 1, 1.0, None)], bilbao_labels={0: 'mX1'},
            kpoint=np.zeros(3), it_number=1, parities=[''],
            irreps=[[np.eye(1)]], mode='magnetic')
        out = capsys.readouterr().out
        assert "1i" in out


class TestLibraryErrorsAreExceptions:
    """Issue 14: library functions called sys.exit(1), forcing every caller
    (web service, notebooks) to catch SystemExit."""

    def test_no_moments_raises_value_error(self, tmp_path):
        from pathlib import Path
        data = Path(__file__).parent / "data" / "1.6_NiO.mcif"
        content = data.read_bytes().decode("ascii", errors="ignore")
        idx = content.index("_atom_site_moment.label")
        loop_start = content.rindex("loop_", 0, idx)
        p = tmp_path / "no_moments.mcif"
        p.write_text(content[:loop_start])
        with pytest.raises(ValueError, match="[Nn]o nonzero magnetic moments"):
            run_analysis(str(p))


class TestParentDerivedFromMoments:
    """mCIFs with no _parent_space_group block whose magnetic cell is a
    supercell of the atomic structure's cell.  The parent group comes from
    spglib and k from how the moments transform under the lost parent
    translations (m(r+t) = -m(r)  =>  k.t = 1/2).

    - KCuF3_mp-1080828_Atype: P_c4/mnc (BNS 128.408) encoding; the BNS
      family group is not the parent.  Previously crashed with spglib >= 2.7
      ("too close distance between atoms") or gave a wrong mGM1+.
    - KCuF3_bulk_{A,G}_AFM: P1 encoding of the I4/mcm structure.  The lost
      I-centering is not listed as an anti-translation, so k must come from
      the moments: A-type flips sign (k != 0), G-type does not (k = 0).
    """

    DATA = __import__("pathlib").Path(__file__).parent / "data"
    MP = str(DATA / "KCuF3_mp-1080828_Atype.mcif")
    BULK_A = str(DATA / "KCuF3_bulk_A_AFM.mcif")
    BULK_G = str(DATA / "KCuF3_bulk_G_AFM.mcif")

    @staticmethod
    def _derive(path):
        from magirrep import parse_mcif
        from magirrep.pipeline import _derive_parent_from_structure
        return _derive_parent_from_structure(
            parse_mcif.parse_mcif_fields(path),
            parse_mcif.get_magnetic_structure(path))

    def test_bns_encoded_type_iv(self):
        import numpy as np
        it_number, kpoint, child_M, _ = self._derive(self.MP)
        assert it_number == 127          # P4/mbm, c_parent = c_mag / 2
        np.testing.assert_allclose(kpoint, [0, 0, 0.5])
        assert np.isclose(abs(np.linalg.det(child_M)), 2)

    def test_p1_encoded_a_type(self):
        import numpy as np
        it_number, kpoint, _, _ = self._derive(self.BULK_A)
        assert it_number == 140          # I4/mcm
        # k.(1/2,1/2,1/2) = 1/2 and k integer: the M/Z-type point of the BCT
        # lattice, e.g. (0,0,1); all such k are equivalent mod T_parent*
        assert np.allclose(kpoint, np.round(kpoint))
        assert np.isclose((kpoint @ [0.5, 0.5, 0.5]) % 1, 0.5)

    def test_p1_encoded_g_type_is_gamma(self):
        import numpy as np
        it_number, kpoint, _, _ = self._derive(self.BULK_G)
        assert it_number == 140
        np.testing.assert_allclose(kpoint, [0, 0, 0])

    def test_inconsistent_signs_rejected(self, tmp_path):
        import pytest
        # flip one Cu so the lost centering maps m -> -m on one pair and
        # m -> +m on the other: not a single k with +/-1 phases
        txt = open(self.BULK_A).read().replace(
            "Cu4 -1.000000 0.000000 0.000000", "Cu4 1.000000 0.000000 0.000000")
        p = tmp_path / "bad.mcif"
        p.write_text(txt)
        with pytest.raises(ValueError, match="propagation vector"):
            run_analysis(str(p), displacive_pass=False)

    @pytest.mark.parametrize("name", ["MP", "BULK_A", "BULK_G"])
    def test_magnetic_run_reconstructs_moments(self, name, capsys):
        run_analysis(getattr(self, name), displacive_pass=False)
        out = capsys.readouterr().out
        assert "‖M - M_rec‖/‖M‖ = 0.0000" in out

    def test_mp_labels(self, capsys):
        run_analysis(self.MP, displacive_pass=False)
        out = capsys.readouterr().out
        assert "P4/mbm" in out and "mGM1+" not in out

    @pytest.mark.parametrize("name", ["MP", "BULK_A"])
    def test_combined_and_displacive_run(self, name):
        from magirrep.pipeline import run_displacive_analysis
        run_analysis(getattr(self, name))
        run_displacive_analysis(getattr(self, name))


class TestCenteredLatticeKLabels:
    """k-point labels must use equivalence modulo the PRIMITIVE reciprocal
    lattice.  (0,0,1) in an I lattice is the M point, not Γ; seekpath points
    must be converted to conventional coordinates correctly."""

    @pytest.mark.parametrize("k,it_number,label", [
        ([0, 0, 1], 140, "M"), ([1, 1, 1], 140, "M"), ([0.5, 0.5, 0], 140, "X"),
        ([0, 0, 0], 140, "GM"), ([1, 1, 0], 140, "GM"),
        ([1, 0, 0], 225, "X"), ([0.5, 0.5, 0.5], 225, "L"), ([2, 0, 0], 225, "GM"),
        ([0, 1, 0], 229, "H"), ([0, 0, 0.5], 127, "Z"), ([0.5, 0.5, 0], 136, "M"),
    ])
    def test_kpoint_label(self, k, it_number, label):
        from magirrep.irrep_label import kpoint_label
        assert kpoint_label(k, it_number) == label

    def test_a_and_g_type_get_different_labels(self, capsys):
        data = __import__("pathlib").Path(__file__).parent / "data"
        run_analysis(str(data / "KCuF3_bulk_A_AFM.mcif"), displacive_pass=False)
        out_a = capsys.readouterr().out
        run_analysis(str(data / "KCuF3_bulk_G_AFM.mcif"), displacive_pass=False)
        out_g = capsys.readouterr().out
        assert "k = (0, 0, 1)  →  M point" in out_a and "at k=Γ" not in out_a
        assert "Identified: mM" in out_a
        assert "Identified: mGM" in out_g
