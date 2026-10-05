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
