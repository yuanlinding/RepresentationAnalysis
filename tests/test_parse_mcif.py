"""Tests for magirrep.parse_mcif."""

from pathlib import Path

import numpy as np
import pytest

from magirrep.parse_mcif import parse_kvector, parse_mcif_fields

DATA = Path(__file__).parent / "data"
NIO_MCIF = DATA / "1.6_NiO.mcif"


class TestParseKvector:
    def test_gamma(self):
        result = parse_kvector("0 0 0")
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_fractions(self):
        result = parse_kvector("1/2 1/2 1/2")
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_brackets_stripped(self):
        result = parse_kvector("[1/2 0 0]")
        np.testing.assert_array_almost_equal(result, [0.5, 0.0, 0.0])

    def test_commas(self):
        result = parse_kvector("1/3, 1/3, 0")
        np.testing.assert_array_almost_equal(result, [1 / 3, 1 / 3, 0.0])

    def test_negative_fraction(self):
        result = parse_kvector("-1/2 0 1/2")
        np.testing.assert_array_almost_equal(result, [-0.5, 0.0, 0.5])


def _write_with_k_rows(tmp_path, k_rows):
    """Copy the NiO mCIF replacing its single k-loop row with *k_rows*."""
    content = NIO_MCIF.read_bytes().decode("ascii", errors="ignore")
    assert "k1 [1/2 1/2 1/2]" in content
    out = tmp_path / "multi_k.mcif"
    out.write_text(content.replace("k1 [1/2 1/2 1/2]", "\n".join(k_rows)))
    return str(out)


class TestPropagationVectors:
    """Issue 5: only the first propagation vector was parsed; extra k-rows in
    the loop were silently dropped and the analysis ran as if single-k."""

    def test_single_k_unchanged(self):
        fields = parse_mcif_fields(str(NIO_MCIF))
        assert fields['kvector_str'] == "1/2 1/2 1/2"
        assert fields['kvector_strs'] == ["1/2 1/2 1/2"]

    def test_multi_k_all_vectors_parsed(self, tmp_path):
        path = _write_with_k_rows(tmp_path, ["k1 [0 0 1/2]", "k2 [1/2 0 0]"])
        fields = parse_mcif_fields(path)
        assert fields['kvector_strs'] == ["0 0 1/2", "1/2 0 0"]
        assert fields['kvector_str'] == "0 0 1/2"

    def test_duplicate_k_rows_treated_as_single_k(self, tmp_path):
        from magirrep.pipeline import run_analysis  # noqa: F401  (import check)
        path = _write_with_k_rows(tmp_path, ["k1 [1/2 1/2 1/2]",
                                             "k2 [1/2 1/2 1/2]"])
        fields = parse_mcif_fields(path)
        assert fields['kvector_str'] == "1/2 1/2 1/2"

    def test_run_analysis_rejects_multi_k(self, tmp_path):
        from magirrep.pipeline import run_analysis
        path = _write_with_k_rows(tmp_path, ["k1 [0 0 1/2]", "k2 [1/2 0 0]"])
        with pytest.raises(ValueError, match="[Mm]ulti-k"):
            run_analysis(path)

    def test_run_displacive_analysis_rejects_multi_k(self, tmp_path):
        from magirrep.pipeline import run_displacive_analysis
        path = _write_with_k_rows(tmp_path, ["k1 [0 0 1/2]", "k2 [1/2 0 0]"])
        with pytest.raises(ValueError, match="[Mm]ulti-k"):
            run_displacive_analysis(path)


class TestParseTransform:
    """Direct unit tests for parse_transform (pin behavior before refactors)."""

    def test_identity_with_shift(self):
        from magirrep.parse_mcif import parse_transform
        M, t = parse_transform("a,b,c;-1/4,1/4,0")
        np.testing.assert_allclose(M, np.eye(3))
        np.testing.assert_allclose(t, [-0.25, 0.25, 0.0])

    def test_doubled_cell(self):
        from magirrep.parse_mcif import parse_transform
        M, t = parse_transform("2a,2b,2c;0,0,0")
        np.testing.assert_allclose(M, 2 * np.eye(3))
        np.testing.assert_allclose(t, [0.0, 0.0, 0.0])

    def test_no_semicolon_means_zero_shift(self):
        from magirrep.parse_mcif import parse_transform
        M, t = parse_transform("a+b,-a+b,c")
        np.testing.assert_allclose(M, [[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(t, [0.0, 0.0, 0.0])

    def test_too_many_semicolons_raises(self):
        from magirrep.parse_mcif import parse_transform
        with pytest.raises(ValueError):
            parse_transform("a,b,c;0,0,0;1,2,3")
