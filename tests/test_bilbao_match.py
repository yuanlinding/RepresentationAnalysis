"""Tests for magirrep.bilbao_match.match_irreps op-alignment logic.

Issue 3: when REPRES HTML yields only Seitz translations (R=None), the old
matcher paired Bilbao ops to little-group ops greedily by translation alone.
Ops sharing a translation could be cross-paired, scrambling the assembled
Bilbao characters and assigning wrong labels with a confident score.

fetch_repres is monkeypatched throughout — no network access.
"""

import numpy as np
import pytest

from magirrep import bilbao_match


# ── Synthetic C2v little group: [E|0], [C2z|τ], [mx|0], [my|τ], τ=(½,½,0) ──
E   = np.eye(3, dtype=int)
C2Z = np.diag([-1, -1, 1])
MX  = np.diag([-1, 1, 1])
MY  = np.diag([1, -1, 1])
TAU = np.array([0.5, 0.5, 0.0])

ROTATIONS    = np.array([E, C2Z, MX, MY])
TRANSLATIONS = np.array([np.zeros(3), TAU, np.zeros(3), TAU])
MLG          = np.array([0, 1, 2, 3])
GAMMA        = np.zeros(3)

# C2v character table in little-group (spgrep) op order [E, C2z, mx, my]
CHAR_TABLE = {
    'A1': [1, 1, 1, 1],
    'A2': [1, 1, -1, -1],
    'B1': [1, -1, 1, -1],
    'B2': [1, -1, -1, 1],
}
SPGREP_IRREPS = [
    [np.array([[c]], dtype=complex) for c in chars]
    for chars in CHAR_TABLE.values()
]


def _bilbao(op_order, with_R):
    """Build a fake fetch_repres payload listing ops in *op_order*
    (indices into the little-group op list)."""
    ops = []
    for i in op_order:
        ops.append({
            'R': ROTATIONS[i] if with_R else None,
            't': TRANSLATIONS[i].copy(),
        })
    irreps = [
        {'label': name, 'chars': [chars[i] for i in op_order]}
        for name, chars in CHAR_TABLE.items()
    ]
    return {'ops': ops, 'irreps': irreps}


class TestTranslationOnlyMatching:
    def test_ambiguous_translations_return_none(self, monkeypatch):
        """E/mx share t=0 and C2z/my share t=τ: translation alone cannot
        identify the ops, so matching must give up (None → caller falls back
        to dimension-sorted labels with a warning) instead of cross-pairing
        ops and returning wrong labels (old code labelled A2 as mB2)."""
        # Bilbao order [mx, E, C2z, my] — greedy-by-t cross-pairs E↔mx
        payload = _bilbao([2, 0, 1, 3], with_R=False)
        monkeypatch.setattr(bilbao_match, 'fetch_repres', lambda sg, k: payload)

        result = bilbao_match.match_irreps(
            SPGREP_IRREPS, ROTATIONS, TRANSLATIONS, MLG, 25, GAMMA)
        assert result is None

    def test_unique_translations_still_match(self, monkeypatch):
        """When every op has a distinct translation, t-only matching is
        unambiguous and must keep working (scrambled Bilbao order)."""
        rot = np.array([E, np.diag([1, -1, -1])])
        trans = np.array([np.zeros(3), np.array([0.5, 0.0, 0.0])])
        mlg = np.array([0, 1])
        irreps = [
            [np.array([[1.0 + 0j]]), np.array([[1.0 + 0j]])],
            [np.array([[1.0 + 0j]]), np.array([[-1.0 + 0j]])],
        ]
        payload = {
            'ops': [{'R': None, 't': trans[1].copy()},
                    {'R': None, 't': trans[0].copy()}],
            'irreps': [{'label': 'X', 'chars': [1, 1]},
                       {'label': 'Y', 'chars': [-1, 1]}],
        }
        monkeypatch.setattr(bilbao_match, 'fetch_repres', lambda sg, k: payload)

        result = bilbao_match.match_irreps(irreps, rot, trans, mlg, 25, GAMMA)
        assert result == {0: 'mX', 1: 'mY'}


class TestRotationMatching:
    def test_scrambled_order_with_rotations(self, monkeypatch):
        """With R available the ambiguity disappears: scrambled Bilbao op
        order must still produce the correct one-to-one labels."""
        payload = _bilbao([2, 0, 1, 3], with_R=True)
        monkeypatch.setattr(bilbao_match, 'fetch_repres', lambda sg, k: payload)

        result = bilbao_match.match_irreps(
            SPGREP_IRREPS, ROTATIONS, TRANSLATIONS, MLG, 25, GAMMA)
        assert result == {0: 'mA1', 1: 'mA2', 2: 'mB1', 3: 'mB2'}

    def test_unreachable_server_returns_none(self, monkeypatch):
        monkeypatch.setattr(bilbao_match, 'fetch_repres', lambda sg, k: None)
        result = bilbao_match.match_irreps(
            SPGREP_IRREPS, ROTATIONS, TRANSLATIONS, MLG, 25, GAMMA)
        assert result is None
