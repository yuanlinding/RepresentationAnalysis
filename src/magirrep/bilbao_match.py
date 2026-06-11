"""Fetch Bilbao REPRES and match spgrep irreps to Bilbao labels.

Matching strategy (character orthogonality, order-independent):
    (1/|G_k|) Σ_{g} χ_B(g)* · χ_S(g) = 1  iff  B ≡ S,  else ≈ 0

We match each Bilbao little-group operation to its spgrep counterpart by
(R, t) comparison, build chi_B in spgrep's ordering, then evaluate the
inner product.  The Bilbao label of the matching irrep is returned with
the 'm' prefix per MAGNDATA convention.
"""

import re
import numpy as np
from html.parser import HTMLParser
from typing import Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_CACHE: dict = {}   # {(sg_number, kpoint_tuple): result or None}

# ── Bravais letter ─────────────────────────────────────────────────────────────

def _bravais_letter(sg_number: int) -> str:
    """Return the conventional Bravais lattice letter for *sg_number*."""
    try:
        from magirrep.little_group import get_hall_number
        import spglib
        hall = get_hall_number(sg_number)
        sg_type = spglib.get_spacegroup_type(hall)
        # SpaceGroupType supports both attribute and dict access
        intl = (getattr(sg_type, 'international_short', None)
                or sg_type.get('international_short', '')
                or sg_type.get('international', ''))
        if intl:
            return intl[0].lower()
    except Exception:
        pass
    # Hardcoded fallback for the common cubic types we use
    F_SGs = {196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228, 229, 230}
    I_SGs = {82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122,
             139, 140, 141, 142, 197, 199, 204, 206, 211, 214, 217, 220}
    if sg_number in F_SGs:
        return 'f'
    if sg_number in I_SGs:
        return 'i'
    return 'p'


# ── HTML table collector ───────────────────────────────────────────────────────

class _TableCollector(HTMLParser):
    """Collect all <table> elements as lists of rows (each row = list of cell strings)."""

    def __init__(self):
        super().__init__()
        self.tables: list = []
        self._depth = 0
        self._cur_table: list = []
        self._cur_row: list = []
        self._cur_cell: list = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._depth += 1
            if self._depth == 1:
                self._cur_table = []
        elif tag == 'tr' and self._depth:
            self._cur_row = []
        elif tag in ('td', 'th') and self._depth:
            self._cur_cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._in_cell:
            self._cur_row.append(''.join(self._cur_cell).strip())
            self._in_cell = False
        elif tag == 'tr' and self._depth:
            if self._cur_row:
                self._cur_table.append(self._cur_row)
        elif tag == 'table':
            self._depth -= 1
            if self._depth == 0 and self._cur_table:
                self.tables.append(self._cur_table)

    def handle_data(self, data):
        if self._in_cell:
            self._cur_cell.append(data)

    def handle_entityref(self, name):
        if self._in_cell:
            if name in ('minus', 'ndash'):
                self._cur_cell.append('-')
            elif name == 'plus':
                self._cur_cell.append('+')


# ── Parsing helpers ────────────────────────────────────────────────────────────

_IRREP_RE = re.compile(r'^[A-Z]{1,3}\d+[+-]?$')
_SEITZ_T_RE = re.compile(r'\{[^|]+\|([^}]+)\}')   # capture translation from {R|t}


def _parse_frac(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        m = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s)
        if m:
            return int(m.group(1)) / int(m.group(2))
        return None


def _parse_char_val(s: str) -> Optional[complex]:
    s = (s.strip()
           .replace('\u2212', '-')   # unicode minus
           .replace('\u2013', '-')   # en-dash
           .replace('&minus;', '-'))
    if not s or s in ('-', '+'):
        return None
    try:
        return complex(float(s))
    except ValueError:
        try:
            return complex(s)
        except ValueError:
            return None


def _parse_repres_html(html: str) -> Optional[dict]:
    """
    Parse Bilbao REPRES HTML → {'ops': [...], 'irreps': [...]}.
    Returns None if the character table cannot be found.
    """
    collector = _TableCollector()
    collector.feed(html)
    tables = collector.tables

    if not tables:
        return None

    # ── Step 1: find the character table (has rows with irrep labels) ─────────
    char_rows = None
    char_table = None
    for tbl in tables:
        label_rows = [r for r in tbl if r and _IRREP_RE.match(r[0])]
        if len(label_rows) >= 2:
            char_table = tbl
            char_rows = label_rows
            break

    if not char_rows:
        return None

    # Determine whether column 1 is 'dim' or already a character
    has_dim_col = False
    try:
        v = float(char_rows[0][1])
        if 1 <= v <= 8 and abs(v - round(v)) < 0.01:
            has_dim_col = True
    except (ValueError, IndexError):
        pass

    char_start = 2 if has_dim_col else 1
    n_ops = len(char_rows[0]) - char_start

    if n_ops <= 0:
        return None

    # ── Step 2: parse irrep labels, dims, and character lists ─────────────────
    bilbao_irreps = []
    for row in char_rows:
        label = row[0]
        dim = 1
        if has_dim_col:
            try:
                dim = int(round(float(row[1])))
            except (ValueError, IndexError):
                pass
        chars = []
        for cell in row[char_start:char_start + n_ops]:
            v = _parse_char_val(cell)
            chars.append(v if v is not None else 0j)
        bilbao_irreps.append({'label': label, 'dim': dim, 'chars': chars})

    # ── Step 3: find operation (R, t) pairs ───────────────────────────────────
    ops = _find_ops(tables, html, n_ops)

    if ops is None or len(ops) != n_ops:
        return None

    return {'ops': ops, 'irreps': bilbao_irreps}


def _find_ops(tables, html: str, n_ops: int) -> Optional[list]:
    """Try several strategies to extract n_ops (R, t) operation descriptors."""

    # Strategy A: table that has exactly n_ops data rows with detectable R matrices
    for tbl in tables:
        data = [r for r in tbl if r and len(r) >= 2]
        if len(data) == n_ops:
            ops = [_row_to_op(r) for r in data]
            if all(op is not None for op in ops):
                return ops

    # Strategy B: regex for "( a b c )( d e f )( g h i )" rotation matrix triples
    pat = (r'\(\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)\s*\)'
           r'\s*\(\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)\s*\)'
           r'\s*\(\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)\s*\)')
    matrices = re.findall(pat, html)
    if len(matrices) == n_ops:
        ops = []
        for m in matrices:
            R = np.array([int(x) for x in m], dtype=int).reshape(3, 3)
            ops.append({'R': R, 't': np.zeros(3)})
        return ops

    # Strategy C: parse Seitz column headers from the char table header row
    # (gives translations only, used for t-matching as fallback)
    if tables:
        for tbl in tables:
            idx_of_first_label = None
            for i, row in enumerate(tbl):
                if row and _IRREP_RE.match(row[0]):
                    idx_of_first_label = i
                    break
            if idx_of_first_label is not None and idx_of_first_label > 0:
                header = tbl[idx_of_first_label - 1]
                seitz_ops = []
                for cell in header[2 if len(header) > n_ops + 1 else 1:]:
                    m = _SEITZ_T_RE.search(cell)
                    if m:
                        parts = [_parse_frac(p) for p in m.group(1).split(',')]
                        if None not in parts and len(parts) == 3:
                            seitz_ops.append({'R': None, 't': np.array(parts)})
                if len(seitz_ops) == n_ops:
                    return seitz_ops

    return None


def _row_to_op(row: list) -> Optional[dict]:
    """Try to extract an operation from a table row."""
    # Look for a cell with a Seitz symbol {R|t}
    for cell in row:
        m = _SEITZ_T_RE.search(cell)
        if m:
            t_parts = [_parse_frac(p) for p in m.group(1).split(',')]
            if None not in t_parts and len(t_parts) == 3:
                return {'R': None, 't': np.array(t_parts)}
    # Look for a cell with 9 integers (rotation matrix flattened)
    for cell in row:
        nums = re.findall(r'-?\d+', cell)
        if len(nums) == 9:
            R = np.array([int(x) for x in nums], dtype=int).reshape(3, 3)
            if abs(abs(np.linalg.det(R)) - 1) < 0.1:  # sanity: det ≈ ±1
                return {'R': R, 't': np.zeros(3)}
    return None


# ── HTTP fetch ─────────────────────────────────────────────────────────────────

_URLS = [
    "https://www.cryst.ehu.es/cgi-bin/cryst/programs/representations_out.pl",
    "https://www.cryst.ehu.es/cgi-bin/cryst/programs/repres",
]


def fetch_repres(sg_number: int, kpoint) -> Optional[dict]:
    """
    Query Bilbao REPRES for *sg_number* and *kpoint*.
    Returns {'ops': [...], 'irreps': [...]} or None on failure.
    Results are cached in-process by (sg_number, kpoint_tuple).
    """
    key = (int(sg_number), tuple(round(float(x), 5) for x in kpoint))
    if key in _CACHE:
        return _CACHE[key]
    if not _HAS_REQUESTS:
        _CACHE[key] = None
        return None

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    bb = _bravais_letter(sg_number)
    kx, ky, kz = float(kpoint[0]), float(kpoint[1]), float(kpoint[2])

    # Try several (URL, params) combinations
    attempts = [
        (_URLS[0], {'sg': sg_number, 'bb': bb, 'x': kx, 'y': ky, 'z': kz}),
        (_URLS[0], {'g':  sg_number, 'b':  bb, 'x': kx, 'y': ky, 'z': kz}),
        (_URLS[1], {'sg': sg_number, 'k': f'{kx},{ky},{kz}'}),
    ]
    for url, params in attempts:
        try:
            resp = requests.get(url, params=params, timeout=15, verify=False)
            if resp.status_code != 200:
                continue
            result = _parse_repres_html(resp.text)
            if result is not None:
                _CACHE[key] = result
                return result
        except Exception:
            continue

    _CACHE[key] = None
    return None


# ── Main entry point ───────────────────────────────────────────────────────────

def match_irreps(irreps, rotations, translations, mapping_little_group,
                 sg_number: int, kpoint) -> Optional[dict]:
    """
    Match spgrep *irreps* to Bilbao labels via character orthogonality.

    Parameters
    ----------
    irreps               : spgrep irrep list (irreps[α][i] is the i-th LG matrix)
    rotations            : all SG rotation matrices (N_ops × 3 × 3)
    translations         : all SG translations    (N_ops × 3)
    mapping_little_group : indices into rotations/translations for LG ops
    sg_number            : IT space group number
    kpoint               : k-vector in fractional reciprocal coordinates

    Returns
    -------
    dict {spgrep_idx: bilbao_label_with_m_prefix}  or  None if unavailable.
    """
    bilbao = fetch_repres(sg_number, kpoint)
    if bilbao is None:
        return None

    N = len(mapping_little_group)
    b_ops   = bilbao['ops']
    b_irreps = bilbao['irreps']

    if len(b_irreps) != len(irreps):
        return None  # unexpected count mismatch

    # ── Build op_map: bilbao_op_idx → little_group_position ───────────────────
    have_R = all(op.get('R') is not None for op in b_ops)

    def _t_match(t_b, sg_idx):
        diff = (t_b - translations[sg_idx]) % 1.0
        diff = np.minimum(diff, 1.0 - diff)
        return np.max(diff) < 1e-4

    op_map: dict = {}   # bi → si  (little-group position index)

    if have_R:
        # (R, t) identifies an op uniquely — first match is the only match
        used_si: set = set()
        for bi, bop in enumerate(b_ops):
            t_b = np.array(bop['t'], dtype=float)
            R_b = np.array(bop['R'], dtype=int)
            for si, sg_idx in enumerate(mapping_little_group):
                if si in used_si:
                    continue
                if _t_match(t_b, sg_idx) and np.array_equal(R_b, rotations[sg_idx].astype(int)):
                    op_map[bi] = si
                    used_si.add(si)
                    break
    else:
        # Translation-only fallback: a pairing is reliable only when the
        # translation identifies exactly one op on BOTH sides.  Ops sharing
        # a translation (e.g. all coset reps of a symmorphic group have t=0)
        # cannot be told apart and must stay unmatched — greedy first-match
        # here cross-pairs ops and scrambles the assembled characters.
        cand = []           # cand[bi] = [si, ...] with matching translation
        claims = {}         # si → number of bilbao ops matching it
        for bop in b_ops:
            t_b = np.array(bop['t'], dtype=float)
            sis = [si for si, sg_idx in enumerate(mapping_little_group)
                   if _t_match(t_b, sg_idx)]
            cand.append(sis)
            for si in sis:
                claims[si] = claims.get(si, 0) + 1
        for bi, sis in enumerate(cand):
            if len(sis) == 1 and claims[sis[0]] == 1:
                op_map[bi] = sis[0]

    if len(op_map) < int(N * 0.8):
        return None   # too few ops aligned unambiguously

    # ── Match each spgrep irrep to its Bilbao counterpart ─────────────────────
    result: dict = {}
    used_b: set  = set()

    mapped_si = np.array(sorted(op_map.values()), dtype=int)

    for alpha, irrep in enumerate(irreps):
        chi_S = np.array([np.trace(irrep[i]) for i in range(N)])

        best_score = -1.0
        best_b = -1
        for b_idx, birr in enumerate(b_irreps):
            if b_idx in used_b:
                continue
            # Assemble chi_B in spgrep little-group ordering
            chi_B = np.zeros(N, dtype=complex)
            for bi, si in op_map.items():
                if bi < len(birr['chars']):
                    chi_B[si] = birr['chars'][bi]
            # Cosine similarity over the unambiguously mapped ops only, so a
            # partial op_map does not deflate the score of a correct match.
            num = abs(np.dot(chi_B[mapped_si].conj(), chi_S[mapped_si]))
            den = (np.linalg.norm(chi_B[mapped_si]) * np.linalg.norm(chi_S[mapped_si]))
            score = num / den if den > 1e-12 else 0.0
            if score > best_score:
                best_score = score
                best_b = b_idx

        if best_b >= 0 and best_score > 0.8:
            result[alpha] = 'm' + b_irreps[best_b]['label']
            used_b.add(best_b)

    return result if len(result) == len(irreps) else None
