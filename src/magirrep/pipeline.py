"""Orchestration pipeline: parse -> transform -> irreps -> decompose -> label."""

import contextlib
import io
import os
import sys
from collections import Counter

import numpy as np
import spglib

from magirrep import parse_mcif, mag_rep, irrep_decompose, irrep_label, bilbao_match
from magirrep.little_group import build_reference_crystal, get_hall_number


@contextlib.contextmanager
def _tee_stdout(filepath):
    """Print to both terminal and *filepath* simultaneously."""
    buf = io.StringIO()
    old = sys.stdout   # save BEFORE class definition so _Tee can close over it

    class _Tee:
        def write(self, s):
            old.write(s)    # write to the real terminal (captured in closure)
            buf.write(s)

        def flush(self):
            old.flush()

    sys.stdout = _Tee()
    try:
        yield
    finally:
        sys.stdout = old
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(buf.getvalue())


def _dbg(verbose, msg):
    """Print a debug line when verbose mode is on."""
    if verbose:
        print(f"[DEBUG] {msg}")


def _dbg_atoms(verbose, label, positions, magmoms):
    """Print atom table in verbose mode."""
    if not verbose:
        return
    print(f"[DEBUG] {label}  ({len(positions)} atoms)")
    for i, (r, m) in enumerate(zip(positions, magmoms)):
        print(f"[DEBUG]   {i+1:2d}  r = ({r[0]:7.4f}, {r[1]:7.4f}, {r[2]:7.4f})"
              f"   m = [{m[0]:7.4f}, {m[1]:7.4f}, {m[2]:7.4f}]  |m| = {np.linalg.norm(m):.4f}")


def _deduplicate_positions(positions: np.ndarray, magmoms: np.ndarray = None,
                            offsets: np.ndarray = None, labels: list = None,
                            tol: float = 1e-4):
    """Keep only unique rows in *positions* (wrapped to [0,1)) within *tol*.

    If *magmoms* / *labels* are provided, the corresponding rows are kept in sync.
    If *offsets* is provided (integer wrap vectors from map_atoms_to_parent_cell),
    prefer the representative with the smallest squared-norm offset among atoms
    mapping to the same parent-cell position.

    Returns
    -------
    Tuple of whichever of (positions, magmoms, labels) were non-None, or just
    positions if all extras are None.
    """
    unique: dict = {}  # key -> (original_index, squared_offset_norm)
    for i, r in enumerate(positions):
        key = tuple(np.round(r / tol).astype(int))
        L_norm_sq = int(np.sum(offsets[i] ** 2)) if offsets is not None else 0
        if key not in unique or L_norm_sq < unique[key][1]:
            unique[key] = (i, L_norm_sq)

    kept_idx = sorted(v[0] for v in unique.values())

    unique_pos = positions[kept_idx] if kept_idx else np.zeros((0, 3))
    if magmoms is not None and labels is not None:
        unique_mag    = magmoms[kept_idx] if kept_idx else np.zeros((0, 3))
        unique_labels = [labels[i] for i in kept_idx]
        return unique_pos, unique_mag, unique_labels
    elif magmoms is not None:
        unique_mag = magmoms[kept_idx] if kept_idx else np.zeros((0, 3))
        return unique_pos, unique_mag
    elif labels is not None:
        unique_labels = [labels[i] for i in kept_idx]
        return unique_pos, unique_labels
    return unique_pos


def _select_primitive_atoms(conv_positions: np.ndarray, it_number: int,
                             magmoms: np.ndarray = None, labels: list = None,
                             tol: float = 1e-4):
    """
    From a list of positions in conventional-cell fractional coordinates,
    return only those belonging to ONE primitive cell.

    For primitive Bravais lattices (P-type) this is a no-op.  For centered
    lattices (F, I, A, B, C, R) it reduces e.g. 4 Ni atoms (Fm-3m
    conventional) to 1 Ni (FCC primitive cell).

    Centering translations (e.g. (½,½,0) for C-type) are lattice-parameter
    independent — they are the same rational fractions for any crystal with
    that Bravais type.  We use them directly instead of a transformation-
    matrix derived from a generic reference lattice (which would be wrong for
    non-cubic systems).

    Returns
    -------
    Tuple of whichever of (positions, magmoms, labels) were non-None, or just
    positions if all extras are None.
    """
    # Extract centering translations: pure-translation ops (R = Identity) from
    # the space-group symmetry database, in conventional fractional coordinates.
    hall_no = get_hall_number(it_number)
    sg_ops = spglib.get_symmetry_from_database(hall_no)
    eye3 = np.eye(3, dtype=int)
    centering: list = []
    for R, t in zip(sg_ops['rotations'], sg_ops['translations']):
        if np.allclose(R, eye3, atol=1e-5):
            ct = t % 1.0
            if not any(np.allclose(ct, c, atol=1e-5) for c in centering):
                centering.append(ct)

    # Keep one atom per centering orbit.  Atom j is a duplicate of already-kept
    # atom i if  r_j ≡ r_i + ct  (mod Z³)  for some centering vector ct.
    kept_idx: list = []
    kept_pos: list = []   # mod-1 positions of atoms already kept

    for i, r in enumerate(conv_positions):
        r_mod = r % 1.0
        is_dup = False
        for r_kept in kept_pos:
            for ct in centering:
                diff = (r_mod - r_kept - ct) % 1.0
                diff = np.where(diff > 0.5, diff - 1.0, diff)  # fold to (−½, ½]
                if np.all(np.abs(diff) < tol):
                    is_dup = True
                    break
            if is_dup:
                break
        if not is_dup:
            kept_idx.append(i)
            kept_pos.append(r_mod)

    result_pos = conv_positions[kept_idx] if kept_idx else np.zeros((0, 3))
    if magmoms is not None and labels is not None:
        result_mag    = magmoms[kept_idx] if kept_idx else np.zeros((0, 3))
        result_labels = [labels[i] for i in kept_idx]
        return result_pos, result_mag, result_labels
    elif magmoms is not None:
        result_mag = magmoms[kept_idx] if kept_idx else np.zeros((0, 3))
        return result_pos, result_mag
    elif labels is not None:
        result_labels = [labels[i] for i in kept_idx]
        return result_pos, result_labels
    return result_pos


# ── character-table helpers ───────────────────────────────────────────────────

_OP_TYPE = {
    ( 3,  1): 'E',   ( 2,  1): 'C6',  ( 1,  1): 'C4',
    ( 0,  1): 'C3',  (-1,  1): 'C2',
    (-3, -1): 'i',   (-2, -1): 'S3',  (-1, -1): 'S4',
    ( 0, -1): 'S6',  ( 1, -1): 'm',
}

_OP_TYPE_ITA = {
    ( 3,  1): '1',   ( 2,  1): '6',   ( 1,  1): '4',
    ( 0,  1): '3',   (-1,  1): '2',
    (-3, -1): '-1',  (-2, -1): '-6',  (-1, -1): '-4',
    ( 0, -1): '-3',  ( 1, -1): 'm',
}


def _rtype(R):
    """Short Schoenflies label for a 3×3 rotation/improper-rotation matrix."""
    tr  = int(round(np.trace(R)))
    det = int(round(np.linalg.det(R)))
    return _OP_TYPE.get((tr, det), f'?{tr}')


def _rtype_ita(R):
    """ITA/international point-group symbol for a 3×3 rotation matrix."""
    tr  = int(round(np.trace(R)))
    det = int(round(np.linalg.det(R)))
    return _OP_TYPE_ITA.get((tr, det), f'?{tr}')


_AXIS_LABELS = {
    (1, 0, 0): 'x',      (0, 1, 0): 'y',      (0, 0, 1): 'z',
    (1, 1, 0): '[110]',  (1,-1, 0): '[1-10]',
    (1, 0, 1): '[101]',  (1, 0,-1): '[10-1]',
    (0, 1, 1): '[011]',  (0, 1,-1): '[01-1]',
    (1, 1, 1): '[111]',  (1, 1,-1): '[11-1]',
    (1,-1, 1): '[1-11]', (1,-1,-1): '[1-1-1]',
    (2,-1, 0): '[2-10]', (1, 2, 0): '[120]',   # hexagonal
    (2, 1, 0): '[210]',  (1,-2, 0): '[1-20]',
}


def _op_axis(R):
    """Return the integer axis/plane-normal vector for a 3×3 rotation matrix.

    Proper rotations  (det=+1): null space of (R − I)  = rotation axis.
    Improper rotations (det=−1): null space of (R + I)  = mirror-normal or
        rotation axis of the proper part (same formula works for m, −3, −4, −6).
    Returns None for E (1) and i (−1), which have no unique axis.
    """
    tr  = int(round(np.trace(R)))
    det = int(round(np.linalg.det(R)))
    if (tr == 3 and det == 1) or (tr == -3 and det == -1):
        return None
    M = R.astype(float) - np.eye(3) if det == 1 else R.astype(float) + np.eye(3)
    _, _, Vt = np.linalg.svd(M)
    n = Vt[-1]
    n /= max(np.max(np.abs(n)), 1e-9)
    for v in n:
        if abs(v) > 1e-4:
            if v < 0:
                n = -n
            break
    return np.round(n).astype(int)


def _rotation_sense(R, n):
    """Return '+' or '-' for the rotation sense of R around axis n.

    Uses the right-hand rule: '+' if the rotation angle θ is positive
    (counter-clockwise when viewed from the tip of n).

    Works for both proper (det=+1) and improper (det=−1) rotations; for the
    latter the sense of the embedded proper rotation part is returned.
    The formula derives from the axial vector of the antisymmetric part of
    the proper-rotation matrix Rp: (Rp−Rp^T)/2 = sin(θ)[n̂]×,
    so dot(axial_vector, n̂) = sin(θ) > 0 for θ ∈ (0, π).
    """
    n_f = np.array(n, dtype=float)
    n_f = n_f / np.linalg.norm(n_f)
    det = int(round(np.linalg.det(R)))
    Rp  = np.array(R, dtype=float) if det == 1 else -np.array(R, dtype=float)
    A   = (Rp - Rp.T) / 2                          # antisymmetric part
    ax  = np.array([A[2, 1], A[0, 2], A[1, 0]])    # axial vector = sin(θ) n̂
    s   = float(np.dot(ax, n_f))
    return '+' if s > 1e-6 else '-'


def _op_symbol_ita(R):
    """Full ITA symbol with axis and rotation sense where needed.

    Examples: '1', '-1', '2z', 'mx', '4+z', '4-z', '-4+z', '3+[111]', '-3-[111]'.
    The '+'/'-' sense suffix is added for 3-, 4-, and 6-fold rotations (and their
    improper counterparts) to distinguish the two senses around the same axis.
    """
    base = _rtype_ita(R)
    n    = _op_axis(R)
    if n is None:
        return base
    key  = tuple(n)
    axis = _AXIS_LABELS.get(key, f'[{n[0]}{n[1]:+d}{n[2]:+d}]')
    if base in ('3', '-3', '4', '-4', '6', '-6'):
        sense = _rotation_sense(R, n)
        return base + sense + axis
    return base + axis


def _t_frac(v, max_denom=12):
    """Format a fractional translation component as a compact fraction string."""
    from fractions import Fraction
    f = Fraction(float(v)).limit_denominator(max_denom)
    if f.numerator == 0:
        return '0'
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _seitz(R, t):
    """Format a space-group operation in Seitz notation {sym | tx, ty, tz}."""
    sym   = _op_symbol_ita(R)
    t_str = ', '.join(_t_frac(ti) for ti in t)
    return f"{{{sym} | {t_str}}}"


def _seitz_to_xyz(R, t):
    """Convert (3×3 int rotation, 3-float translation) to xyz string like '-x+1/2,-y+1/2,z'."""
    from fractions import Fraction
    vars_ = ['x', 'y', 'z']
    parts = []
    for row_i, ti in zip(R, t):
        terms = []
        for c, v in zip(row_i, vars_):
            ic = int(round(c))
            if ic == 1:    terms.append(f'+{v}')
            elif ic == -1: terms.append(f'-{v}')
            elif ic != 0:  terms.append(f'+{ic}{v}')
        if abs(ti) > 1e-9:
            frac = Fraction(float(ti)).limit_denominator(12)
            s = f'+{frac}' if frac > 0 else str(frac)
            terms.append(s)
        s = ''.join(terms).lstrip('+') or '0'
        parts.append(s)
    return ','.join(parts)


def _dedup_lg_ops(mapping_little_group, rotations):
    """Return (i_lg, idx) pairs keeping one representative per unique rotation R.

    For centered Bravais lattices the little group contains multiple coset
    representatives {R|t} and {R|t+t_c} that share the same rotation part R
    (centering copies).  This function returns only the first occurrence of
    each unique R, giving the |G⁰_k| coset representatives of G_k / T.
    """
    seen_R = set()
    result = []
    for i_lg, idx in enumerate(mapping_little_group):
        R_key = tuple(rotations[idx].flatten().astype(int))
        if R_key not in seen_R:
            seen_R.add(R_key)
            result.append((i_lg, idx))
    return result


# Mapping from spglib international symbol → Schoenflies symbol
# for all 32 crystallographic point groups.
_INTL_TO_SCHOENFLIES = {
    '1':     'C1',   '-1':    'Ci',
    '2':     'C2',   'm':     'Cs',    '2/m':   'C2h',
    '222':   'D2',   'mm2':   'C2v',   '2mm':   'C2v',
    'm2m':   'C2v',  'mmm':   'D2h',
    '4':     'C4',   '-4':    'S4',    '4/m':   'C4h',
    '422':   'D4',   '4mm':   'C4v',   '-42m':  'D2d',
    '-4m2':  'D2d',  '4/mmm': 'D4h',
    '3':     'C3',   '-3':    'S6',
    '32':    'D3',   '312':   'D3',    '321':   'D3',
    '3m':    'C3v',  '3m1':   'C3v',   '31m':   'C3v',
    '-3m':   'D3d',  '-3m1':  'D3d',   '-31m':  'D3d',
    '6':     'C6',   '-6':    'C3h',   '6/m':   'C6h',
    '622':   'D6',   '6mm':   'C6v',   '-6m2':  'D3h',
    '-62m':  'D3h',  '6/mmm': 'D6h',
    '23':    'T',    'm-3':   'Th',    '432':   'O',
    '-43m':  'Td',   'm-3m':  'Oh',
}


def _little_group_point_group(mapping_little_group, rotations, translations, kpoint):
    """Schoenflies symbol of the point group of the *small group* of k.

    The small group = {g ∈ G_k : exp(−2πi k·t_g) = 1}.  For primitive
    Bravais lattices this equals G_k; for centred lattices at zone-boundary
    k-points the centering translations give phase −1, so the small group is
    a proper subgroup of G_k.  The small group's point group is the physically
    correct little co-group of k in the primitive-cell BZ.
    """
    kpt = np.asarray(kpoint, dtype=float)
    seen = []
    for idx in mapping_little_group:
        phase = np.exp(-2j * np.pi * np.dot(kpt, translations[idx]))
        if abs(phase - 1.0) < 1e-4:
            R = rotations[idx].astype(int)
            if not any(np.array_equal(R, s) for s in seen):
                seen.append(R)
    if not seen:
        return '?'
    try:
        pg_result = spglib.get_pointgroup(np.array(seen))
        intl = pg_result[0].strip()          # e.g. '-3m', '4/mmm'
        schf = _INTL_TO_SCHOENFLIES.get(intl, intl)
        return f"{schf} ({intl})"
    except Exception:
        return '?'


def _conjugacy_classes_lg(ops_R, ops_t, tol=1e-4):
    """Group little-group operations into conjugacy classes (union-find).

    g_k · g_i · g_k^{-1} = g_j  (checked mod lattice translations).
    Returns list of lists of indices into ops_R / ops_t.
    """
    n = len(ops_R)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for k in range(n):
        Rk     = ops_R[k].astype(float)
        tk     = ops_t[k]
        Rk_inv = np.linalg.inv(Rk)
        for i in range(n):
            Ri = ops_R[i].astype(float)
            ti = ops_t[i]
            Rj = Rk @ Ri @ Rk_inv
            tj = -Rj @ tk + Rk @ ti + tk
            for j in range(n):
                if np.allclose(ops_R[j], Rj, atol=tol):
                    diff = tj - ops_t[j]
                    if np.allclose(diff - np.round(diff), 0, atol=tol):
                        union(i, j)
                        break

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ── section printers ──────────────────────────────────────────────────────────

def _crystal_system(it_number: int) -> str:
    """Return crystal system name from IT number."""
    if it_number <= 2:   return 'Triclinic'
    if it_number <= 15:  return 'Monoclinic'
    if it_number <= 74:  return 'Orthorhombic'
    if it_number <= 142: return 'Tetragonal'
    if it_number <= 194: return 'Trigonal/Hexagonal'
    return 'Cubic'


def _atomic_number(label: str) -> int:
    """Atomic number from element or species string (e.g. 'Mn', 'Sr2+', 'O2-')."""
    from pymatgen.core.periodic_table import get_el_sp
    try:
        return int(get_el_sp(label).Z)
    except Exception:
        return 1


def _get_wyckoff_sites(it_number: int, parent_positions: np.ndarray,
                       labels: list = None) -> list:
    """Return Wyckoff letter for each atom in *parent_positions*."""
    ref_lattice, _, _ = build_reference_crystal(it_number)
    hall_no = get_hall_number(it_number)
    if labels is not None:
        numbers = [_atomic_number(lbl) for lbl in labels]
    else:
        numbers = list(np.ones(len(parent_positions), dtype=int))

    # Extract centering translations (pure-translation ops: R = Identity)
    sg_ops = spglib.get_symmetry_from_database(hall_no)
    eye3 = np.eye(3, dtype=int)
    centering = []
    for R, t in zip(sg_ops['rotations'], sg_ops['translations']):
        if np.allclose(R, eye3, atol=1e-5):
            ct = t % 1.0
            if not any(np.allclose(ct, c, atol=1e-5) for c in centering):
                centering.append(ct)

    # Expand primitive representatives to the full conventional cell
    exp_positions = []
    exp_numbers = []
    orig_exp_idx = []  # index in exp_positions for each original atom
    for i, (r, n) in enumerate(zip(parent_positions, numbers)):
        first = True
        for ct in centering:
            r_exp = (r + ct) % 1.0
            if not any(np.allclose(r_exp, ep, atol=1e-4) for ep in exp_positions):
                exp_positions.append(r_exp)
                exp_numbers.append(n)
                if first:
                    orig_exp_idx.append(len(exp_positions) - 1)
                    first = False
        if first:  # no copy was added (shouldn't happen, but be safe)
            orig_exp_idx.append(len(exp_positions) - 1)

    try:
        dataset = spglib.get_symmetry_dataset(
            (ref_lattice, np.array(exp_positions), np.array(exp_numbers)),
            symprec=1e-3, hall_number=hall_no
        )
    except Exception:
        dataset = None
    # Fallback: partial structure may have higher apparent symmetry than the parent SG
    # (e.g. 2 identical atoms at (0,0,0)+(½,½,½) look body-centred), so retry without
    # the hall_number constraint and let spglib auto-detect.
    if dataset is None:
        try:
            dataset = spglib.get_symmetry_dataset(
                (ref_lattice, np.array(exp_positions), np.array(exp_numbers)),
                symprec=1e-3
            )
        except Exception:
            dataset = None
    if dataset is None:
        return ['?' for _ in parent_positions]
    wyckoffs = (dataset.wyckoffs if hasattr(dataset, 'wyckoffs')
                else dataset['wyckoffs'])
    return [str(wyckoffs[idx]) for idx in orig_exp_idx]


def _get_wyckoff_groups(it_number, positions, labels):
    """Group atoms by (species_label, wyckoff_letter).

    Returns list of (group_name, atom_indices) in order of first appearance.
    Atoms at the same Wyckoff site with the same element form one orbit.
    """
    wyckoffs = _get_wyckoff_sites(it_number, positions, labels)
    seen_keys = []
    key_to_indices = {}
    for i, (lbl, wy) in enumerate(zip(labels, wyckoffs)):
        key = (lbl, wy)
        if key not in key_to_indices:
            seen_keys.append(key)
            key_to_indices[key] = []
        key_to_indices[key].append(i)
    result = []
    for key in seen_keys:
        species, wy = key
        result.append((f"{species} ({wy})", np.array(key_to_indices[key])))
    return result


def _print_sg_info(it_number: int):
    """Section (1): Parent space group info."""
    hall = get_hall_number(it_number)
    sg_type = spglib.get_spacegroup_type(hall)

    def _f(key):
        return sg_type[key] if hasattr(sg_type, '__getitem__') else getattr(sg_type, key)

    intl      = _f('international_short')
    ptgp_schf = _f('pointgroup_schoenflies')
    ptgp_intl = _f('pointgroup_international')
    choice    = _f('choice')

    symbol = f"{intl}  [origin choice {choice}]" if choice else intl

    print("(1) PARENT SPACE GROUP")
    print(f"  IT Number      : {it_number}")
    print(f"  Symbol         : {symbol}")
    print(f"  Crystal system : {_crystal_system(it_number)}")
    print(f"  Point group    : {ptgp_schf}  ({ptgp_intl})")
    print()


def _print_propagation_and_lg(kpoint, mapping_little_group, rotations, translations,
                               it_number=None):
    """Sections (2)+(3): Propagation vector and little group operation list.

    Only the |G⁰_k| unique coset representatives {R|t} are listed — one per
    distinct rotation R.  For centered Bravais lattices G_k contains additional
    centering copies {R|t+t_c} that are omitted here because small representations
    of G_k depend only on the coset structure (G_k / T), not on the choice of t_c.
    """
    def _fk(v):
        return f"{v:.4g}" if v != 0 else "0"
    k_str   = f"({_fk(kpoint[0])}, {_fk(kpoint[1])}, {_fk(kpoint[2])})"
    k_label = irrep_label.kpoint_label(kpoint, it_number=it_number)

    print("(2) PROPAGATION VECTOR")
    print(f"  k = {k_str}  →  {k_label} point")
    print()

    co_group = _little_group_point_group(mapping_little_group, rotations, translations,
                                         kpoint)
    kpt      = np.asarray(kpoint, dtype=float)
    is_gamma = np.allclose(kpt % 1.0, 0.0, atol=1e-4)
    dedup    = _dedup_lg_ops(mapping_little_group, rotations)
    n_coset  = len(dedup)          # = |G⁰_k|

    if is_gamma and it_number is not None:
        hall    = get_hall_number(it_number)
        sg_type = spglib.get_spacegroup_type(hall)
        def _f(key):
            return sg_type[key] if hasattr(sg_type, '__getitem__') else getattr(sg_type, key)
        sg_intl = _f('international_short')
        gk_info = f"G_k = G = {sg_intl} (#{it_number}) at k=Γ"
    else:
        gk_info = f"G_k ⊂ G,  little co-group G⁰_k: {co_group}"

    print(f"(3) LITTLE GROUP  G_k  (|G⁰_k| = {n_coset} coset representatives)")
    print(f"  {gk_info}")
    if is_gamma:
        print(f"  Little co-group G⁰_k (point group of wave vector): {co_group}")

    # Column widths for {R|t} Seitz symbol and xyz form
    seitz_w = max(20, max(len(_seitz(rotations[idx], translations[idx]))
                          for _, idx in dedup) + 1)
    xyz_w   = max(16, max(len(_seitz_to_xyz(rotations[idx], translations[idx]))
                          for _, idx in dedup) + 1)
    print(f"  {'#':>3}  {'{{R | t}}':<{seitz_w}}  {'xyz form':<{xyz_w}}")
    print("  " + "-" * (6 + seitz_w + 2 + xyz_w))

    for i, (_i_lg, idx) in enumerate(dedup):
        R = rotations[idx]
        t = translations[idx]
        print(f"  {i+1:>3}  {_seitz(R, t):<{seitz_w}}  {_seitz_to_xyz(R, t):<{xyz_w}}")
    print()


def _print_wyckoff_and_permutation(it_number, parent_positions, atom_labels,
                                    perm_data, kpoint,
                                    mapping_little_group, rotations, translations,
                                    mode='magnetic'):
    """Sections (4)+(5): Wyckoff sites and permutation of the Wyckoff orbit."""
    atom_mappings, _chi_perm, _chi_axial = perm_data
    N = len(parent_positions)

    wyckoffs = _get_wyckoff_sites(it_number, parent_positions, atom_labels)

    atom_header = "MAGNETIC ATOMS" if mode == 'magnetic' else "ATOMS (DISPLACIVE MODE)"
    print(f"(4) {atom_header} — WYCKOFF SITES")
    print(f"  {'#':>3}  {'Element':<8}  {'Wyckoff':<8}  Fractional coordinates")
    print("  " + "-" * 56)
    for i, (r, lbl, wy) in enumerate(zip(parent_positions, atom_labels, wyckoffs)):
        print(f"  {i+1:>3}  {lbl:<8}  {wy:<8}  "
              f"({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})")
    print()

    # Permutation table — one row per unique coset representative {R|t}
    dedup   = _dedup_lg_ops(mapping_little_group, rotations)
    n_coset = len(dedup)

    # Column widths
    dw = max(3, len(str(N)) + 2)  # destination index column width
    sw = 12                        # lattice-shift column width
    op_w = max(20, max(len(_seitz(rotations[idx], translations[idx]))
                       for _, idx in dedup) + 1)

    hdr_dest  = "".join(f"{f'{n+1}→':>{dw}}" for n in range(N))
    hdr_shift = "".join(f"{'L'+str(n+1):<{sw}}" for n in range(N))

    print("(5) PERMUTATION OF WYCKOFF SITE UNDER G_k")
    print(f"  {'#':>3}  {'{{R | t}}':<{op_w}}  {hdr_dest}   {hdr_shift}")
    print("  " + "-" * (6 + op_w + dw * N + 3 + sw * N))

    for i, (i_lg, idx) in enumerate(dedup):
        op_str   = _seitz(rotations[idx], translations[idx])
        mappings = atom_mappings[i_lg]
        dest_str  = "".join(
            f"{m[0]+1:>{dw}}" if m[0] >= 0 else f"{'?':>{dw}}"
            for m in mappings
        )
        shift_str = "".join(
            f"({m[1][0]},{m[1][1]},{m[1][2]})".ljust(sw)
            for m in mappings
        )
        print(f"  {i+1:>3}  {op_str:<{op_w}}  {dest_str}   {shift_str}")
    print()


def _print_representation_characters(mapping_little_group, rotations, translations,
                                      perm_data, chi_rep, verbose=False,
                                      mode='magnetic'):
    """Section (6): Per-operation χ_perm, χ_axial, χ_rep (one row per unique {R|t}).

    mode='magnetic': chi_axial = det(R)·Tr(R), output labelled χ_mag
    mode='displacive':   chi_axial = Tr(R),         output labelled χ_disp
    """
    _atom_mappings, chi_perm, chi_axial = perm_data

    def _fmtc(z, w=9):
        """Format a complex (or real) character value compactly."""
        if abs(z.imag) < 5e-4:
            v  = z.real
            iv = int(round(v))
            s  = str(iv) if abs(v - iv) < 5e-4 else f"{v:.4f}"
        else:
            s = f"{z.real:.3f}{z.imag:+.3f}j"
        return f"{s:>{w}}"

    dedup  = _dedup_lg_ops(mapping_little_group, rotations)
    op_w   = max(20, max(len(_seitz(rotations[idx], translations[idx]))
                         for _, idx in dedup) + 1)

    rep_label = "χ_mag" if mode == 'magnetic' else "χ_disp"

    print("(6) REPRESENTATION ANALYSIS")
    print("  χ_perm(g)  = Σ_{fixed atoms} exp(−2πi k·L)")
    if mode == 'magnetic':
        print("  χ_axial(g) = det(R) · Tr(R)")
        print(f"  χ_mag(g)   = χ_perm × χ_axial")
    else:
        print("  χ_polar(g) = Tr(R)  [polar vector, no det factor]")
        print(f"  χ_disp(g)  = χ_perm × χ_polar")
    print()

    if mode == 'magnetic':
        print(f"  {'#':>3}  {'{{R | t}}':<{op_w}}  {'χ_perm':>9}  {'det(R)':>7}  {'Tr(R)':>6}  "
              f"{'χ_axial':>9}  {rep_label:>9}")
        print("  " + "-" * (6 + op_w + 62))
    else:
        print(f"  {'#':>3}  {'{{R | t}}':<{op_w}}  {'χ_perm':>9}  {'Tr(R)':>6}  "
              f"{'χ_polar':>9}  {rep_label:>9}")
        print("  " + "-" * (6 + op_w + 54))

    for i, (i_lg, idx) in enumerate(dedup):
        R     = rotations[idx].astype(float)
        det_R = int(round(np.linalg.det(R)))
        tr_R  = int(round(np.trace(R)))
        cp    = chi_perm[i_lg]
        ca    = chi_axial[i_lg]
        cm    = chi_rep[idx]
        op_str = _seitz(rotations[idx], translations[idx])
        if mode == 'magnetic':
            print(f"  {i+1:>3}  {op_str:<{op_w}}  {_fmtc(cp):>9}  {det_R:>+7}  {tr_R:>6}  "
                  f"  {ca:>+9.4f}  {_fmtc(cm):>9}")
        else:
            print(f"  {i+1:>3}  {op_str:<{op_w}}  {_fmtc(cp):>9}  {tr_R:>6}  "
                  f"  {ca:>+9.4f}  {_fmtc(cm):>9}")
    print()


def _print_character_table(irreps, parities, rotations, translations,
                            mapping_little_group, kpoint, it_number,
                            active_irreps=None, bilbao_labels=None,
                            mode='magnetic'):
    """Section (7): Small representations of the little group G_k (space group irreps).

    Columns = conjugacy classes of G_k as a space group (sorted E, C6…C2, i, S3…m).
    Rows    = small representations Γ^α_k (spgrep index order).
    Values  = χ^α({R|t}) = Tr[Γ^α_k({R|t})] for a representative {R|t} from each class.
    """
    # Use only the unique coset representatives (one per rotation R) so that
    # the conjugacy classes are those of the little co-group G⁰_k.
    dedup   = _dedup_lg_ops(mapping_little_group, rotations)
    dedup_ilgs = [i_lg for i_lg, _idx in dedup]
    ops_R = np.array([rotations[mapping_little_group[i_lg]].astype(float) for i_lg in dedup_ilgs])
    ops_t = np.array([translations[mapping_little_group[i_lg]]             for i_lg in dedup_ilgs])

    classes = _conjugacy_classes_lg(ops_R, ops_t)

    def _cls_key(cls):
        tr  = int(round(np.trace(ops_R[cls[0]])))
        det = int(round(np.linalg.det(ops_R[cls[0]])))
        return (0 if det == 1 else 1, -tr if det == 1 else tr)

    classes.sort(key=_cls_key)

    # Column labels: use ITA sym (with axis) + disambiguating suffix
    rtype_total = Counter(_op_symbol_ita(ops_R[c[0]]) for c in classes)
    rtype_seen: Counter = Counter()
    col_labels = []
    for cls in classes:
        rt    = _op_symbol_ita(ops_R[cls[0]])
        n_ops = len(cls)
        rtype_seen[rt] += 1
        sfx = chr(ord('a') + rtype_seen[rt] - 1) if rtype_total[rt] > 1 else ''
        col_labels.append(f"{n_ops}·{rt}{sfx}" if n_ops > 1 else f"{rt}{sfx}")

    def _fmt(z):
        if abs(z.imag) < 5e-4:
            v  = z.real
            iv = int(round(v))
            return str(iv) if abs(v - iv) < 5e-4 else f"{v:.4f}"
        return f"{z.real:.3f}{z.imag:+.3f}j"

    lw  = 10
    cw  = max(6, max(len(l) for l in col_labels) + 1)

    active_set = {idx for idx, *_ in (active_irreps or [])} if mode == 'magnetic' else set()

    print("(7) SMALL REPRESENTATIONS OF G_k  (space group irreps, χ = Tr[Γ_k({R|t})])")
    hdr = f"  {'Irrep':<{lw}}  d  |"
    for lbl in col_labels:
        hdr += f" {lbl:^{cw}}"
    print(hdr)
    print("  " + "-" * lw + "-----+" + ("-" * (cw + 1)) * len(classes))

    for alpha, irrep in enumerate(irreps):
        d   = irrep[0].shape[0]
        p   = parities[alpha] if parities else ''
        if bilbao_labels and alpha in bilbao_labels:
            lbl = bilbao_labels[alpha]
        else:
            lbl = irrep_label.irrep_name(kpoint, it_number, alpha, d, p)
        mark = " <--" if alpha in active_set else ""
        row = f"  {lbl:<{lw}}  {d}  |"
        for cls in classes:
            chi = np.trace(irrep[dedup_ilgs[cls[0]]])
            row += f" {_fmt(chi):^{cw}}"
        row += mark
        print(row)
    print()


def _decomp_str(n_mu_array, irreps, parities, bilbao_labels, kpoint, it_number):
    """Return a compact decomposition string like '1·mGM5- ⊕ 2·mGM1+'."""
    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    def _nstr(n):
        return str(int(round(n))) if abs(n - round(n)) < 0.01 else f"{n:.3f}"

    active = [(i, n_mu_array[i]) for i in range(len(n_mu_array)) if abs(n_mu_array[i]) > 0.001]
    terms = [f"{_nstr(n)}·{_lbl(idx)}" for idx, n in active]
    return "  ⊕  ".join(terms) if terms else "0"


def _print_decomposition_extended(
        n_mu_mag, n_mu_perm_mag,
        n_mu_mech, n_mu_perm_all,
        n_mu_axial, n_mu_polar,
        irreps, bilbao_labels, identified, identified_mech,
        kpoint, it_number, parities,
        wyckoff_mech=None, wyckoff_perm=None):
    """Section (8): Extended decomposition with Γ_perm, Γ_axial, Γ_polar, Γ_mag, Γ_mech.

    n_mu_mag / n_mu_perm_mag may be None for displacive-only runs.
    n_mu_mech / n_mu_perm_all may be None for magnetic-only runs.
    wyckoff_mech: list of (group_name, n_mu_wyck) for per-Wyckoff-site Γ_mech breakdown.
    wyckoff_perm: list of (group_name, n_mu_wyck) for per-Wyckoff-site Γ_perm breakdown.
    """
    def _ds(n_mu):
        return _decomp_str(n_mu, irreps, parities, bilbao_labels, kpoint, it_number)

    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    def _nstr(n):
        return str(int(round(n))) if abs(n - round(n)) < 0.01 else f"{n:.3f}"

    SEP = "─" * 66

    print("(8) DECOMPOSITION INTO SMALL REPRESENTATIONS OF G_k\n")
    print(f"  {'Representation':<18}  Decomposition")
    print(f"  {SEP}")

    # ── Magnetic block ────────────────────────────────────────────────────────
    if n_mu_mag is not None:
        print(f"  Γ_perm  [mag]    = {_ds(n_mu_perm_mag)}")
        print(f"  Γ_axial           = {_ds(n_mu_axial)}        (χ = det(R)·Tr(R))")
        print(f"  Γ_mag   = Γ_perm ⊗ Γ_axial")
        print(f"           = {_ds(n_mu_mag)}    ← primary result")
        print()

    # ── Displacive/mechanical block ─────────────────────────────────────────────
    if n_mu_mech is not None:
        print(f"  Γ_perm  [all]    = {_ds(n_mu_perm_all)}")
        if wyckoff_perm:
            name_w = max(len(g) for g, _ in wyckoff_perm)
            print(f"  Per-Wyckoff-site contributions to Γ_perm:")
            for group_name, n_mu_wyck in wyckoff_perm:
                print(f"    Γ_perm({group_name:<{name_w}}) = {_ds(n_mu_wyck)}")
        print(f"  Γ_polar           = {_ds(n_mu_polar)}        (χ = Tr(R))")
        print(f"  Γ_mech  = Γ_perm ⊗ Γ_polar")
        print(f"           = {_ds(n_mu_mech)}  [total]")
        print()
        if wyckoff_mech:
            # Determine the maximum group-name width for alignment
            name_w = max(len(g) for g, _ in wyckoff_mech)
            print(f"  Per-Wyckoff-site contributions to Γ_mech:")
            for group_name, n_mu_wyck in wyckoff_mech:
                print(f"    Γ_mech({group_name:<{name_w}}) = {_ds(n_mu_wyck)}")
            print()

    # ── Reference Γ_axial for displacive-only mode ────────────────────────────────
    if n_mu_mag is None and n_mu_mech is not None:
        print(f"  Γ_axial           = {_ds(n_mu_axial)}        (χ = det(R)·Tr(R))   [reference]")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    has_mag  = n_mu_mag  is not None
    has_mech = n_mu_mech is not None
    # displacive-only mode: omit "← ACTIVE" markers (all active irreps are displacive branches)
    displacive_only = (not has_mag and has_mech)

    n_mu_a = n_mu_mag  if has_mag  else np.zeros(len(irreps))
    n_mu_m = n_mu_mech if has_mech else np.zeros(len(irreps))

    all_active = sorted(set(
        [i for i in range(len(irreps)) if abs(n_mu_a[i]) > 0.001] +
        [i for i in range(len(irreps)) if abs(n_mu_m[i]) > 0.001]
    ), key=lambda i: _lbl(i))

    if not all_active:
        return

    eta_map   = {idx: eta for idx, _n, eta, _ in (identified or [])}
    best_mag  = identified[0][0]       if identified       else None
    best_mech = identified_mech[0][0]  if identified_mech  else None

    if has_mag and has_mech:
        print(f"  {'Irrep':<12}  {'dim':>4}  {'n_μ(mag)':>9}  {'n_μ(mech)':>10}  {'η':>7}")
        print("  " + "─" * 50)
    elif has_mag:
        print(f"  {'Irrep':<12}  {'dim':>4}  {'n_μ(mag)':>9}  {'η':>7}")
        print("  " + "─" * 40)
    else:
        print(f"  {'Irrep':<12}  {'dim':>4}  {'n_μ(mech)':>10}")
        print("  " + "─" * 30)

    for idx in all_active:
        d   = irreps[idx][0].shape[0]
        lbl = _lbl(idx)
        eta = eta_map.get(idx, 0.0)
        n_m = n_mu_a[idx]
        n_p = n_mu_m[idx]

        if has_mag and has_mech:
            marker = "  ← ACTIVE" if idx == best_mag else ""
            print(f"  {lbl:<12}  {d:>4}  {_nstr(n_m):>9}  {_nstr(n_p):>10}  {eta:>7.3f}{marker}")
        elif has_mag:
            marker = "  ← ACTIVE" if idx == best_mag else ""
            print(f"  {lbl:<12}  {d:>4}  {_nstr(n_m):>9}  {eta:>7.3f}{marker}")
        else:
            print(f"  {lbl:<12}  {d:>4}  {_nstr(n_p):>10}")
    print()


def _print_decomposition(active_irreps, irreps, n_mu_array, bilbao_labels,
                          identified, kpoint, it_number, parities, mode='magnetic'):
    """Section (8): Decomposition of Γ_mag/Γ_mech into small representations of G_k."""
    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    def _nstr(n):
        return str(int(round(n))) if abs(n - round(n)) < 0.01 else f"{n:.3f}"

    terms = [f"{_nstr(n)}·{_lbl(idx)}" for idx, n in active_irreps]

    gamma_label = "Γ_mag" if mode == 'magnetic' else "Γ_mech"

    print("(8) DECOMPOSITION INTO SMALL REPRESENTATIONS OF G_k")
    if terms:
        print(f"  {gamma_label}  =  " + "  ⊕  ".join(terms))
    else:
        print(f"  {gamma_label}  =  0  (no active small representations found)")
    print()

    if active_irreps:
        eta_map  = {idx: eta for idx, _n, eta, _ in (identified or [])}
        best_idx = identified[0][0] if identified else None

        print(f"  {'Irrep':<12}  {'dim':>4}  {'n_μ':>5}  {'η':>7}")
        print("  " + "-" * 36)
        for idx, n in active_irreps:
            d   = irreps[idx][0].shape[0]
            lbl = _lbl(idx)
            eta = eta_map.get(idx, 0.0)
            marker = "  ← ACTIVE" if idx == best_idx else ""
            print(f"  {lbl:<12}  {d:>4}  {_nstr(n):>5}  {eta:>7.3f}{marker}")
        print()


def _scale_to_integers(v, tol=1e-4):
    """Scale a real vector so its smallest nonzero |component| = 1.

    Returns an integer array when the rescaled values are all close to
    integers (|error| < 0.02 and max ≤ 20); otherwise normalises by the
    maximum absolute value.
    """
    v_real = np.real(v).copy()
    nonzero_abs = np.abs(v_real[np.abs(v_real) > tol])
    if len(nonzero_abs) == 0:
        return v_real
    scale = np.min(nonzero_abs)
    v_sc  = v_real / scale
    v_rnd = np.round(v_sc)
    if np.allclose(v_sc, v_rnd, atol=0.02) and np.max(np.abs(v_rnd)) <= 20:
        return v_rnd.astype(int)
    mx = np.max(np.abs(v_real))
    return v_real / mx if mx > tol else v_real


def _fmt_bv_val(x):
    """Format one basis-vector component as a compact integer or decimal."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    iv = int(round(float(x)))
    if abs(float(x) - iv) < 0.02:
        return str(iv)
    return f"{float(x):.3f}"


def _print_basis_vectors(active_irreps, all_basis, atom_labels, parent_positions,
                          identified, bilbao_labels, kpoint, it_number,
                          parities, irreps, mode='magnetic', wyckoff_groups=None):
    """Section (9): Symmetry-adapted basis vectors.

    mode='magnetic': one monolithic table with all atoms; columns m1a m1b ...;
                     IR label shown on every row; active irrep marked.
    mode='displacive' with wyckoff_groups: one sub-table per Wyckoff site;
                     columns show only atoms in that site; IR label on every row;
                     BVs with all-zero components at that site are skipped.
    BV indices ψ_n are numbered globally across all active irreps.
    """
    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    N        = len(parent_positions)
    eta_map  = {idx: eta for idx, _n, eta, _m in (identified or [])}
    best_idx = identified[0][0] if identified else None

    # Pre-assign global BV numbers
    bv_numbers = {}   # (irrep_idx, local_i) -> global_num
    bv_count = 0
    for idx, _n in active_irreps:
        for i in range(len(all_basis[idx])):
            bv_count += 1
            bv_numbers[(idx, i)] = bv_count

    axes = ['a', 'b', 'c']

    # ── Per-Wyckoff mode (displacive) ────────────────────────────────────────────
    if wyckoff_groups is not None:
        ir_w = max(8, max((len(_lbl(idx)) for idx, _ in active_irreps), default=8))

        print("(9) BASIS VECTORS PER WYCKOFF SITE  [displacive mode]\n")

        for group_name, site_indices in wyckoff_groups:
            n_site = len(site_indices)
            # Number atoms within this site: Mn1, Mn2, ...
            site_elem = atom_labels[site_indices[0]]
            site_numbered = [f"{atom_labels[si]}{j+1}" for j, si in enumerate(site_indices)]
            col_hdrs = [f"{sn}{ax}" for sn in site_numbered for ax in axes]
            cw = max(4, max(len(h) for h in col_hdrs) + 1)
            bv_w = max(4, len(f"ψ{bv_count}") + 1)

            # Site header
            pos_strs = [f"({parent_positions[si, 0]:.4f},{parent_positions[si, 1]:.4f},"
                        f"{parent_positions[si, 2]:.4f})" for si in site_indices]
            atom_info = "  ".join(f"{site_numbered[j]} {pos_strs[j]}"
                                   for j in range(n_site))
            print(f"  ── {group_name}:  {atom_info}")

            hdr = f"  {'IR':<{ir_w}}  {'BV':<{bv_w}}"
            sep = f"  {'':-<{ir_w}}  {'':-<{bv_w}}"
            for h in col_hdrs:
                hdr += f"  {h:>{cw}}"
                sep += f"  {'':->{cw}}"
            print(hdr)
            print(sep)

            any_row = False
            for idx, _n in active_irreps:
                lbl = _lbl(idx)
                first_at_site = True
                for i, v in enumerate(all_basis[idx]):
                    # Extract components for atoms in this Wyckoff group
                    site_vals = np.array([v[3 * si + c]
                                          for si in site_indices for c in range(3)])
                    if np.all(np.abs(site_vals) < 1e-4):
                        continue  # this BV has no weight at this site
                    global_bv = bv_numbers[(idx, i)]
                    scaled = _scale_to_integers(np.real(site_vals))
                    bv_label = f"ψ{global_bv}"
                    ir_field = lbl if first_at_site else ""
                    row = f"  {ir_field:<{ir_w}}  {bv_label:<{bv_w}}"
                    for val in scaled:
                        row += f"  {_fmt_bv_val(val):>{cw}}"
                    print(row)
                    any_row = True
                    first_at_site = False

            if not any_row:
                print("  (no basis vectors at this site)")
            print()

        return

    # ── Monolithic mode (magnetic, or displacive without grouping) ───────────────
    col_prefix = 'm' if mode == 'magnetic' else 'u'
    comp_hdrs  = [f"{col_prefix}{a+1}{ax}" for a in range(N) for ax in axes]
    cw  = max(4, max(len(h) for h in comp_hdrs) + 1)
    ir_w = max(8, max((len(_lbl(idx)) for idx, _ in active_irreps), default=8))
    bv_w = max(3, len(f"ψ{bv_count}") + 1)

    print("(9) BASIS VECTORS OF ACTIVE SMALL REPRESENTATIONS\n")

    hdr = f"  {'IR':<{ir_w}}  {'BV':<{bv_w}}"
    sep = f"  {'':-<{ir_w}}  {'':-<{bv_w}}"
    for h in comp_hdrs:
        hdr += f"  {h:>{cw}}"
        sep += f"  {'':->{cw}}"
    print(hdr)
    print(sep)

    for idx, _n in active_irreps:
        lbl        = _lbl(idx)
        eta        = eta_map.get(idx, 0.0)
        basis_vecs = all_basis[idx]
        is_active  = (idx == best_idx)

        if not basis_vecs:
            row = f"  {lbl:<{ir_w}}  {'—':<{bv_w}}  (no basis vectors found)"
            if is_active and mode == 'magnetic':
                row += f"  ← ACTIVE  η={eta:.3f}"
            print(row)
            continue

        for i, v in enumerate(basis_vecs):
            global_bv = bv_numbers[(idx, i)]
            v_sc      = _scale_to_integers(np.real(v))
            bv_label  = f"ψ{global_bv}"
            ir_field  = lbl if i == 0 else ""
            row = f"  {ir_field:<{ir_w}}  {bv_label:<{bv_w}}"
            for val in v_sc:
                row += f"  {_fmt_bv_val(val):>{cw}}"
            if i == 0 and is_active and mode == 'magnetic':
                row += f"  ← ACTIVE  η={eta:.3f}"
            print(row)

    print()


def _print_moment_consistency(active_irreps, all_basis, parent_magmoms,
                               atom_labels, parent_positions, identified,
                               bilbao_labels, kpoint, it_number, parities, irreps):
    """Section (10): Verify actual moments lie in the active irrep subspace.

    Decomposes M_actual in the active irrep's normalized basis,
    reconstructs M_rec = Σ c_i ψ_i, and shows a per-atom comparison table.

    The display coefficients α_i satisfy M ≈ Σ α_i φ_i where φ_i are the
    integer-scaled basis vectors from section (9); they equal
        α_i = <ψ_i | M>  ×  min_nonzero(|ψ_i|)
    which gives physically meaningful amplitudes (e.g. −3.600 for the
    dominant mode in CuMnAs with |m_Mn| = 3.6 μB/f.u.).
    """
    if not identified:
        return

    best_idx = identified[0][0]

    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    label      = _lbl(best_idx)
    N          = len(parent_positions)
    M_flat     = parent_magmoms.flatten().astype(float)
    basis_vecs = all_basis[best_idx]

    # Global ψ index offset: count BVs in active irreps that precede best_idx
    bv_offset = 0
    for idx, _ in active_irreps:
        if idx == best_idx:
            break
        bv_offset += len(all_basis[idx])

    print(f"(10) MOMENT–IRREP CONSISTENCY  (active small representation: {label})\n")

    if not basis_vecs:
        print(f"  No basis vectors available for {label}; cannot verify.")
        print()
        return

    # Coefficients in the normalised basis: c_i = <ψ_i | M>
    coeffs_norm = [float(np.real(np.vdot(v, M_flat))) for v in basis_vecs]

    # Display coefficients α_i = c_i × min_nonzero(|ψ_i|)
    # so that M ≈ Σ α_i φ_i  (φ_i = integer-scaled display vectors)
    def _disp_coeff(v, c, tol=1e-4):
        nz = np.abs(v.real[np.abs(v.real) > tol])
        return c * float(np.min(nz)) if len(nz) else 0.0

    alphas = [_disp_coeff(v, c) for v, c in zip(basis_vecs, coeffs_norm)]

    # Reconstructed moment vector from the active-irrep basis
    M_rec = np.real(sum(c * v for c, v in zip(coeffs_norm, basis_vecs)))

    # Decomposition formula
    terms = [f"({a:+.4f})·ψ{bv_offset + i + 1}"
             for i, a in enumerate(alphas)]
    print("  M  =  " + "  +  ".join(terms))
    print()

    # Per-atom table
    cw = 28
    print(f"  {'Atom':<4}  {'Position':>24}  "
          f"{'m_actual':>{cw}}  {'m_rec':>{cw}}  {'|δ|':>7}")
    print("  " + "-" * (4 + 2 + 24 + 2 + cw + 2 + cw + 2 + 7))

    for i in range(N):
        r       = parent_positions[i]
        m_act   = parent_magmoms[i]
        m_rec_i = M_rec[3*i:3*i+3]
        delta   = np.linalg.norm(m_act - m_rec_i)
        pos_str = f"({r[0]:.4f},{r[1]:.4f},{r[2]:.4f})"
        act_str = f"[{m_act[0]:+.4f},{m_act[1]:+.4f},{m_act[2]:+.4f}]"
        rec_str = f"[{m_rec_i[0]:+.4f},{m_rec_i[1]:+.4f},{m_rec_i[2]:+.4f}]"
        print(f"  {atom_labels[i]:<4}  {pos_str:>24}  "
              f"{act_str:>{cw}}  {rec_str:>{cw}}  {delta:>7.4f}")

    print()
    M_norm   = np.linalg.norm(M_flat)
    residual = np.linalg.norm(M_flat - M_rec) / M_norm if M_norm > 1e-10 else 0.0
    check    = "✓" if residual < 1e-3 else "✗"
    print(f"  ‖M - M_rec‖/‖M‖ = {residual:.4f}  {check}")
    print()


def _format_sk_constraints(irrep_idx, n_mu, eta, all_basis, atom_labels, parent_positions,
                            bilbao_labels, kpoint, it_number, parities, irreps,
                            mode='magnetic'):
    """Print Fourier coefficient constraints Sk(j) for the given irrep.

    For irrep with basis vectors ψ_1 … ψ_d (each length 3N), the Fourier
    coefficient at atom j is M_j = Σ_i α_i · ψ_i[3j:3j+3].
    Parameter names: α_1=u, α_2=v, α_3=w, ...
    mode='magnetic': header says "active irrep"; mode='displacive': "irrep".
    """
    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    label      = _lbl(irrep_idx)
    basis_vecs = all_basis[irrep_idx] if irrep_idx < len(all_basis) else []
    N          = len(parent_positions)
    d          = len(basis_vecs)

    if d == 0:
        return

    param_names = ['u', 'v', 'w', 'p', 'q', 'r', 's', 't', 'a', 'b'][:d]

    # Scale each basis vector to smallest-integer representation
    scaled = [_scale_to_integers(np.real(bv)) for bv in basis_vecs]

    prefix = "active irrep" if mode == 'magnetic' else "irrep"
    print(f"  Fourier coefficient constraints  "
          f"({prefix} {label},  n={int(round(n_mu))},  η={eta:.3f}):\n")

    lbl_w = max(6, max(len(l) for l in atom_labels))
    print(f"    {'Atom':<{lbl_w}}  {'Fractional position':<30}  Sk constraint")
    print("    " + "─" * (lbl_w + 2 + 30 + 2 + 30))

    free_params = []
    for j in range(N):
        r = parent_positions[j]
        components = []
        for c in range(3):
            terms = []
            for sc, pname in zip(scaled, param_names):
                coef = float(sc[3 * j + c])
                iv = int(round(coef))
                if abs(coef - iv) > 0.02:
                    if abs(coef) > 1e-4:
                        terms.append(f"{coef:+.3f}{pname}")
                else:
                    if iv == 0:
                        continue
                    elif iv == 1:
                        terms.append(f"+{pname}")
                    elif iv == -1:
                        terms.append(f"-{pname}")
                    else:
                        terms.append(f"{iv:+d}{pname}")
            expr = ''.join(terms).lstrip('+') if terms else '0'
            components.append(expr)

        pos_str = f"({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})"
        sk_str  = f"({', '.join(components)})"
        print(f"    {atom_labels[j]:<{lbl_w}}  {pos_str:<30}  {sk_str}")

    # Collect free parameters: any param whose basis vector has at least one nonzero component
    for i, pname in enumerate(param_names):
        if np.any(np.abs(scaled[i]) > 1e-4):
            free_params.append(pname)

    print()
    print(f"    Free parameters: {', '.join(free_params)}")
    print()


def _print_validation(fields, identified, bilbao_labels, kpoint, it_number,
                       parities, irreps):
    """Print validation line comparing identified irrep to stored _irrep_id."""
    expected_id = fields.get('expected_irrep_id')

    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    print("Validation:")
    if identified:
        best_idx = identified[0][0]
        best_label = _lbl(best_idx)
        if expected_id:
            match = "  \u2713" if best_label == expected_id else \
                    f"  \u2717 (identified: {best_label})"
            print(f"  Stored _irrep_id = {expected_id}{match}")
        else:
            print(f"  Identified: {best_label}  (no _irrep_id stored in mCIF)")
    else:
        print(f"  Stored _irrep_id = {expected_id}  (no active irrep identified)")
    print()


def run_analysis(mcif_path: str, verbose: bool = False, output_file: str = None,
                 displacive_pass: bool = True):
    """Run the full magnetic irrep analysis pipeline on *mcif_path*.

    displacive_pass=True (default): also compute Γ_mech and per-Wyckoff displacive decomposition.
    displacive_pass=False: magnetic analysis only (faster; no displacive block in output).
    """

    # ── 1. Parse fields ───────────────────────────────────────────────────────
    fields = parse_mcif.parse_mcif_fields(mcif_path)
    kpoint = parse_mcif.parse_kvector(fields['kvector_str'])

    child_M, child_t = parse_mcif.parse_transform(fields['child_transform_str'])
    if 'parent_transform_str' in fields:
        parent_M, parent_t = parse_mcif.parse_transform(fields['parent_transform_str'])
    else:
        parent_M, parent_t = np.eye(3), np.zeros(3)

    it_number = fields['it_number']

    _dbg(verbose, f"IT number: {it_number}")
    _dbg(verbose, f"k-vector: {kpoint}")
    _dbg(verbose, f"Child transform  P = {child_M.tolist()},  p = {child_t.tolist()}")
    _dbg(verbose, f"Parent transform P = {parent_M.tolist()},  p = {parent_t.tolist()}")

    # ── 2. Extract nonzero-moment atoms + species labels ─────────────────────
    structure = parse_mcif.get_magnetic_structure(mcif_path)
    # Parse 3D crystal-axis moments directly from mCIF (avoids pymatgen reading
    # _atom_site_moment.magnitude as a scalar instead of crystalaxis_x/y/z vector)
    gemmi_moments = parse_mcif.parse_moments_from_mcif(mcif_path)
    # Map explicitly listed atom positions to their 3D moments.  This lets us
    # distinguish the reference atom from its symmetry-generated equivalents
    # (which share the same site label in pymatgen but may have opposite moment sign).
    gemmi_explicit_pos = parse_mcif.parse_explicit_atom_positions(mcif_path)
    explicit_pos_to_moment = {
        lbl: (np.array(pos) % 1.0, gemmi_moments[lbl])
        for lbl, pos in gemmi_explicit_pos.items()
        if lbl in gemmi_moments
    }

    mag_positions = []
    magmoms       = []
    atom_labels   = []

    for site in structure:
        site_pos = site.frac_coords % 1.0
        site_label = getattr(site, 'label', None) or site.properties.get('label', '')
        raw_m = site.properties.get("magmom")

        # 1. Position-based match: explicitly listed atom → use gemmi 3D vector as-is
        m_vec = None
        for _lbl, (ep, em) in explicit_pos_to_moment.items():
            if np.allclose(site_pos, ep, atol=1e-3):
                m_vec = em
                break

        if m_vec is None and site_label and site_label in gemmi_moments:
            # 2. Label match for symmetry-generated equivalent: apply pymatgen's
            #    signed scalar to re-orient the reference direction vector.
            #    (Covers collinear AFM where equiv atoms have opposite sign.)
            m_ref = gemmi_moments[site_label]
            ref_norm = np.linalg.norm(m_ref)
            if ref_norm > 1e-8 and raw_m is not None:
                try:
                    scalar = float(raw_m)
                    m_vec = (scalar / ref_norm) * m_ref
                except (TypeError, ValueError):
                    m_vec = m_ref
            else:
                m_vec = m_ref

        if m_vec is None:
            # 3. Fall back to pymatgen's raw magmom property
            if raw_m is None:
                raw_m = getattr(site, "magmom", None)
            if raw_m is None:
                continue
            if hasattr(raw_m, 'moment'):
                m_vec = np.array(raw_m.moment)
            elif hasattr(raw_m, 'global_moment'):
                m_vec = np.array(raw_m.global_moment)
            else:
                m_vec = np.array(raw_m)
            if m_vec.shape == () or m_vec.shape == (1,):
                m_vec = np.array([0.0, 0.0, float(m_vec)])

        if np.linalg.norm(m_vec) < 0.01:
            continue

        mag_positions.append(site.frac_coords)
        magmoms.append(m_vec)
        atom_labels.append(site.species_string)

    mag_positions = np.array(mag_positions)
    magmoms       = np.array(magmoms)

    if len(mag_positions) == 0:
        print("No nonzero magnetic moments found in the structure.")
        sys.exit(1)

    _dbg_atoms(verbose, "Magnetic atoms from mCIF (crystal-axis coords, |m|>0.01):",
               mag_positions, magmoms)

    # ── 3. Transform to parent cell ───────────────────────────────────────────
    parent_positions, parent_magmoms, wrap_offsets = mag_rep.map_atoms_to_parent_cell(
        mag_positions, magmoms, child_M, child_t, parent_M, parent_t
    )

    _dbg_atoms(verbose, "After map_atoms_to_parent_cell:", parent_positions, parent_magmoms)

    # ── 4. Deduplicate + reduce to primitive cell ─────────────────────────────
    n_before_dedup = len(parent_positions)
    parent_positions, parent_magmoms, atom_labels = _deduplicate_positions(
        parent_positions, parent_magmoms, offsets=wrap_offsets, labels=atom_labels)
    _dbg(verbose, f"Deduplication: {n_before_dedup} -> {len(parent_positions)} atoms")

    n_before_prim = len(parent_positions)
    parent_positions, parent_magmoms, atom_labels = _select_primitive_atoms(
        parent_positions, it_number, parent_magmoms, labels=atom_labels)
    _dbg(verbose, f"Primitive selection: {n_before_prim} -> {len(parent_positions)} atoms")

    _dbg_atoms(verbose, "Atoms used for chi_mag (parent cell, conventional coords):",
               parent_positions, parent_magmoms)

    if len(parent_positions) == 0:
        print("No magnetic positions found in primitive cell after transformation.")
        sys.exit(1)

    # ── 5. Get irreps from spgrep ─────────────────────────────────────────────
    irreps, rotations, translations, mapping_little_group = irrep_decompose.get_little_group_irreps(
        it_number, kpoint
    )

    _dbg(verbose, f"spgrep: {len(rotations)} total SG ops, "
                  f"|G_k| = {len(mapping_little_group)}, "
                  f"{len(irreps)} irreps")

    if len(irreps) == 0:
        print("Spgrep returned no irreps.")
        sys.exit(1)

    # ── 6. Build D(g) matrices ────────────────────────────────────────────────
    D_matrices_lg = mag_rep.build_mag_rep_matrices(
        rotations, translations, kpoint, parent_positions, mapping_little_group
    )

    # ── 7. Compute magnetic representation characters ─────────────────────────
    chi_mag = mag_rep.compute_characters(rotations, translations, kpoint, parent_positions)

    if verbose:
        chi_lg = chi_mag[mapping_little_group]
        _dbg(verbose, f"chi_mag summary: "
                      f"max|Re| = {np.max(np.abs(chi_lg.real)):.4f}, "
                      f"max|Im| = {np.max(np.abs(chi_lg.imag)):.4f}, "
                      f"chi(E) = {chi_lg[0].real:.4f}  "
                      f"[expected = {3 * len(parent_positions):.0f} for identity]")

    # ── 8. Compute permutation representation data ────────────────────────────
    perm_data = mag_rep.compute_permutation_rep(
        rotations, translations, kpoint, parent_positions, mapping_little_group
    )

    # ── 9. Decompose ──────────────────────────────────────────────────────────
    n_mu_array   = irrep_decompose.decompose(irreps, chi_mag, mapping_little_group,
                                              translations=translations, kpoint=kpoint)
    active_irreps = irrep_decompose.find_active_irrep(n_mu_array)

    _dbg(verbose, f"Decomposition: {len(active_irreps)} active irreps "
                  f"(n > 0.001) out of {len(irreps)}")

    # ── 10. Parity suffixes, projection operators, Bilbao labels ─────────────
    parities  = irrep_decompose.compute_parity_suffixes(irreps, rotations, mapping_little_group)
    proj_ops  = irrep_decompose.compute_projection_operators(
        irreps, D_matrices_lg, mapping_little_group)

    bilbao_labels = bilbao_match.match_irreps(
        irreps, rotations, translations, mapping_little_group, it_number, kpoint)
    if bilbao_labels is None:
        bilbao_labels = irrep_label.bilbao_ordered_labels(kpoint, it_number, irreps, parities)
        _dbg(verbose, "Bilbao REPRES unavailable — using dimension-sorted labels")
    else:
        _dbg(verbose, f"Bilbao labels matched via REPRES: {bilbao_labels}")

    # ── 11. Identify the physically active irrep ──────────────────────────────
    moment_vector = parent_magmoms.flatten()
    _dbg(verbose, f"Moment vector M  shape={moment_vector.shape}  "
                  f"||M|| = {np.linalg.norm(moment_vector):.4f}  "
                  f"values = {np.round(moment_vector, 4).tolist()}")

    identified = irrep_decompose.identify_active_irrep(active_irreps, proj_ops, moment_vector)

    # ── 12. Compute symmetry-adapted basis vectors (magnetic atoms) ───────────
    all_basis = irrep_decompose.compute_basis_vectors(proj_ops, len(parent_positions))

    # ── 13. Single-op reps + Γ_perm[mag] (always computed) ───────────────────
    chi_axial_sg = np.array([
        np.linalg.det(rotations[i].astype(float)) * np.trace(rotations[i].astype(float))
        for i in range(len(rotations))])
    chi_polar_sg = np.array([
        np.trace(rotations[i].astype(float)) for i in range(len(rotations))])
    chi_perm_mag_sg = mag_rep.compute_perm_characters_all(
        rotations, translations, kpoint, parent_positions)

    n_mu_perm_mag = irrep_decompose.decompose(
        irreps, chi_perm_mag_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)
    n_mu_axial = irrep_decompose.decompose(
        irreps, chi_axial_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)
    n_mu_polar = irrep_decompose.decompose(
        irreps, chi_polar_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)

    # ── 14. Displacive pass — all atoms (optional) ────────────────────────────────
    n_mu_mech = None
    n_mu_perm_all = None
    all_par_pos = None
    all_labels = None
    N_all = 0
    all_basis_mech = {}
    active_mech = []
    identified_mech = []
    dim_check_mech = 0.0
    wyckoff_mech_decomps = []

    if displacive_pass:
        all_positions_raw = np.array([site.frac_coords for site in structure])
        all_labels_raw    = [site.species_string for site in structure]
        zero_magmoms_raw  = np.zeros((len(all_positions_raw), 3))

        all_par_pos, _, all_wrap_off = mag_rep.map_atoms_to_parent_cell(
            all_positions_raw, zero_magmoms_raw, child_M, child_t, parent_M, parent_t)
        all_par_pos, all_labels = _deduplicate_positions(
            all_par_pos, offsets=all_wrap_off, labels=all_labels_raw)
        all_par_pos, all_labels = _select_primitive_atoms(
            all_par_pos, it_number, labels=all_labels)
        N_all = len(all_par_pos)

        D_displacive_lg = mag_rep.build_displacive_rep_matrices(
            rotations, translations, kpoint, all_par_pos, mapping_little_group)
        chi_phon = mag_rep.compute_displacive_characters(
            rotations, translations, kpoint, all_par_pos)

        perm_data_all_raw = mag_rep.compute_permutation_rep(
            rotations, translations, kpoint, all_par_pos, mapping_little_group)
        atom_mappings_all, chi_perm_all_lg, _ = perm_data_all_raw
        chi_axial_phon_all = np.array([
            np.trace(rotations[idx].astype(float)) for idx in mapping_little_group])

        n_mu_mech = irrep_decompose.decompose(
            irreps, chi_phon, mapping_little_group,
            translations=translations, kpoint=kpoint)
        active_mech = irrep_decompose.find_active_irrep(n_mu_mech)
        dim_check_mech = sum(n * irreps[idx][0].shape[0] for idx, n in active_mech)

        proj_ops_all   = irrep_decompose.compute_projection_operators(
            irreps, D_displacive_lg, mapping_little_group)
        all_basis_mech = irrep_decompose.compute_basis_vectors(proj_ops_all, N_all)

        def _sort_mech(item):
            idx, n = item
            return (-round(n), irreps[idx][0].shape[0])
        identified_mech = [(idx, n, 1.0, None)
                           for idx, n in sorted(active_mech, key=_sort_mech)]

        # ── 14. Γ_perm[all] decomposition (unique to displacive pass) ────────────
        chi_perm_all_sg = mag_rep.compute_perm_characters_all(
            rotations, translations, kpoint, all_par_pos)
        n_mu_perm_all = irrep_decompose.decompose(
            irreps, chi_perm_all_sg, mapping_little_group,
            translations=translations, kpoint=kpoint)

        # Per-Wyckoff displacive decompositions
        for group_name, indices in _get_wyckoff_groups(it_number, all_par_pos, all_labels):
            chi_wyck = mag_rep.compute_displacive_characters(
                rotations, translations, kpoint, all_par_pos[indices])
            n_mu_wyck = irrep_decompose.decompose(
                irreps, chi_wyck, mapping_little_group,
                translations=translations, kpoint=kpoint)
            wyckoff_mech_decomps.append((group_name, n_mu_wyck))

    n_mu_mag = n_mu_array   # alias for clarity in _print_decomposition_extended

    # ── 15. Print full Bertaut-style report ───────────────────────────────────
    _cm = _tee_stdout(output_file) if output_file else contextlib.nullcontext()
    with _cm:
        print(f"=== Magnetic Irrep Analysis:  {mcif_path} ===\n")

        _print_sg_info(it_number)
        _print_propagation_and_lg(kpoint, mapping_little_group, rotations, translations,
                                   it_number=it_number)
        _print_wyckoff_and_permutation(it_number, parent_positions, atom_labels,
                                       perm_data, kpoint,
                                       mapping_little_group, rotations, translations)
        _print_representation_characters(mapping_little_group, rotations, translations,
                                         perm_data, chi_mag, verbose=verbose)
        _print_character_table(irreps, parities, rotations, translations,
                               mapping_little_group, kpoint, it_number,
                               active_irreps=active_irreps,
                               bilbao_labels=bilbao_labels)
        _print_decomposition_extended(
            n_mu_mag, n_mu_perm_mag,
            n_mu_mech, n_mu_perm_all,
            n_mu_axial, n_mu_polar,
            irreps, bilbao_labels, identified, identified_mech,
            kpoint, it_number, parities,
            wyckoff_mech=wyckoff_mech_decomps if displacive_pass else None)
        _print_basis_vectors(active_irreps, all_basis, atom_labels, parent_positions,
                             identified, bilbao_labels, kpoint, it_number, parities, irreps)
        if identified:
            best_idx, n_best, eta_best, _ = identified[0]
            _format_sk_constraints(best_idx, n_best, eta_best, all_basis, atom_labels,
                                   parent_positions, bilbao_labels, kpoint, it_number,
                                   parities, irreps, mode='magnetic')
        if displacive_pass and active_mech:
            _print_basis_vectors(active_mech, all_basis_mech, all_labels, all_par_pos,
                                 identified_mech, bilbao_labels, kpoint, it_number,
                                 parities, irreps, mode='displacive')
        _print_moment_consistency(active_irreps, all_basis, parent_magmoms,
                                   atom_labels, parent_positions, identified,
                                   bilbao_labels, kpoint, it_number, parities, irreps)
        _print_validation(fields, identified, bilbao_labels, kpoint, it_number,
                          parities, irreps)
        if displacive_pass:
            print(f"  Σ n_μ·d_μ (mech) = {dim_check_mech:.3f}  "
                  f"(expected 3×N_all = {3*N_all})")
            print()


def _get_it_number_from_structure(structure) -> int:
    """Use spglib to get the IT (international table) number from a pymatgen Structure."""
    from pymatgen.core import Element
    cell = (
        structure.lattice.matrix,
        structure.frac_coords,
        [Element(s.specie.symbol).Z for s in structure],
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    if dataset is None:
        raise RuntimeError("spglib could not determine the space group of the structure.")
    return int(dataset.number if hasattr(dataset, 'number') else dataset['number'])


def run_displacive_analysis(path: str, kvector_str: str = None, verbose: bool = False,
                        output_file: str = None,
                        distort_amplitude: float = None,
                        keep_magnetic: bool = False,
                        out_dir: str = None):
    """Run displacive/mechanical representation analysis on *path* (mCIF or CIF).

    For mCIF input: k and IT# are read from the mCIF metadata.  ALL atoms are
    used (no magnetic-moment filter).  Transforms from the mCIF are applied.
    For plain CIF input: IT# is determined from spglib; k comes from *kvector_str*
    (default '0,0,0' → Γ-point).

    If *distort_amplitude* is set, distorted CIF files are generated per
    (active irrep, Wyckoff site, basis vector) after the analysis.
    """
    from pymatgen.core import Structure

    # ── 1. Parse input file ───────────────────────────────────────────────────
    try:
        fields = parse_mcif.parse_mcif_fields(path)
        is_mcif = True
    except Exception:
        fields = {}
        is_mcif = False

    if is_mcif:
        kpoint    = parse_mcif.parse_kvector(fields['kvector_str'])
        it_number = fields['it_number']
        child_M, child_t = parse_mcif.parse_transform(fields['child_transform_str'])
        if 'parent_transform_str' in fields:
            parent_M, parent_t = parse_mcif.parse_transform(fields['parent_transform_str'])
        else:
            parent_M, parent_t = np.eye(3), np.zeros(3)
        structure = parse_mcif.get_magnetic_structure(path)
    else:
        # Plain CIF
        structure = Structure.from_file(path)
        it_number = _get_it_number_from_structure(structure)
        kv_str    = kvector_str if kvector_str is not None else "0,0,0"
        kpoint    = parse_mcif.parse_kvector(kv_str)
        child_M, child_t   = np.eye(3), np.zeros(3)
        parent_M, parent_t = np.eye(3), np.zeros(3)

    _dbg(verbose, f"Displacive mode — IT number: {it_number}")
    _dbg(verbose, f"k-vector: {kpoint}")

    # ── 2. Collect ALL atoms (no magmom filter) ───────────────────────────────
    all_positions = []
    atom_labels   = []
    all_magmoms   = []   # real moments (if mCIF) or zeros (CIF)

    for site in structure:
        all_positions.append(site.frac_coords)
        atom_labels.append(site.species_string)
        if is_mcif and 'magmom' in site.properties and site.properties['magmom'] is not None:
            m = site.properties['magmom']
            all_magmoms.append(list(m) if hasattr(m, '__iter__') else [0.0, 0.0, float(m)])
        else:
            all_magmoms.append([0.0, 0.0, 0.0])

    all_positions = np.array(all_positions)
    all_magmoms   = np.array(all_magmoms)

    if len(all_positions) == 0:
        print("No atoms found in the structure.")
        import sys; sys.exit(1)

    _dbg(verbose, f"Total atoms from structure: {len(all_positions)}")

    # ── 3. Transform to parent cell ───────────────────────────────────────────
    parent_positions, parent_magmoms, wrap_offsets = mag_rep.map_atoms_to_parent_cell(
        all_positions, all_magmoms, child_M, child_t, parent_M, parent_t
    )

    # ── 4. Deduplicate + reduce to primitive cell ─────────────────────────────
    n_before_dedup = len(parent_positions)
    parent_positions, parent_magmoms, atom_labels = _deduplicate_positions(
        parent_positions, magmoms=parent_magmoms, offsets=wrap_offsets, labels=atom_labels)
    _dbg(verbose, f"Deduplication: {n_before_dedup} → {len(parent_positions)} atoms")

    n_before_prim = len(parent_positions)
    parent_positions, parent_magmoms, atom_labels = _select_primitive_atoms(
        parent_positions, it_number, magmoms=parent_magmoms, labels=atom_labels)
    _dbg(verbose, f"Primitive selection: {n_before_prim} → {len(parent_positions)} atoms")

    if len(parent_positions) == 0:
        print("No atoms found in primitive cell after transformation.")
        import sys; sys.exit(1)

    N_prim = len(parent_positions)

    # ── 5. Get irreps from spgrep ─────────────────────────────────────────────
    irreps, rotations, translations, mapping_little_group = irrep_decompose.get_little_group_irreps(
        it_number, kpoint
    )

    _dbg(verbose, f"spgrep: {len(rotations)} total SG ops, "
                  f"|G_k| = {len(mapping_little_group)}, "
                  f"{len(irreps)} irreps")

    if len(irreps) == 0:
        print("Spgrep returned no irreps.")
        import sys; sys.exit(1)

    # ── 6. Build D(g) matrices (displacive/polar-vector version) ────────────────
    D_matrices_lg = mag_rep.build_displacive_rep_matrices(
        rotations, translations, kpoint, parent_positions, mapping_little_group
    )

    # ── 7. Compute χ_disp (no det factor) ────────────────────────────────────
    # compute_displacive_characters takes the full rotations/translations arrays
    # and returns chi for ALL ops (indexed by all op index, not lg index).
    chi_disp = mag_rep.compute_displacive_characters(
        rotations, translations, kpoint, parent_positions
    )

    if verbose:
        chi_lg = chi_disp[mapping_little_group]
        _dbg(verbose, f"chi_disp summary: "
                      f"max|Re| = {np.max(np.abs(chi_lg.real)):.4f}, "
                      f"chi(E) = {chi_lg[0].real:.4f}  "
                      f"[expected = {3 * N_prim:.0f} for identity]")

    # ── 8. Permutation representation data ───────────────────────────────────
    # For displacive mode chi_axial = Tr(R). compute_permutation_rep returns
    # chi_axial = det(R)·Tr(R), so we override it for display.
    perm_data_raw = mag_rep.compute_permutation_rep(
        rotations, translations, kpoint, parent_positions, mapping_little_group
    )
    atom_mappings, chi_perm, _chi_axial_mag = perm_data_raw

    # Recompute chi_axial for displacive (Tr(R) only)
    chi_axial_phon = np.array([
        np.trace(rotations[idx].astype(float))
        for idx in mapping_little_group
    ])
    perm_data = (atom_mappings, chi_perm, chi_axial_phon)

    # ── 9. Decompose ──────────────────────────────────────────────────────────
    n_mu_array    = irrep_decompose.decompose(irreps, chi_disp, mapping_little_group,
                                               translations=translations, kpoint=kpoint)
    active_irreps = irrep_decompose.find_active_irrep(n_mu_array)

    _dbg(verbose, f"Decomposition: {len(active_irreps)} active irreps out of {len(irreps)}")

    # Cross-check: Σ n_μ d_μ should equal 3*N_prim
    dim_check = sum(n * irreps[idx][0].shape[0] for idx, n in active_irreps)
    _dbg(verbose, f"Σ n_μ·d_μ = {dim_check:.3f}  (expected 3×{N_prim} = {3*N_prim})")

    # ── 10. Parity, projection operators, Bilbao labels ──────────────────────
    parities  = irrep_decompose.compute_parity_suffixes(irreps, rotations, mapping_little_group)
    proj_ops  = irrep_decompose.compute_projection_operators(
        irreps, D_matrices_lg, mapping_little_group)

    bilbao_labels = bilbao_match.match_irreps(
        irreps, rotations, translations, mapping_little_group, it_number, kpoint)
    if bilbao_labels is None:
        bilbao_labels = irrep_label.bilbao_ordered_labels(
            kpoint, it_number, irreps, parities, magnetic=False)
        _dbg(verbose, "Bilbao REPRES unavailable — using dimension-sorted labels")

    # ── 11. Identify leading irrep (no observed displacement vector for displacive) ─
    # Sort by n_μ descending (ties broken by dimension ascending).
    # η is set to 1.0 for all active irreps (no projection metric without obs. displacements).
    if active_irreps:
        def _sort_key(item):
            idx, n = item
            d = irreps[idx][0].shape[0]
            return (-round(n), d)
        identified = [(idx, n, 1.0, None)
                      for idx, n in sorted(active_irreps, key=_sort_key)]
    else:
        identified = []

    # ── 12. Compute symmetry-adapted basis vectors ────────────────────────────
    all_basis = irrep_decompose.compute_basis_vectors(proj_ops, N_prim)

    # ── 12b. Γ_perm, Γ_axial, Γ_polar decompositions (displacive mode) ───────────
    chi_perm_all_sg = mag_rep.compute_perm_characters_all(
        rotations, translations, kpoint, parent_positions)
    chi_axial_sg = np.array([
        np.linalg.det(rotations[i].astype(float)) * np.trace(rotations[i].astype(float))
        for i in range(len(rotations))])
    chi_polar_sg = np.array([
        np.trace(rotations[i].astype(float)) for i in range(len(rotations))])

    n_mu_perm_all = irrep_decompose.decompose(
        irreps, chi_perm_all_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)
    n_mu_axial = irrep_decompose.decompose(
        irreps, chi_axial_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)
    n_mu_polar = irrep_decompose.decompose(
        irreps, chi_polar_sg, mapping_little_group,
        translations=translations, kpoint=kpoint)

    # Per-Wyckoff decompositions (Γ_perm and Γ_mech)
    wyckoff_groups = _get_wyckoff_groups(it_number, parent_positions, atom_labels)
    wyckoff_perm_decomps = []
    wyckoff_mech_decomps = []
    for group_name, indices in wyckoff_groups:
        chi_perm_wyck = mag_rep.compute_perm_characters_all(
            rotations, translations, kpoint, parent_positions[indices])
        n_mu_perm_wyck = irrep_decompose.decompose(
            irreps, chi_perm_wyck, mapping_little_group,
            translations=translations, kpoint=kpoint)
        wyckoff_perm_decomps.append((group_name, n_mu_perm_wyck))

        chi_wyck = mag_rep.compute_displacive_characters(
            rotations, translations, kpoint, parent_positions[indices])
        n_mu_wyck = irrep_decompose.decompose(
            irreps, chi_wyck, mapping_little_group,
            translations=translations, kpoint=kpoint)
        wyckoff_mech_decomps.append((group_name, n_mu_wyck))

    n_mu_mech = n_mu_array   # alias for _print_decomposition_extended

    # ── 13. Print full Bertaut-style report ───────────────────────────────────
    _cm = _tee_stdout(output_file) if output_file else contextlib.nullcontext()
    with _cm:
        print(f"=== Displacive Mode Analysis:  {path} ===\n")

        _print_sg_info(it_number)
        _print_propagation_and_lg(kpoint, mapping_little_group, rotations, translations,
                                   it_number=it_number)
        _print_wyckoff_and_permutation(it_number, parent_positions, atom_labels,
                                       perm_data, kpoint,
                                       mapping_little_group, rotations, translations,
                                       mode='displacive')
        _print_representation_characters(mapping_little_group, rotations, translations,
                                         perm_data, chi_disp, verbose=verbose,
                                         mode='displacive')
        _print_character_table(irreps, parities, rotations, translations,
                               mapping_little_group, kpoint, it_number,
                               bilbao_labels=bilbao_labels,
                               mode='displacive')
        _print_decomposition_extended(
            None, None,
            n_mu_mech, n_mu_perm_all,
            n_mu_axial, n_mu_polar,
            irreps, bilbao_labels, [], identified,
            kpoint, it_number, parities,
            wyckoff_mech=wyckoff_mech_decomps,
            wyckoff_perm=wyckoff_perm_decomps)
        _print_basis_vectors(active_irreps, all_basis, atom_labels, parent_positions,
                             identified, bilbao_labels, kpoint, it_number, parities,
                             irreps, mode='displacive', wyckoff_groups=wyckoff_groups)
        # Section (9b): Displacement constraints Sk(j) for each active displacive irrep
        if active_irreps:
            print("(9b) DISPLACEMENT CONSTRAINTS  Sk(j)  [displacive mode]\n")
            sorted_active = sorted(active_irreps,
                                   key=lambda x: bilbao_labels.get(x[0], f"z{x[0]}"))
            for irrep_idx, n_mu in sorted_active:
                _format_sk_constraints(irrep_idx, n_mu, 1.0, all_basis, atom_labels,
                                       parent_positions, bilbao_labels, kpoint, it_number,
                                       parities, irreps, mode='displacive')
        # Section (10) moment-consistency is skipped for displacive mode
        print(f"  Σ n_μ·d_μ = {dim_check:.3f}  (expected 3×N_prim = {3*N_prim})")
        print()

    # ── Distorted CIF generation (optional) ───────────────────────────────────
    if distort_amplitude is not None and active_irreps:
        from magirrep.distort import generate_distorted_cifs
        do_magnetic = keep_magnetic and is_mcif
        moms_for_distort = parent_magmoms if do_magnetic else None
        generated = generate_distorted_cifs(
            path, parent_positions, atom_labels, it_number, kpoint,
            child_M, structure, all_basis, active_irreps, bilbao_labels,
            wyckoff_groups, amplitude=distort_amplitude,
            keep_magnetic=do_magnetic,
            parent_magmoms=moms_for_distort,
            out_dir=out_dir,
        )
        print(f"\n  Generated {len(generated)} distorted CIF file(s):")
        for f in generated:
            print(f"    {f}")
        print()
