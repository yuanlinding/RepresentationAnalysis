"""Orchestration pipeline: parse -> transform -> irreps -> decompose -> label."""

import sys
from collections import Counter

import numpy as np
import spglib

from magirrep import parse_mcif, mag_rep, irrep_decompose, irrep_label, bilbao_match
from magirrep.little_group import build_reference_crystal, get_hall_number


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

    Returns
    -------
    Tuple of whichever of (positions, magmoms, labels) were non-None, or just
    positions if all extras are None.
    """
    lattice_conv, _, _ = build_reference_crystal(it_number)
    cell_ref = (lattice_conv, np.array([[0.0, 0.0, 0.0]]), np.array([1]))
    prim_cell = spglib.find_primitive(cell_ref, symprec=1e-3)

    if prim_cell is None:
        if magmoms is not None and labels is not None:
            return conv_positions, magmoms, labels
        elif magmoms is not None:
            return conv_positions, magmoms
        elif labels is not None:
            return conv_positions, labels
        return conv_positions

    prim_lattice = prim_cell[0]
    P = prim_lattice @ np.linalg.inv(lattice_conv)
    P_inv_T = np.linalg.inv(P.T)

    seen_prim: list = []
    kept_idx: list = []
    result: list = []
    for i, r_conv in enumerate(conv_positions):
        r_prim = (P_inv_T @ r_conv) % 1.0
        if not any(np.allclose(r_prim, s, atol=tol) for s in seen_prim):
            seen_prim.append(r_prim)
            result.append(r_conv)
            kept_idx.append(i)

    result_pos = np.array(result) if result else np.zeros((0, 3))
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


def _get_wyckoff_sites(it_number: int, parent_positions: np.ndarray) -> list:
    """Return Wyckoff letter for each atom in *parent_positions*."""
    ref_lattice, _, _ = build_reference_crystal(it_number)
    numbers = np.ones(len(parent_positions), dtype=int)
    dataset = spglib.get_symmetry_dataset(
        (ref_lattice, parent_positions, numbers), symprec=1e-3
    )
    if dataset is None:
        return ['?' for _ in parent_positions]
    wyckoffs = (dataset.wyckoffs if hasattr(dataset, 'wyckoffs')
                else dataset['wyckoffs'])
    return [str(w) for w in wyckoffs]


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

    # Column width for the {R|t} Seitz symbol
    seitz_w = max(20, max(len(_seitz(rotations[idx], translations[idx]))
                          for _, idx in dedup) + 1)
    print(f"  {'#':>3}  {'{{R | t}}':<{seitz_w}}")
    print("  " + "-" * (6 + seitz_w))

    for i, (_i_lg, idx) in enumerate(dedup):
        R = rotations[idx]
        t = translations[idx]
        print(f"  {i+1:>3}  {_seitz(R, t):<{seitz_w}}")
    print()


def _print_wyckoff_and_permutation(it_number, parent_positions, atom_labels,
                                    perm_data, kpoint,
                                    mapping_little_group, rotations, translations):
    """Sections (4)+(5): Wyckoff sites and permutation of the Wyckoff orbit."""
    atom_mappings, _chi_perm, _chi_axial = perm_data
    N = len(parent_positions)

    wyckoffs = _get_wyckoff_sites(it_number, parent_positions)

    print("(4) MAGNETIC ATOMS — WYCKOFF SITES")
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
                                      perm_data, chi_mag, verbose=False):
    """Section (6): Per-operation χ_perm, χ_axial, χ_mag (one row per unique {R|t})."""
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

    print("(6) REPRESENTATION ANALYSIS")
    print("  χ_perm(g)  = Σ_{fixed atoms} exp(−2πi k·L)")
    print("  χ_axial(g) = det(R) · Tr(R)")
    print("  χ_mag(g)   = χ_perm × χ_axial")
    print()
    print(f"  {'#':>3}  {'{{R | t}}':<{op_w}}  {'χ_perm':>9}  {'det(R)':>7}  {'Tr(R)':>6}  "
          f"{'χ_axial':>9}  {'χ_mag':>9}")
    print("  " + "-" * (6 + op_w + 62))

    for i, (i_lg, idx) in enumerate(dedup):
        R     = rotations[idx].astype(float)
        det_R = int(round(np.linalg.det(R)))
        tr_R  = int(round(np.trace(R)))
        cp    = chi_perm[i_lg]
        ca    = chi_axial[i_lg]
        cm    = chi_mag[idx]
        op_str = _seitz(rotations[idx], translations[idx])
        print(f"  {i+1:>3}  {op_str:<{op_w}}  {_fmtc(cp):>9}  {det_R:>+7}  {tr_R:>6}  "
              f"  {ca:>+9.4f}  {_fmtc(cm):>9}")
    print()


def _print_character_table(irreps, parities, rotations, translations,
                            mapping_little_group, kpoint, it_number,
                            active_irreps=None, bilbao_labels=None):
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

    active_set = {idx for idx, *_ in (active_irreps or [])}

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


def _print_decomposition(active_irreps, irreps, n_mu_array, bilbao_labels,
                          identified, kpoint, it_number, parities):
    """Section (8): Decomposition of Γ_mag into small representations of G_k."""
    def _lbl(alpha):
        if bilbao_labels and alpha in bilbao_labels:
            return bilbao_labels[alpha]
        d = irreps[alpha][0].shape[0]
        p = parities[alpha] if parities else ''
        return irrep_label.irrep_name(kpoint, it_number, alpha, d, p)

    def _nstr(n):
        return str(int(round(n))) if abs(n - round(n)) < 0.01 else f"{n:.3f}"

    terms = [f"{_nstr(n)}·{_lbl(idx)}" for idx, n in active_irreps]

    print("(8) DECOMPOSITION INTO SMALL REPRESENTATIONS OF G_k")
    if terms:
        print("  Γ_mag  =  " + "  ⊕  ".join(terms))
    else:
        print("  Γ_mag  =  0  (no active small representations found)")
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
                          parities, irreps):
    """Section (9): Symmetry-adapted basis vectors in Bilbao-style table.

    Columns: IR | BV | m1a m1b m1c | m2a m2b m2c | ...
    Values are scaled to the smallest-integer representation (min nonzero = ±1).
    The IR label is printed only on the first row of each irrep.
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

    # Column labels: m1a m1b m1c m2a ...
    axes      = ['a', 'b', 'c']
    comp_hdrs = [f"m{a+1}{ax}" for a in range(N) for ax in axes]
    cw = max(4, max(len(h) for h in comp_hdrs) + 1)   # data column width

    # IR and BV column widths
    ir_w = max(8, max((len(_lbl(idx)) for idx, _ in active_irreps), default=8))
    bv_w = 3

    print("(9) BASIS VECTORS OF ACTIVE SMALL REPRESENTATIONS\n")

    # Header row
    hdr = f"  {'IR':<{ir_w}}  {'BV':<{bv_w}}"
    sep = f"  {'':-<{ir_w}}  {'':-<{bv_w}}"
    for h in comp_hdrs:
        hdr += f"  {h:>{cw}}"
        sep += f"  {'':->{cw}}"
    print(hdr)
    print(sep)

    bv_num = 0
    for idx, n in active_irreps:
        lbl        = _lbl(idx)
        eta        = eta_map.get(idx, 0.0)
        basis_vecs = all_basis[idx]
        is_active  = (idx == best_idx)

        if not basis_vecs:
            row = f"  {lbl:<{ir_w}}  {'—':<{bv_w}}  (no basis vectors found)"
            if is_active:
                row += f"  ← ACTIVE  η={eta:.3f}"
            print(row)
            continue

        first = True
        for v in basis_vecs:
            bv_num  += 1
            v_sc     = _scale_to_integers(np.real(v))
            ir_field = lbl if first else ""
            bv_label = f"ψ{bv_num}"
            row = f"  {ir_field:<{ir_w}}  {bv_label:<{bv_w}}"
            for val in v_sc:
                row += f"  {_fmt_bv_val(val):>{cw}}"
            if first and is_active:
                row += f"  ← ACTIVE  η={eta:.3f}"
            print(row)
            first = False

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


def run_analysis(mcif_path: str, verbose: bool = False):
    """Run the full magnetic irrep analysis pipeline on *mcif_path*."""

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
    mag_positions = []
    magmoms       = []
    atom_labels   = []

    for site in structure:
        raw_m = site.properties.get("magmom")
        if raw_m is None:
            raw_m = getattr(site, "magmom", None)

        if raw_m is not None:
            if hasattr(raw_m, 'global_moment'):
                m_vec = raw_m.global_moment
            elif hasattr(raw_m, 'moment'):
                m_vec = raw_m.moment
            else:
                m_vec = raw_m

            m_vec = np.array(m_vec)
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

    # ── 12. Compute symmetry-adapted basis vectors ────────────────────────────
    all_basis = irrep_decompose.compute_basis_vectors(proj_ops, len(parent_positions))

    # ── 13. Print full Bertaut-style report ───────────────────────────────────
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
    _print_decomposition(active_irreps, irreps, n_mu_array, bilbao_labels,
                         identified, kpoint, it_number, parities)
    _print_basis_vectors(active_irreps, all_basis, atom_labels, parent_positions,
                         identified, bilbao_labels, kpoint, it_number, parities, irreps)
    _print_moment_consistency(active_irreps, all_basis, parent_magmoms,
                               atom_labels, parent_positions, identified,
                               bilbao_labels, kpoint, it_number, parities, irreps)
    _print_validation(fields, identified, bilbao_labels, kpoint, it_number,
                      parities, irreps)
