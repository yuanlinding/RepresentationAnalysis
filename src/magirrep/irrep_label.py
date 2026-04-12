import numpy as np

# Module-level cache: it_number -> dict{rounded_k_conv_tuple -> bilbao_label}
_SEEKPATH_CACHE: dict = {}


def _sp_to_bilbao(label: str) -> str:
    """Map a seekpath k-point label to the Bilbao convention.

    The only substantive change is GAMMA → GM.  Subscripted copies such as
    'X_1', 'L_2' have their underscore stripped so they become 'X1', 'L2'.
    """
    base, *rest = label.split('_')
    bilbao_base = 'GM' if base == 'GAMMA' else base
    return bilbao_base + ''.join(rest)  # e.g. 'X_1' → 'X1', 'L' → 'L'


def _get_seekpath_labels(it_number: int) -> dict:
    """Build (and cache) a mapping from conventional reciprocal k-coords to
    Bilbao labels for the Bravais lattice of space group *it_number*.

    Uses seekpath (Setyawan-Curtarolo convention) with the reference crystal
    from build_reference_crystal().  Returns an empty dict if seekpath is not
    installed or the lookup fails.
    """
    if it_number in _SEEKPATH_CACHE:
        return _SEEKPATH_CACHE[it_number]

    try:
        import seekpath
        import spglib
        from magirrep.little_group import build_reference_crystal
    except ImportError:
        _SEEKPATH_CACHE[it_number] = {}
        return {}

    try:
        lat_conv, pos_conv, nums_conv = build_reference_crystal(it_number)
        A_conv = np.array(lat_conv, dtype=float)

        # Find primitive cell (may differ from conventional for F, I, A, B, C, R)
        prim = spglib.find_primitive((lat_conv, pos_conv, nums_conv), symprec=1e-3)
        A_prim = np.array(prim[0], dtype=float) if prim is not None else A_conv

        # Transformation:  k_prim = k_conv @ inv(A_conv) @ A_prim
        T = np.linalg.inv(A_conv) @ A_prim        # k_conv → k_prim
        T_inv = np.linalg.inv(T)                   # k_prim → k_conv

        # High-symmetry k-points from seekpath (in primitive reciprocal fractional)
        result = seekpath.get_path((A_prim.tolist(), [[0, 0, 0]], [1]))

        label_map = {}
        for sp_label, k_prim in result['point_coords'].items():
            k_conv = np.array(k_prim) @ T_inv
            bilbao = _sp_to_bilbao(sp_label)
            # Store the raw (unwrapped) conventional k-vector.
            # Do NOT also store the wrapped version: some zone-boundary
            # points (e.g. FCC X, BCC H) wrap to (0,0,0), which would
            # silently overwrite GAMMA.  The lookup side handles wrapping.
            key = tuple(np.round(k_conv, 6))
            label_map[key] = bilbao

        _SEEKPATH_CACHE[it_number] = label_map
        return label_map

    except Exception:
        _SEEKPATH_CACHE[it_number] = {}
        return {}


def _kpoint_label_heuristic(kpoint) -> str:
    """Fallback heuristic k-point label (works for common cases regardless of SG)."""
    k = np.abs(np.array(kpoint))
    if np.allclose(k, [0, 0, 0], atol=1e-4):
        return 'GM'
    if np.allclose(k, [0.5, 0.5, 0.5], atol=1e-4):
        return 'L'
    if (np.allclose(k, [0.5, 0, 0], atol=1e-4) or
            np.allclose(k, [0, 0.5, 0], atol=1e-4) or
            np.allclose(k, [0, 0, 0.5], atol=1e-4)):
        return 'X'
    if np.allclose(k, [0.5, 0.5, 0], atol=1e-4):
        return 'M'
    return f"[{kpoint[0]:.3f}_{kpoint[1]:.3f}_{kpoint[2]:.3f}]"


def kpoint_label(kpoint, it_number: int = None) -> str:
    """Map a k-point to its high-symmetry Bilbao label.

    Parameters
    ----------
    kpoint : array-like, shape (3,)
        Propagation vector in conventional reciprocal fractional coordinates.
    it_number : int, optional
        IT space-group number.  When provided, seekpath is used (Setyawan-
        Curtarolo convention) via the reference crystal for this SG.
        'GAMMA' is remapped to 'GM'; subscripted copies like 'X_1' become
        'X1'.  Falls back to the built-in heuristic when seekpath is not
        installed or the k-point is not a named high-symmetry point.

    Notes
    -----
    seekpath follows the Setyawan-Curtarolo (2010) convention, which may
    differ from the CDML/Bilbao convention for some crystal systems.  For
    the common Γ and zone-boundary points of cubic/tetragonal/hexagonal
    structures the two conventions agree.
    """
    k = np.array(kpoint, dtype=float)

    if it_number is not None:
        label_map = _get_seekpath_labels(it_number)
        if label_map:
            key = tuple(np.round(k, 6))
            if key in label_map:
                return label_map[key]
            key_w = tuple(np.round(k % 1.0, 6))
            if key_w in label_map:
                return label_map[key_w]

    return _kpoint_label_heuristic(kpoint)


def irrep_name(kpoint, sg_number: int, irrep_idx: int, irrep_dim: int,
               parity: str = '', magnetic: bool = True) -> str:
    """Assemble a Bilbao-style label, e.g. mGM5- (magnetic) or GM5- (displacive).

    Parameters
    ----------
    parity : str
        '+', '-', or '' (no suffix when inversion is not in the little group).
    magnetic : bool
        If True (default), prepend 'm' to indicate a magnetic small representation.
        If False (displacive mode), omit the 'm' prefix.
    """
    k_label = kpoint_label(kpoint, it_number=sg_number)
    prefix  = 'm' if magnetic else ''
    label   = f"{prefix}{k_label}{irrep_idx + 1}{parity}"
    return label


def bilbao_ordered_labels(kpoint, sg_number: int, irreps, parities,
                           magnetic: bool = True) -> dict:
    """Assign Bilbao-convention labels by sorting within each parity group by dimension.

    Bilbao numbers irreps within each parity group ('+', '-') starting from 1,
    ordered by increasing dimension.  This matches Bilbao's convention for common
    little groups (D4h at Γ, D3d at L, etc.) and correctly identifies the active
    irrep in the CuMnAs and NiO validation cases.

    Parameters
    ----------
    magnetic : bool
        If True (default), prefix labels with 'm' (magnetic small representations).
        If False (displacive mode), omit the 'm' prefix.

    Returns
    -------
    dict {spgrep_idx: [m]k_label + number + parity}
    """
    from collections import defaultdict

    k_label = kpoint_label(kpoint, it_number=sg_number)
    prefix  = 'm' if magnetic else ''
    groups  = defaultdict(list)   # parity → [(dim, alpha)]

    for alpha, (irr, p) in enumerate(zip(irreps, parities)):
        dim = irr[0].shape[0]
        groups[p].append((dim, alpha))

    labels = {}
    for p in ['+', '-', '']:
        if p not in groups:
            continue
        for num, (dim, alpha) in enumerate(sorted(groups[p]), start=1):
            labels[alpha] = f"{prefix}{k_label}{num}{p}"

    return labels
