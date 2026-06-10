import numpy as np

_NO_CENTERING = (np.zeros(3),)


def _match_with_centering(diff, centerings, tol=1e-4):
    """Match an atom-mapping difference vector modulo the PRIMITIVE lattice.

    *diff* = R@r_src + t − r_dst in conventional fractional coordinates.
    For centered Bravais lattices the primitive lattice contains non-integer
    conventional vectors (the centering translations), so after reduction to
    one atom per centering orbit a symmetry op may map an atom onto the
    removed centering copy of another.  Matching only modulo Z³ then fails
    and the character silently loses that contribution — which breaks the
    small-representation phase relation χ({R|t+t_c}) = e^(−2πik·t_c) χ({R|t})
    and produces fractional multiplicities in decompose().

    Returns the true primitive-lattice vector L (possibly half-integer in
    conventional coordinates) or None if no match.
    """
    for ct in centerings:
        d2 = diff - ct
        if np.allclose(d2 - np.round(d2), 0, atol=tol):
            return np.round(d2) + ct
    return None


def map_atoms_to_parent_cell(positions, magmoms, child_transform_M, child_transform_t, parent_transform_M, parent_transform_t):
    """
    Transforms magnetic-cell fractional coordinates and magnetic moments to the 
    standard parent setting fractional coordinates.
    
    positions: Nx3 array
    magmoms: Nx3 array
    child_transform_M: 3x3 array (parent to child basis transform)
    child_transform_t: 3x1 array (parent to child origin shift)
    parent_transform_M: 3x3 array (standard parent to parent basis transform)
    """
    parent_positions = []
    parent_magmoms = []
    wrap_offsets = []

    # Usually: r_child = P_child^-1 r_parent - p_child
    # r_parent = P_child (r_child + p_child)
    # P_child is child_transform_M.
    # Wait, pymatgen's SymmOp parsed from '2a,2b,2c;0,0,0' gives the forward matrix M where a'=Ma.
    # The transformation of fractional coordinates r' = M^-T r ? No.
    # If basis vectors are row vectors A: A_child = M @ A_parent
    # Then fractional coordinates x column vector: A_parent.T @ x_parent = A_child.T @ x_child
    # A_parent.T @ x_parent = (M @ A_parent).T @ x_child = A_parent.T @ M.T @ x_child
    # So x_parent = M.T @ x_child.
    # Wait, the transformations in CIF child_transform_Pp_abc are usually P, p where:
    # A_child = P @ A_parent
    # The Bilbao convention uses (P,p) as:
    # r_parent = P @ r_child + p
    # So we just use M @ r_child + t.
    # We apply this first for the child transform, then for the parent transform.

    for r, m in zip(positions, magmoms):
        # 1. From child supercell to parent cell.
        # The MAGNDATA child_transform_Pp_abc string 'c,a,b-c;1/4,0,0' describes
        # child basis vectors in terms of parent basis (rows of P).  In fractional
        # coordinates the atom transform is r_parent = P^T @ r_child + p, where
        # P = child_transform_M as returned by parse_transform.
        r_p = child_transform_M.T @ r + child_transform_t

        # 2. From parent cell to standard parent setting (raw, before wrapping)
        r_std_raw = parent_transform_M.T @ r_p + parent_transform_t

        # Wrapping to [0, 1); track integer offset for canonical-representative selection
        r_std = r_std_raw % 1.0
        L_wrap = np.round(r_std_raw - r_std).astype(int)

        # Moment vectors in fractional components transform exactly like
        # coordinates under a pure basis change (m_cart = A.T @ m_frac is
        # invariant, A_child = M @ A_parent  ⇒  m_frac_parent = M.T @ m_frac_child).
        # No det factor: that applies to physical symmetry operations on axial
        # vectors, not to relabelling the basis.
        m_p = child_transform_M.T @ m
        m_std = parent_transform_M.T @ m_p

        parent_positions.append(r_std)
        parent_magmoms.append(m_std)
        wrap_offsets.append(L_wrap)

    return np.array(parent_positions), np.array(parent_magmoms), np.array(wrap_offsets)

def build_mag_rep_matrices(rotations, translations, kpoint, parent_positions,
                            mapping_little_group, tol=1e-4, centerings=None):
    """
    Build the full 3N×3N D(g) matrices of the magnetic representation for each
    little-group operation.

    For operation g = {R | t}, atom src maps to atom dst with lattice shift L:
        R @ r_src + t ≈ r_dst + L   (r_dst in parent_positions, L in Z³)
    Block (dst, src):
        D(g)[3*dst:3*dst+3, 3*src:3*src+3] = det(R) * R * exp(-2πi k·L)

    Tr(D(g)) == chi_mag(g) for each little-group op (self-consistency check).

    Parameters
    ----------
    rotations : np.ndarray, shape (N_ops, 3, 3)
    translations : np.ndarray, shape (N_ops, 3)
    kpoint : array-like, shape (3,)
    parent_positions : np.ndarray, shape (N_prim, 3)
        Primitive-cell magnetic atom positions in fractional coordinates.
    mapping_little_group : array-like
        Indices into rotations/translations for the little-group ops.

    Returns
    -------
    D_matrices : list[np.ndarray]
        One 3N×3N complex matrix per little-group operation, in the same order
        as mapping_little_group.
    """
    N = len(parent_positions)
    kpoint = np.asarray(kpoint, dtype=float)
    if centerings is None:
        centerings = _NO_CENTERING
    D_matrices = []

    for idx in mapping_little_group:
        R = rotations[idx].astype(float)
        t = translations[idx]
        det_R = np.linalg.det(R)
        D = np.zeros((3 * N, 3 * N), dtype=complex)

        for src, r_src in enumerate(parent_positions):
            r_transformed = R @ r_src + t
            for dst, r_dst in enumerate(parent_positions):
                L = _match_with_centering(r_transformed - r_dst, centerings, tol)
                if L is not None:
                    phase = np.exp(-2j * np.pi * np.dot(kpoint, L))
                    D[3*dst:3*dst+3, 3*src:3*src+3] = det_R * R * phase
                    break  # each src maps to exactly one dst

        D_matrices.append(D)

    return D_matrices


def compute_permutation_rep(rotations, translations, kpoint,
                             parent_positions, mapping_little_group, tol=1e-4,
                             centerings=None):
    """
    For each little-group operation compute the permutation representation data.

    Parameters
    ----------
    rotations : np.ndarray, shape (N_ops, 3, 3)
    translations : np.ndarray, shape (N_ops, 3)
    kpoint : array-like, shape (3,)
    parent_positions : np.ndarray, shape (N, 3)
    mapping_little_group : array-like

    Returns
    -------
    atom_mappings : list[list[(int, np.ndarray)]]
        atom_mappings[i][src] = (dst_idx, L_vec) where R@r_src+t ≈ r_dst+L_vec.
        dst_idx is 0-based; L_vec is integer array of shape (3,).
    chi_perm : np.ndarray, shape (|G_k|,), complex
        chi_perm[i] = Σ_{j: R r_j+t ≡ r_j (mod Z³)} exp(−2πi k·L)
    chi_axial : np.ndarray, shape (|G_k|,), float
        chi_axial[i] = det(R) · Tr(R)
    """
    N = len(parent_positions)
    kpoint = np.asarray(kpoint, dtype=float)
    if centerings is None:
        centerings = _NO_CENTERING

    atom_mappings = []
    chi_perm  = np.zeros(len(mapping_little_group), dtype=complex)
    chi_axial = np.zeros(len(mapping_little_group), dtype=float)

    for i_lg, idx in enumerate(mapping_little_group):
        R = rotations[idx].astype(float)
        t = translations[idx]
        det_R = np.linalg.det(R)
        tr_R  = np.trace(R)
        chi_axial[i_lg] = det_R * tr_R

        mappings_this_op = []
        chi_p = 0.0 + 0.0j

        for src, r_src in enumerate(parent_positions):
            r_transformed = R @ r_src + t
            found = False
            for dst, r_dst in enumerate(parent_positions):
                L_vec = _match_with_centering(r_transformed - r_dst, centerings, tol)
                if L_vec is not None:
                    mappings_this_op.append((dst, L_vec))
                    if src == dst:
                        chi_p += np.exp(-2j * np.pi * np.dot(kpoint, L_vec))
                    found = True
                    break
            if not found:
                mappings_this_op.append((-1, np.zeros(3)))

        atom_mappings.append(mappings_this_op)
        chi_perm[i_lg] = chi_p

    return atom_mappings, chi_perm, chi_axial


def compute_displacive_characters(R_lk, t_lk, kpoint, positions, tol=1e-4,
                                  centerings=None):
    """
    Computes the character of the mechanical (displacement) representation for each
    operation in the little group.

    χ_disp(g) = Tr(R) · Σ_{fixed atoms} exp(−2πi k·L)

    No det(R) factor — displacements are polar vectors, not axial.

    Parameters
    ----------
    R_lk : array of 3x3 integer matrices
    t_lk : array of 3x1 float vectors
    kpoint : 3-vector
    positions : Nx3 array of atom fractional positions

    Returns
    -------
    chi_disp : np.ndarray, shape (|G_k|,), complex
    """
    chi_disp = np.zeros(len(R_lk), dtype=complex)
    if centerings is None:
        centerings = _NO_CENTERING

    for i, (R, t) in enumerate(zip(R_lk, t_lk)):
        chi_axial = np.trace(R)   # polar vector: no det(R) factor

        trace_perm = 0.0 + 0.0j
        for r in positions:
            L = _match_with_centering(R @ r + t - r, centerings, tol)
            if L is not None:
                trace_perm += np.exp(-2j * np.pi * np.dot(kpoint, L))

        chi_disp[i] = chi_axial * trace_perm

    return chi_disp


def build_displacive_rep_matrices(rotations, translations, kpoint, parent_positions,
                               mapping_little_group, tol=1e-4, centerings=None):
    """
    Build the full 3N×3N D(g) matrices of the mechanical (displacement) representation
    for each little-group operation.

    For operation g = {R | t}, atom src maps to atom dst with lattice shift L:
        R @ r_src + t ≈ r_dst + L
    Block (dst, src):
        D(g)[3*dst:3*dst+3, 3*src:3*src+3] = R * exp(-2πi k·L)

    No det(R) factor — displacements are polar vectors.

    Parameters
    ----------
    rotations : np.ndarray, shape (N_ops, 3, 3)
    translations : np.ndarray, shape (N_ops, 3)
    kpoint : array-like, shape (3,)
    parent_positions : np.ndarray, shape (N_prim, 3)
    mapping_little_group : array-like

    Returns
    -------
    D_matrices : list[np.ndarray]
        One 3N×3N complex matrix per little-group operation.
    """
    N = len(parent_positions)
    kpoint = np.asarray(kpoint, dtype=float)
    if centerings is None:
        centerings = _NO_CENTERING
    D_matrices = []

    for idx in mapping_little_group:
        R = rotations[idx].astype(float)
        t = translations[idx]
        D = np.zeros((3 * N, 3 * N), dtype=complex)

        for src, r_src in enumerate(parent_positions):
            r_transformed = R @ r_src + t
            for dst, r_dst in enumerate(parent_positions):
                L = _match_with_centering(r_transformed - r_dst, centerings, tol)
                if L is not None:
                    phase = np.exp(-2j * np.pi * np.dot(kpoint, L))
                    D[3*dst:3*dst+3, 3*src:3*src+3] = R * phase   # no det_R
                    break

        D_matrices.append(D)

    return D_matrices


def compute_perm_characters_all(rotations, translations, kpoint, positions, tol=1e-4,
                                centerings=None):
    """Return chi_perm indexed over ALL space-group ops (needed for decompose()).

    chi_perm[i] = Σ_{fixed atoms} exp(−2πi k·L_j)  for op i.
    """
    chi = np.zeros(len(rotations), dtype=complex)
    kpt = np.asarray(kpoint, dtype=float)
    if centerings is None:
        centerings = _NO_CENTERING
    for i, (R, t) in enumerate(zip(rotations, translations)):
        for r in positions:
            L = _match_with_centering(R @ r + t - r, centerings, tol)
            if L is not None:
                chi[i] += np.exp(-2j * np.pi * np.dot(kpt, L))
    return chi


def compute_characters(R_lk, t_lk, kpoint, mag_positions, tol=1e-4, centerings=None):
    """
    Computes the character of the magnetic representation for each operation in the little group.
    
    R_lk: Array of 3x3 integer matrices (rotations in real space)
    t_lk: Array of 3x1 float vectors (translations in real space fractional)
    kpoint: 3x1 float vector
    mag_positions: Nx3 array of magnetic atom fractional positions
    
    Returns:
    chi_mag: Array of length |G_k| containing the character for each operation.
    """
    chi_mag = np.zeros(len(R_lk), dtype=complex)
    if centerings is None:
        centerings = _NO_CENTERING

    for i, (R, t) in enumerate(zip(R_lk, t_lk)):
        # Axial vector character: det(R) * Tr(R)
        chi_axial = np.linalg.det(R) * np.trace(R)

        # Permutation character: a magnetic site contributes exp(-2πi k·L)
        # when R r + t ≡ r modulo the PRIMITIVE lattice (L may be a centering
        # vector for centered Bravais lattices).
        trace_perm = 0.0 + 0.0j
        for r in mag_positions:
            L = _match_with_centering(R @ r + t - r, centerings, tol)
            if L is not None:
                trace_perm += np.exp(-2j * np.pi * np.dot(kpoint, L))

        chi_mag[i] = chi_axial * trace_perm

    return chi_mag
